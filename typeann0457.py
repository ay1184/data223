# Python Function Type Annotation Recovery.
# Fine-tunes a general-purpose pretrained code language model (Qwen2.5-Coder-0.5B base, not a
# type-annotation checkpoint) on train.csv inside this run, then generates the slot types of
# every test row from that row's own function text. No test row is used for fitting.
# Usage: python solution.py <public_dir> <submission_csv>
#
# Runtime: not measured end to end (no platform-class GPU was available while writing it).
# Fixed bounds (all in CFG): one model, EPOCHS=2 over at most 43,279 training rows,
# MAX_SRC_TOKENS=512 prompt tokens, MAX_TGT_TOKENS=96, greedy decoding with MAX_NEW_TOKENS=96,
# HOLDOUT=1000 training rows scored after the submission is written.

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import math
import random
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

CFG = dict(
    model="Qwen/Qwen2.5-Coder-0.5B",
    seed=42,
    holdout=1000,
    epochs=2,
    lr=5e-5,
    weight_decay=0.0,
    warmup_frac=0.03,
    min_lr_frac=0.1,
    max_src_tokens=512,
    head_tokens=400,
    max_tgt_tokens=96,
    batch_tokens=6144,
    max_batch_rows=64,
    grad_clip=1.0,
    max_new_tokens=96,
    gen_batch_rows=48,
    log_every=100,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ----------------------------------------------------------------------------------------------
# Type grammar (submission validity) and the official canonicalisation (held-out scoring)
# ----------------------------------------------------------------------------------------------
_TOK_RE = re.compile(r"\.\.\.|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|[\[\],]")
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_MISSING = {"nan", "null", "na", "n/a", "nat", "nil", "<na>", "#n/a", "-nan", "undefined"}
_CAP = {"list": "List", "dict": "Dict", "set": "Set", "frozenset": "FrozenSet", "tuple": "Tuple",
        "type": "Type", "deque": "Deque", "defaultdict": "DefaultDict"}
LIST = "__list__"


def parse_type(s):
    """Parse one type into (name, args) / '...' / (LIST, [types]); None if it breaks the grammar."""
    s = re.sub(r"\s+", "", s)
    if not s or len(s) > 200:
        return None
    toks = _TOK_RE.findall(s)
    if "".join(toks) != s:
        return None
    depth, maxdepth = 0, 0
    for t in toks:
        depth += (t == "[") - (t == "]")
        maxdepth = max(maxdepth, depth)
        if depth < 0:
            return None
    if depth != 0 or maxdepth > 8:
        return None
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def p_type():
        nonlocal pos
        name = peek()
        if name is None or not _NAME_RE.fullmatch(name):
            raise ValueError
        if name.split(".")[-1].lower() in _MISSING:
            raise ValueError
        pos += 1
        if peek() != "[":
            return (name, None)
        pos += 1
        args = [p_arg()]
        while peek() == ",":
            pos += 1
            args.append(p_arg())
        if peek() != "]" or len(args) > 16:
            raise ValueError
        pos += 1
        last = name.split(".")[-1]
        if last == "Optional" and (len(args) != 1 or not isinstance(args[0], tuple) or args[0][0] == LIST):
            raise ValueError
        if last == "Union" and any(not isinstance(a, tuple) or a[0] == LIST for a in args):
            raise ValueError
        return (name, args)

    def p_arg():
        nonlocal pos
        if peek() == "...":
            pos += 1
            return "..."
        if peek() == "[":
            pos += 1
            items = []
            if peek() == "]":
                pos += 1
                return (LIST, items)
            items.append(p_type())
            while peek() == ",":
                pos += 1
                items.append(p_type())
            if peek() != "]" or len(items) > 16:
                raise ValueError
            pos += 1
            return (LIST, items)
        return p_type()

    try:
        tree = p_type()
    except ValueError:
        return None
    if pos != len(toks):
        return None
    return tree


def render(tree):
    if tree == "...":
        return "..."
    name, args = tree
    if name == LIST:
        return "[" + ", ".join(render(a) for a in args) + "]"
    if name == "None":
        name = "NoneType"
    if args is None:
        return name
    return name + "[" + ", ".join(render(a) for a in args) + "]"


def canon(tree):
    """Official canonical form: no module prefixes, typing capitals, Text->str, None, flat unions."""
    if tree == "...":
        return "..."
    name, args = tree
    if name == LIST:
        return "[" + ",".join(canon(a) for a in args) + "]"
    base = name.split(".")[-1]
    base = _CAP.get(base, base)
    if base == "Text":
        base = "str"
    if base == "NoneType":
        base = "None"
    if base in ("Union", "Optional") and args:
        members = []

        def collect(t):
            if t != "..." and t[0] != LIST:
                b = t[0].split(".")[-1]
                if b in ("Union", "Optional") and t[1]:
                    for a in t[1]:
                        collect(a)
                    if b == "Optional":
                        members.append("None")
                    return
            members.append(canon(t))

        for a in args:
            collect(a)
        if base == "Optional":
            members.append("None")
        uniq = sorted(set(members))
        rest = [m for m in uniq if m != "None"]
        if "None" in uniq:
            if not rest:
                return "None"
            inner = rest[0] if len(rest) == 1 else "Union[" + ",".join(rest) + "]"
            return "Optional[" + inner + "]"
        return rest[0] if len(rest) == 1 else "Union[" + ",".join(rest) + "]"
    if args is None:
        return base
    return base + "[" + ",".join(canon(a) for a in args) + "]"


def canon_str(s):
    tree = parse_type(s)
    return canon(tree) if tree is not None else re.sub(r"\s+", "", s)


def slot_credit(ref, pred):
    r, p = canon_str(ref), canon_str(pred)
    if r == p:
        return 1.0
    ropt, popt = r.startswith("Optional["), p.startswith("Optional[")
    rcore = r[9:-1] if ropt else r
    pcore = p[9:-1] if popt else p
    if ropt == popt and rcore.split("[")[0] == pcore.split("[")[0]:
        return 0.6
    if rcore == pcore:
        return 0.3
    return 0.0


def row_score(ref_cell, pred_cell):
    refs, preds = split_cell(ref_cell), split_cell(pred_cell)
    if len(refs) != len(preds):
        return 0.0
    credits = [slot_credit(r, p) for r, p in zip(refs, preds)]
    return 0.5 * float(np.mean(credits)) + 0.5 * float(all(c == 1.0 for c in credits))


def mean_score(ref_cells, pred_cells):
    return float(np.mean([row_score(r, p) for r, p in zip(ref_cells, pred_cells)]))


def split_cell(cell):
    return [t.strip() for t in str(cell).split(";")]


# ----------------------------------------------------------------------------------------------
# Fallback prior (second path) and output sanitising
# ----------------------------------------------------------------------------------------------
def slot_kinds(slots):
    return ["return" if s == "return" else "param" for s in slots.split()]


def fit_fallback(train):
    """Most frequent training type per slot kind; used for invalid or missing generated types."""
    cnt = {"param": Counter(), "return": Counter()}
    for slots, types in zip(train["slots"], train["types"]):
        for k, t in zip(slot_kinds(slots), split_cell(types)):
            cnt[k][t] += 1
    out = {}
    for k, c in cnt.items():
        for t, _ in c.most_common():
            tree = parse_type(t)
            if tree is not None:
                out[k] = render(tree)
                break
    return out


def fallback_cell(slots, fb):
    return " ; ".join(fb[k] for k in slot_kinds(slots))


def clean_type(t, fallback):
    tree = parse_type(t)
    if tree is not None:
        return render(tree)
    m = _NAME_RE.match(re.sub(r"\s+", "", t))
    if m:
        tree = parse_type(m.group(0))
        if tree is not None:
            return render(tree)
    return fallback


def clean_cell(text, slots, fb):
    kinds = slot_kinds(slots)
    parts = [p for p in split_cell(text) if p]
    parts = parts[: len(kinds)]
    out = [clean_type(p, fb[k]) for p, k in zip(parts, kinds)]
    out += [fb[k] for k in kinds[len(out):]]
    return " ; ".join(out)


def write_submission(path, ids, cells):
    df = pd.DataFrame({"id": list(ids), "types": list(cells)})
    df.to_csv(path, index=False, lineterminator="\n")
    print(f"wrote {len(df)} rows -> {path}", flush=True)


# ----------------------------------------------------------------------------------------------
# Prompting, batching, training, generation
# ----------------------------------------------------------------------------------------------
def encode_prompts(tok, functions, slots, cfg):
    code_ids = tok(list(functions), add_special_tokens=False)["input_ids"]
    suffix = ["\n### slots: " + s + "\n### types:" for s in slots]
    suf_ids = tok(suffix, add_special_tokens=False)["input_ids"]
    head = cfg["head_tokens"]
    tail = cfg["max_src_tokens"] - head
    out = []
    for c, s in zip(code_ids, suf_ids):
        if len(c) > head + tail:
            c = c[:head] + c[-tail:]
        out.append(c + s)
    return out


def encode_targets(tok, types, cfg):
    ids = tok([" " + t for t in types], add_special_tokens=False)["input_ids"]
    return [t[: cfg["max_tgt_tokens"]] + [tok.eos_token_id] for t in ids]


def make_batches(lengths, cfg, rng=None):
    order = np.argsort(np.asarray(lengths), kind="stable")
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = lengths[i]
        new_max = max(cur_max, L)
        if cur and (new_max * (len(cur) + 1) > cfg["batch_tokens"] or len(cur) >= cfg["max_batch_rows"]):
            batches.append(cur)
            cur, new_max = [], L
        cur.append(int(i))
        cur_max = new_max
    if cur:
        batches.append(cur)
    if rng is not None:
        batches = [batches[j] for j in rng.permutation(len(batches))]
    return batches


def load_lm(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"])
    model = model.float().to(DEVICE)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok, model


def train_lm(model, tok, prompts, targets, cfg):
    seqs = [p + t for p, t in zip(prompts, targets)]
    starts = [len(p) for p in prompts]
    lengths = [len(s) for s in seqs]
    rng = np.random.RandomState(cfg["seed"])
    n_batches = len(make_batches(lengths, cfg))
    total = n_batches * cfg["epochs"]
    warm = max(1, int(total * cfg["warmup_frac"]))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=cfg["weight_decay"])

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total - warm)
        return cfg["min_lr_frac"] + (1 - cfg["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    body, head = model.base_model, model.get_output_embeddings()
    pad = tok.pad_token_id
    model.config.use_cache = False
    model.train()
    step, run_loss, run_n, nonfinite = 0, 0.0, 0, 0
    print(f"training: {len(seqs)} rows, {n_batches} batches/epoch, {total} steps", flush=True)
    for ep in range(cfg["epochs"]):
        for b in make_batches(lengths, cfg, rng):
            L = max(lengths[i] for i in b)
            ids = torch.full((len(b), L), pad, dtype=torch.long)
            att = torch.zeros((len(b), L), dtype=torch.long)
            lab = torch.zeros((len(b), L), dtype=torch.bool)
            for r, i in enumerate(b):
                s = seqs[i]
                ids[r, : len(s)] = torch.tensor(s)
                att[r, : len(s)] = 1
                lab[r, starts[i]: len(s)] = True
            ids, att, lab = ids.to(DEVICE), att.to(DEVICE), lab.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                h = body(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
                sel = lab[:, 1:]
                logits = head(h[:, :-1][sel])
            loss = F.cross_entropy(logits.float(), ids[:, 1:][sel])
            if not torch.isfinite(loss):
                nonfinite += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run_loss += float(loss.detach())
            run_n += 1
            if step % cfg["log_every"] == 0 or step == total:
                print(f"ep {ep} step {step}/{total} loss {run_loss / run_n:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} nonfinite {nonfinite}", flush=True)
                run_loss, run_n = 0.0, 0
    print(f"training done, non-finite losses: {nonfinite}", flush=True)
    del opt
    model.eval()
    return model


@torch.no_grad()
def generate(model, tok, prompts, cfg):
    pad = tok.pad_token_id
    lengths = [len(p) for p in prompts]
    order = np.argsort(np.asarray(lengths), kind="stable")
    outs = [""] * len(prompts)
    bs = cfg["gen_batch_rows"]
    for k in range(0, len(order), bs):
        b = [int(i) for i in order[k: k + bs]]
        L = max(lengths[i] for i in b)
        ids = torch.full((len(b), L), pad, dtype=torch.long)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for r, i in enumerate(b):
            p = prompts[i]
            ids[r, L - len(p):] = torch.tensor(p)
            att[r, L - len(p):] = 1
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            gen = model.generate(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE),
                                 max_new_tokens=cfg["max_new_tokens"], do_sample=False, num_beams=1,
                                 eos_token_id=tok.eos_token_id, pad_token_id=pad, use_cache=True)
        for r, i in enumerate(b):
            outs[i] = tok.decode(gen[r, L:].tolist(), skip_special_tokens=True).strip()
    return outs


def main(argv, cfg=CFG):
    data_dir, out_path = argv[1], argv[2]
    set_seed(cfg["seed"])
    print(f"python {sys.version.split()[0]} torch {torch.__version__} numpy {np.__version__} "
          f"pandas {pd.__version__} device {DEVICE}", flush=True)
    rd = dict(keep_default_na=False, na_filter=False, dtype=str)
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), **rd)
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), **rd)
    print(f"train {train.shape} test {test.shape}", flush=True)

    # 1) valid fallback submission before any training
    fb = fit_fallback(train)
    print(f"fallback types: {fb}", flush=True)
    write_submission(out_path, test["id"], [fallback_cell(s, fb) for s in test["slots"]])

    # 2) held-out rows of the training file (scored after the submission is written)
    perm = np.random.RandomState(cfg["seed"]).permutation(len(train))
    n_hold = min(cfg["holdout"], len(train) // 5)
    hold = train.iloc[np.sort(perm[:n_hold])].reset_index(drop=True)
    fit = train.iloc[np.sort(perm[n_hold:])].reset_index(drop=True)
    assert not set(hold["id"]) & set(fit["id"])

    # 3) fine-tune the pretrained code model on the fit rows
    tok, model = load_lm(cfg)
    fit_prompts = encode_prompts(tok, fit["function"], fit["slots"], cfg)
    fit_targets = encode_targets(tok, fit["types"], cfg)
    model = train_lm(model, tok, fit_prompts, fit_targets, cfg)
    model = model.to(torch.bfloat16)
    model.config.use_cache = True

    # 4) test predictions, one row's prompt from that row's own function text
    test_prompts = encode_prompts(tok, test["function"], test["slots"], cfg)
    raw = generate(model, tok, test_prompts, cfg)
    cells = [clean_cell(t, s, fb) for t, s in zip(raw, test["slots"])]
    n_fixed = sum(clean_cell(t, s, fb) != " ; ".join(split_cell(t)) for t, s in zip(raw, test["slots"]))
    print(f"test rows repaired by the sanitiser: {n_fixed}/{len(cells)}", flush=True)
    write_submission(out_path, test["id"], cells)

    # 5) validation on held-out training rows with the official metric
    hold_prompts = encode_prompts(tok, hold["function"], hold["slots"], cfg)
    hraw = generate(model, tok, hold_prompts, cfg)
    hcells = [clean_cell(t, s, fb) for t, s in zip(hraw, hold["slots"])]
    s_model = mean_score(hold["types"], hcells)
    s_fb = mean_score(hold["types"], [fallback_cell(s, fb) for s in hold["slots"]])
    print(f"HOLDOUT ({len(hold)} rows, random split, optimistic vs project split): "
          f"model {s_model:.4f}  fallback prior {s_fb:.4f}", flush=True)
    for i in range(5):
        print(f"  {hold['slots'][i]} | ref {hold['types'][i]} | pred {hcells[i]}", flush=True)


if __name__ == "__main__":
    main(sys.argv)
