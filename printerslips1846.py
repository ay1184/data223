"""Register repair for the Tolmie / Dawson British Columbia vocabularies.

Learned component (trained in this script, on the provided training data):
  MatchNet, a neural pair scorer. For two words it builds the symbol-by-symbol match matrix
  (exact symbol identity + learned symbol-embedding similarity) and reads it with a small CNN into
  one logit. It is trained on the reference columns of the training varieties: for a word of one
  variety and all the words of a sister variety on the same page, a cross-entropy picks the sister
  word of the same English row (the test condition: compare words within a page).
Decode (selects from the learned evidence only):
  for every pair of columns of a sheet the MatchNet logits of all their cells are turned into
  row + column log-softmax evidence; for a column, the evidence that listed cell i belongs to row j
  is the sum over the other columns of the evidence against their cell currently at row j.
  A beam search over compositions of up to 4 faults of the published law (cyclic shift of 1-3 over
  5-18 cells or the whole column, reversal of 4-14, exchange of neighbouring pairs over 4-10)
  finds each column's best repair, with a penalty LAM per fault; a greedy joint ascent repairs, at
  each step, only the column whose repair gains most given the others' current rows. Cells left in
  place that an independent decode of their column (penalty LAM2) moves are then moved too: with one
  sister the page cannot tell which of two misaligned columns slipped, so both readings are kept.
LAM and LAM2 are chosen inside this script from a small fixed grid, by the competition metric on
test-like validation sheets (2-4 columns, mostly two families with one sister each) cut from the
training sheets of folds 1 and 2, scored by a pair model trained on fold 0 only (both families
unseen); the final pair model is then trained on all training varieties and applied to the test.
Entries keep their listed column (crossings are not repaired by this version).

Usage: python solution.py <data_dir> <submission_csv>

Runtime (measured, CPU only, 4 threads, no GPU, this exact file with default CFG):
  537 s end to end (validation pair model + LAM grid + final pair model + test), exit code 0.
  Not run on a GPU.
"""
import sys
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = dict(
    seed=1234,
    emb=32, hid=16, nlayers=2, maxlen=16,
    s1_steps=1000, s1_batch=12, s1_lr=2e-3, s1_wd=0.01, s1_tokdrop=0.05, m_anchors=3,
    beam=8, depth=4, max_steps=12,
    lam_grid=(4.0, 8.0), lam2_grid=(8.0, 16.0),
    val_train_fold=0, val_folds=(1, 2), val_per_page=24,
    # runtime measured: 537 s on CPU (4 threads), exit 0; not run on a GPU
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def rownum(code):
    return int(code.split("_", 1)[0])


# ----------------------------------------------------------------------------- data
def load_data(data_dir):
    tr = pd.read_csv(f"{data_dir}/train.csv")
    te = pd.read_csv(f"{data_dir}/test.csv")
    tg = pd.read_csv(f"{data_dir}/train_targets.csv")
    si = pd.read_csv(f"{data_dir}/sheet_index.csv")
    rc = [json.loads(l) for l in open(f"{data_dir}/reference_columns.jsonl", encoding="utf-8")]
    P = tg.prediction.map(json.loads)
    for k in ["row", "column", "kind", "reach", "crossed", "block"]:
        tg["g_" + k] = P.map(lambda d: d[k])
    tr = tr.merge(tg.drop(columns=["prediction"]), on="target_id", how="left")
    tr = tr.merge(si[["case_id", "page", "fold"]], on="case_id", how="left")
    te = te.merge(si[["case_id", "page", "fold"]], on="case_id", how="left")
    fam = {r["column"]: r["family"] for r in rc}
    cols = tr.groupby("case_id").g_column.apply(lambda s: sorted(set(s))).to_dict()
    tr["g_sisters"] = [sum(1 for c in cols[cid] if c != gc and fam[c] == fam[gc])
                       for cid, gc in zip(tr.case_id, tr.g_column)]
    return tr, te, rc


def build_vocab(rc, tr):
    toks = set()
    for r in rc:
        for c in r["cells"]:
            for s in c["segments"]:
                toks.update(s.split())
    for s in tr.segments:
        toks.update(s.split())
    return {t: i + 2 for i, t in enumerate(sorted(toks))}  # 0 pad, 1 unknown


def encode_seg(seg, vocab, maxlen):
    ids = [vocab.get(t, 1) for t in seg.split()][:maxlen]
    return ids if ids else [1]


def page_of_rows(tr):
    """row code -> page, from the training sheets (a page is a fixed section of the list)."""
    m = tr.drop_duplicates("listed_row").set_index("listed_row").page.to_dict()
    bounds = tr.assign(n=tr.listed_row.map(rownum)).groupby("page").n.agg(["min", "max"])
    return m, bounds


def page_lookup(row, m, bounds):
    if row in m:
        return int(m[row])
    n = rownum(row)
    for p, b in bounds.iterrows():
        if b["min"] <= n <= b["max"]:
            return int(p)
    d = (bounds["min"] - n).abs().combine((bounds["max"] - n).abs(), min)
    return int(d.idxmin())


def pad_batch(seqs):
    L = max(len(s) for s in seqs)
    out = np.zeros((len(seqs), L), dtype=np.int64)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return torch.from_numpy(out)


# ----------------------------------------------------------------------------- learned pair scorer
class MatchNet(nn.Module):
    """Pair scorer: a CNN over the symbol-by-symbol match matrix of two words
    (exact symbol identity + learned symbol similarity), pooled to one logit."""

    def __init__(self, nvocab, cfg):
        super().__init__()
        d = cfg["emb"]
        h = cfg["hid"]
        self.emb = nn.Embedding(nvocab, d, padding_idx=0)
        self.c0 = nn.Conv2d(3, h, 3, padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(h, h, 3, padding=d_, dilation=d_) for d_ in (1, 2, 1)][:cfg["nlayers"]])
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))
        self.d = d

    def forward(self, a, b):  # a: (P, La), b: (P, Lb) long
        ma = (a > 0).float()
        mb = (b > 0).float()
        m = ma[:, :, None] * mb[:, None, :]
        ea = self.emb(a)
        eb = self.emb(b)
        sim = torch.einsum("pid,pjd->pij", ea, eb) / np.sqrt(self.d)
        same = ((a[:, :, None] == b[:, None, :]) & (a[:, :, None] > 1)).float()
        x = torch.stack([same * m, sim * m, m], 1)
        h = F.gelu(self.c0(x))
        for conv in self.convs:
            h = h + F.gelu(conv(h))
        h = h * m[:, None]
        mx = (h + (m[:, None] - 1) * 1e4).flatten(2).max(2).values
        mean = h.flatten(2).sum(2) / m.flatten(1).sum(1, keepdim=True).clamp(min=1)
        return self.head(torch.cat([mx, mean], 1)).squeeze(1)


def build_pair_pool(rc, folds, vocab, pmap, pbounds, cfg):
    """(A, B, page) triples of sister varieties (same family) in the given folds."""
    cols = [r for r in rc if r["fold"] in folds]
    pages = {}
    for r in cols:
        d = {}
        for c in r["cells"]:
            p = page_lookup(c["row"], pmap, pbounds)
            d.setdefault(p, []).append((c["row"], [encode_seg(s, vocab, cfg["maxlen"]) for s in c["segments"]]))
        pages[r["column"]] = d
    triples = []
    for a in cols:
        for b in cols:
            if a["column"] == b["column"] or a["family"] != b["family"]:
                continue
            for p in pages[a["column"]]:
                if p not in pages[b["column"]]:
                    continue
                A = pages[a["column"]][p]
                B = pages[b["column"]][p]
                brow = {row: k for k, (row, _) in enumerate(B)}
                tgt = [brow.get(row, -1) for row, _ in A]
                if sum(t >= 0 for t in tgt) >= 3:
                    triples.append((A, B, tgt))
    return triples


def train_match(rc, folds, vocab, pmap, pbounds, cfg, seed):
    set_seed(seed)
    triples = build_pair_pool(rc, folds, vocab, pmap, pbounds, cfg)
    model = MatchNet(len(vocab) + 2, cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["s1_lr"], weight_decay=cfg["s1_wd"])
    steps = cfg["s1_steps"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["s1_lr"], total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(seed)
    model.train()
    for step in range(steps):
        idx = rng.choice(len(triples), size=cfg["s1_batch"], replace=False)
        la, lb, groups = [], [], []
        for t in idx:
            A, B, tgt = triples[t]
            valid = [i for i, x in enumerate(tgt) if x >= 0]
            anchors = rng.choice(valid, size=min(cfg["m_anchors"], len(valid)), replace=False)
            bseqs = [syn[rng.integers(len(syn))] for _, syn in B]
            flip = rng.random() < 0.5
            for i in anchors:
                aseq = A[i][1][rng.integers(len(A[i][1]))]
                start = len(la)
                for bsq in bseqs:
                    if flip:
                        la.append(bsq)
                        lb.append(aseq)
                    else:
                        la.append(aseq)
                        lb.append(bsq)
                groups.append((start, len(bseqs), tgt[i]))
        if cfg["s1_tokdrop"] > 0:
            la = [[(1 if rng.random() < cfg["s1_tokdrop"] else v) for v in s] for s in la]
            lb = [[(1 if rng.random() < cfg["s1_tokdrop"] else v) for v in s] for s in lb]
        s = model(pad_batch(la).to(DEVICE), pad_batch(lb).to(DEVICE))
        loss = 0.0
        for start, n, t in groups:
            loss = loss + F.cross_entropy(s[start:start + n][None], torch.tensor([t], device=DEVICE))
        loss = loss / len(groups)
        if not torch.isfinite(loss):
            raise SystemExit("non-finite pair-model loss at step %d" % step)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def match_matrix(model, seqs, cols_of, bs=8192):
    """Symmetric logit matrix between all entries of a sheet that sit in different columns."""
    N = len(seqs)
    ii, jj = np.triu_indices(N, 1)
    keep = cols_of[ii] != cols_of[jj]
    ii, jj = ii[keep], jj[keep]
    M = np.full((N, N), -1e4, dtype=np.float32)
    out = []
    for s in range(0, len(ii), bs):
        a = pad_batch([seqs[i] for i in ii[s:s + bs]]).to(DEVICE)
        b = pad_batch([seqs[j] for j in jj[s:s + bs]]).to(DEVICE)
        out.append(0.5 * (model(a, b) + model(b, a)).float().cpu().numpy())
    if out:
        v = np.concatenate(out)
        if not np.isfinite(v).all():
            raise SystemExit("non-finite pair-model output")
        M[ii, jj] = v
        M[jj, ii] = v
    return M


# ----------------------------------------------------------------------------- sheets
def build_sheets(df, model, vocab, cfg):
    """Group the entries of every sheet into columns and cells (cells ordered by listed row) and
    score every pair of cells of different columns with the pair model (max over synonyms)."""
    sheets = []
    df = df.reset_index(drop=True)
    for case, s in df.groupby("case_id", sort=True):
        cols = sorted(s.column.unique())
        colinfo = []
        order, starts, ncell = [], [], []
        for c in cols:
            sc = s[s.column == c]
            rows = sorted(sc.listed_row.unique(), key=rownum)
            cells = []
            for r in rows:
                e = sc.index[sc.listed_row == r].to_numpy()
                cells.append(e)
                starts.append(len(order))
                order.extend(e.tolist())
            ncell.append(len(cells))
            colinfo.append(dict(column=c, rows=rows, cells=cells))
        seqs = [encode_seg(df.segments.iat[e], vocab, cfg["maxlen"]) for e in order]
        cols_of = np.concatenate([[a] * sum(len(e) for e in colinfo[a]["cells"]) for a in range(len(cols))])
        M = match_matrix(model, seqs, cols_of)
        st = np.array(starts)
        M = np.maximum.reduceat(np.maximum.reduceat(M, st, axis=0), st, axis=1).astype(np.float32)
        off = np.concatenate([[0], np.cumsum(ncell)])
        S = {}
        for a in range(len(cols)):
            for b in range(len(cols)):
                if a != b:
                    S[(a, b)] = M[off[a]:off[a + 1], off[b]:off[b + 1]]
        sheets.append(dict(case_id=case, cols=colinfo, S=S))
    return sheets


def log_softmax_np(x, axis):
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    return y - np.log(np.exp(y).sum(axis=axis, keepdims=True))


def evidence_mats(sheet):
    """Row + column log-softmax of every column-pair logit matrix (learned pair evidence)."""
    if "Z" not in sheet:
        Zd = {}
        for key, M in sheet["S"].items():
            Zd[key] = (log_softmax_np(M, 1) + log_softmax_np(M, 0) + np.log(M.shape[0] * M.shape[1])) / 2
        sheet["Z"] = Zd
    return sheet["Z"]


def column_evidence(sheet, a, assign):
    """E[i, j]: summed evidence that listed cell i of column a belongs to row j, from every other
    column with its cells in their current rows."""
    cols = sheet["cols"]
    ca = cols[a]
    n = len(ca["rows"])
    Zd = evidence_mats(sheet)
    E = np.zeros((n, n), dtype=np.float64)
    for b in range(len(cols)):
        if b == a:
            continue
        cb = cols[b]
        Z = Zd[(a, b)]
        rows_b = [cb["rows"][k] for k in assign[b]]
        where = {r: k for k, r in enumerate(rows_b)}
        kidx = np.array([where.get(r, -1) for r in ca["rows"]])
        has = kidx >= 0
        E += np.where(has[None, :], Z[:, np.where(has, kidx, 0)], 0.0)
    return E


# ----------------------------------------------------------------------------- structured decode
_OPS = {}


def block_ops(n):
    """All single faults of the law on n cells, as position maps sigma (listed pos -> pos)."""
    if n in _OPS:
        return _OPS[n]
    ident = np.arange(n)
    seen = {tuple(ident)}
    ops = []

    def add(sig):
        t = tuple(sig)
        if t not in seen:
            seen.add(t)
            ops.append(sig)

    lens = sorted(set([L for L in range(5, 19) if L <= n] + ([n] if n >= 2 else [])))
    for L in lens:
        k = np.arange(L)
        for st in range(0, n - L + 1):
            for s in (1, 2, 3, -1, -2, -3):
                if abs(s) >= L:
                    continue
                sig = ident.copy()
                sig[st:st + L] = st + (k + s) % L
                add(sig)
    for L in range(4, min(14, n) + 1):
        for st in range(0, n - L + 1):
            sig = ident.copy()
            sig[st:st + L] = sig[st:st + L][::-1]
            add(sig)
    for L in range(4, min(10, n) + 1):
        for st in range(0, n - L + 1):
            sig = ident.copy()
            for p in range(0, L - 1, 2):
                sig[st + p], sig[st + p + 1] = sig[st + p + 1], sig[st + p]
            add(sig)
    arr = np.array(ops, dtype=np.int64).reshape(-1, n)
    _OPS[n] = arr
    return arr


def beam_decode(lp, beam, depth, lam):
    """Best composition of up to `depth` faults: max sum_i lp[i, t(i)] - lam * #faults.
    Returns (t, #faults, objective)."""
    n = lp.shape[0]
    ident = np.arange(n)
    ops = block_ops(n)
    ar = np.arange(n)[None, :]
    base = float(lp[np.arange(n), ident].sum())
    best = (base, ident, 0)
    states = [(base, ident)]
    if len(ops) == 0:
        return ident, 0, base
    for d in range(depth):
        cands = []
        for _, t in states:
            T = t[ops]
            vals = lp[ar, T].sum(1) - lam * (d + 1)
            top = np.argpartition(-vals, min(beam, len(vals) - 1))[:beam]
            for k in top:
                cands.append((float(vals[k]), T[k]))
        cands.sort(key=lambda x: -x[0])
        uniq, seen = [], set()
        for v, t in cands:
            key = t.tobytes()
            if key in seen:
                continue
            seen.add(key)
            uniq.append((v, t))
            if len(uniq) >= beam:
                break
        states = uniq
        if states and states[0][0] > best[0]:
            best = (states[0][0], states[0][1], d + 1)
    return best[1], best[2], best[0]


def greedy_joint(sheet, cfg, lam):
    """Joint ascent over the columns of a sheet: at every step only the column whose re-decoding
    gains most (given the others' current rows) is updated, until no column gains."""
    cols = sheet["cols"]
    C = len(cols)
    assign = [np.arange(len(c["rows"])) for c in cols]
    nops = [0] * C
    if C < 2:
        return assign
    for step in range(cfg["max_steps"]):
        best = None
        for a in range(C):
            n = len(cols[a]["rows"])
            if n < 2:
                continue
            E = column_evidence(sheet, a, assign)
            t, k, obj = beam_decode(E, cfg["beam"], cfg["depth"], lam)
            cur = float(E[np.arange(n), assign[a]].sum()) - lam * nops[a]
            gain = obj - cur
            if gain > 1e-6 and not np.array_equal(t, assign[a]) and (best is None or gain > best[0]):
                best = (gain, a, t, k)
        if best is None:
            break
        _, a, t, k = best
        assign[a] = t
        nops[a] = k
    return assign


def union_moves(sheet, assign, cfg, lam2):
    """Cells that the joint repair left in place but that an independent decode of their column
    (against the listed sisters) moves, are moved as that decode says. With one sister the page cannot
    tell which of two misaligned columns slipped, so both readings are kept."""
    cols = sheet["cols"]
    ident = [np.arange(len(c["rows"])) for c in cols]
    out = [a.copy() for a in assign]
    for a in range(len(cols)):
        n = len(cols[a]["rows"])
        if n < 2 or len(cols) < 2:
            continue
        E = column_evidence(sheet, a, ident)
        t, _, _ = beam_decode(E, cfg["beam"], cfg["depth"], lam2)
        for i in range(n):
            if assign[a][i] == i and t[i] != i:
                out[a][i] = t[i]
    return out


def decode(sheet, cfg, lam, lam2):
    return union_moves(sheet, greedy_joint(sheet, cfg, lam), cfg, lam2)


def predict_rows(df, sheets, assigns):
    pr = np.array(df.listed_row.values, dtype=object)
    pc = np.array(df.column.values, dtype=object)
    for sh, asg in zip(sheets, assigns):
        for c, a in zip(sh["cols"], asg):
            for i, e in enumerate(c["cells"]):
                pr[e] = c["rows"][a[i]]
    return pr, pc


# ----------------------------------------------------------------------------- metric (published)
def metric(g, pr, pc):
    """The competition score as published in the Overview, on labelled training sheets."""
    disp = (g.g_row.values != g.listed_row.values) | (g.g_column.values != g.column.values)
    right = (pr == g.g_row.values) & (pc == g.g_column.values)
    moved = (pr != g.listed_row.values) | (pc != g.column.values)
    guard = right[~disp].mean() if (~disp).any() else 1.0

    def u_of(V, Vref):
        S = np.clip((V - Vref) / (1 - Vref), 0, 1) if Vref < 1 else 0.0
        return (S + 0.25 * V) / 1.25

    kind, reach, crossed = g.g_kind.values, g.g_reach.values, g.g_crossed.values
    block, sis = g.g_block.values, g.g_sisters.values
    groups = [
        [disp & (kind == k) for k in ["shift", "reverse", "swap", "cross", "compound"]],
        [disp & (reach == 1), disp & (reach >= 2) & (reach <= 5), disp & (reach >= 6),
         disp & (reach == 0) & (crossed == 1)],
        [disp & (block <= 10), disp & (block >= 11) & (block <= 14), disp & (block >= 15)],
        [disp & (sis >= 2), disp & (sis == 1), disp & (sis == 0)],
    ]
    axes = []
    for gs in groups:
        us = [u_of(0.5 * right[m].mean() + 0.5 * guard, 0.5) for m in gs if m.sum() > 0]
        axes.append(float(np.mean(us)))
    tp = (moved & disp).sum()
    prec = tp / moved.sum() if moved.sum() else 0.0
    rec = tp / disp.sum() if disp.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    axes.append(float(u_of(f1, 0.0)))
    key = g.case_id.values + "|" + g.g_column.values
    agg = pd.DataFrame({"k": key, "r": right, "rr": ~disp}).groupby("k").agg(
        n=("r", "size"), a=("r", "all"), ar=("rr", "all"))
    V = (agg.n * agg.a).sum() / agg.n.sum()
    Vr = (agg.n * agg.ar).sum() / agg.n.sum()
    axes.append(float(u_of(V, Vr)))
    return float(np.clip(100 * np.mean(axes), 0.01, 100))


# ----------------------------------------------------------------------------- test-like validation
def build_mixed(tr, rc, folds, nper, seed):
    """Validation sheets shaped like the test: 2-4 columns, mostly two families with one sister each,
    cut from training sheets of the same page of two folds (whole families the pair model never saw)."""
    rng = np.random.default_rng(seed)
    fam = {r["column"]: r["family"] for r in rc}
    by = {}
    for case, f, p in tr[["case_id", "fold", "page"]].drop_duplicates().itertuples(index=False):
        by.setdefault((int(f), int(p)), []).append(case)
    for k in by:
        by[k] = sorted(by[k])
    colsof = tr.groupby("case_id").column.apply(lambda s: sorted(set(s))).to_dict()
    groups = {case: s for case, s in tr.groupby("case_id")}

    def pick(case, n, need_sis):
        cols = colsof[case]
        byf = {}
        for c in cols:
            byf.setdefault(fam[c], []).append(c)
        big = [v for v in byf.values() if len(v) >= n]
        if need_sis and big:
            v = big[rng.integers(len(big))]
            return list(rng.choice(v, size=n, replace=False))
        return list(rng.choice(cols, size=min(n, len(cols)), replace=False))

    parts = []
    pages = sorted(set(p for (_, p) in by))
    for page in pages:
        for t in range(nper):
            X = by[(folds[0], page)][rng.integers(len(by[(folds[0], page)]))]
            Y = by[(folds[1], page)][rng.integers(len(by[(folds[1], page)]))]
            kind = ["2+2", "2+2", "2+2", "2+1", "3", "2"][rng.integers(6)]
            if kind == "2+2":
                keep = [(X, pick(X, 2, True)), (Y, pick(Y, 2, True))]
            elif kind == "2+1":
                keep = [(X, pick(X, 2, True)), (Y, pick(Y, 1, False))]
            else:
                Z = X if rng.random() < 0.5 else Y
                keep = [(Z, pick(Z, 3 if kind == "3" else 2, True))]
            s = pd.concat([groups[c][groups[c].column.isin(k) & groups[c].g_column.isin(k)] for c, k in keep]).copy()
            s["case_id"] = "mix_%d_%d" % (page, t)
            kc = set(s.g_column)
            s["g_sisters"] = [sum(1 for c in kc if c != gc and fam[c] == fam[gc]) for gc in s.g_column]
            parts.append(s)
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- pipeline
def main(data_dir, out_path):
    cfg = CFG
    set_seed(cfg["seed"])
    tr, te, rc = load_data(data_dir)
    vocab = build_vocab(rc, tr)
    pmap, pbounds = page_of_rows(tr)

    # 1) choose (LAM, LAM2) on test-like sheets of two families the pair model did not see
    m = train_match(rc, [cfg["val_train_fold"]], vocab, pmap, pbounds, cfg, cfg["seed"])
    mix = build_mixed(tr, rc, cfg["val_folds"], cfg["val_per_page"], cfg["seed"]).reset_index(drop=True)
    vs = build_sheets(mix, m, vocab, cfg)
    print("validation sheets", len(vs), "entries", len(mix), flush=True)
    best = None
    for lam in cfg["lam_grid"]:
        greedy = [greedy_joint(sh, cfg, lam) for sh in vs]
        for lam2 in cfg["lam2_grid"]:
            pr, pc = predict_rows(mix, vs, [union_moves(sh, a, cfg, lam2) for sh, a in zip(vs, greedy)])
            sc = metric(mix, pr, pc)
            print("LAM", lam, "LAM2", lam2, "test-like validation score", round(sc, 3), flush=True)
            if best is None or sc > best[0]:
                best = (sc, lam, lam2)
    _, lam, lam2 = best
    print("chosen LAM", lam, "LAM2", lam2, flush=True)

    # 2) final pair model on all training varieties, applied to the test sheets
    m = train_match(rc, [0, 1, 2], vocab, pmap, pbounds, cfg, cfg["seed"] + 7)
    te = te.reset_index(drop=True)
    sheets = build_sheets(te, m, vocab, cfg)
    pr, pc = predict_rows(te, sheets, [decode(sh, cfg, lam, lam2) for sh in sheets])
    sub = pd.DataFrame({"target_id": te.target_id,
                        "prediction": [json.dumps({"row": r, "column": c}) for r, c in zip(pr, pc)]})
    if len(sub) != len(te) or sub.target_id.duplicated().any():
        raise SystemExit("submission rows do not match the test entries")
    sub.to_csv(out_path, index=False)
    print("test entries", len(te), "moved", int((pr != te.listed_row.values).sum()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
