"""Regulatory junction and connector reconstruction from biochemical diagram images.
Pipeline: ink mask against the local background, entity outlines from enclosed regions and tinted fills,
catalysis circle detection, skeleton tracing of control and reaction connectors, arrowhead test for outputs.
"""
import math, json
import numpy as np, cv2
from skimage.morphology import skeletonize

INK_T = 215
OFF8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class Cand:
    def __init__(self, cnt, kind):
        self.cnt = cnt
        x, y, w, h = cv2.boundingRect(cnt)
        self.kind = kind
        self.rect = (x, y, w, h)
        self.box = [x - 0.5, y - 0.5, x + w - 0.5, y + h - 0.5]
        self.area = cv2.contourArea(cnt)


def detect_shapes(img):
    H, W = img.shape[:2]
    mn = img.min(axis=2)
    bgc = np.stack([cv2.medianBlur(np.ascontiguousarray(img[:, :, k]), 21) for k in range(3)], 2)
    diff = np.abs(img.astype(np.int16) - bgc.astype(np.int16)).max(axis=2)
    ink = ((diff > 45) & (mn < 235)).astype(np.uint8)
    # holes: enclosed background regions
    bg = 1 - ink
    n, lab, st, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
    holes = []
    circles = []
    region = np.zeros((H, W), np.int32)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 8 or h < 8:
            continue
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if w * h > 0.4 * H * W:
            continue
        m = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea) + np.array([x, y])
        fa = cv2.contourArea(c)
        fill = fa / float(w * h)
        if 6 <= w <= 72 and 6 <= h <= 72 and abs(w - h) <= max(2, 0.15 * max(w, h)) and 0.6 < fill < 0.86 and a > 0.85 * fa:
            circles.append((x + (w - 1) / 2.0, y + (h - 1) / 2.0, (w + h) / 4.0))
            continue
        if w < 16 or h < 12:
            continue
        holes.append((c, fill, (x, y, w, h)))
    cands = []
    for c, fill, (x, y, w, h) in holes:
        # grow contour by ~1.5px to the border centreline
        cc = Cand(c, 'hole')
        cc.box = [x - 1.5, y - 1.5, x + w + 0.5, y + h + 0.5]
        cc.fill = fill
        cands.append(cc)
    # tinted fills (complex enclosures, filled shapes)
    tint = ((ink == 0) & (mn < 250)).astype(np.uint8)
    tint = cv2.morphologyEx(tint, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(tint, connectivity=8)
    for i in range(1, n2):
        x, y, w, h, a = st2[i]
        if w < 20 or h < 14 or a < 150:
            continue
        if w * h > 0.4 * H * W:
            continue
        m = (lab2[y:y + h, x:x + w] == i).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea) + np.array([x, y])
        cc = Cand(c, 'tint')
        cc.box = [x - 1.0, y - 1.0, x + w, y + h]
        cc.fill = a / float(w * h)
        cands.append(cc)
    return ink, cands, circles


def stroke_width(ink):
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    sk = skeletonize(ink > 0)
    v = dt[sk]
    v = v[(v > 0) & (v < 6)]
    return float(2 * np.median(v)) if len(v) else 2.0


def entity_like(c, H, W):
    x, y, w, h = c.rect
    hull = cv2.convexHull(c.cnt)
    ha = cv2.contourArea(hull)
    sol = c.area / ha if ha > 0 else 0
    c.solidity = sol
    return sol > 0.93 and c.fill > 0.7 and w < 0.6 * W and h < 0.6 * H


def build_line_mask(ink, cands, H, W, sw):
    ring = np.zeros((H, W), np.uint8)
    fillm = np.zeros((H, W), np.uint8)
    ents = [c for c in cands if entity_like(c, H, W)]
    t = int(max(3, round(2 * sw + 2)))
    for c in ents:
        cv2.drawContours(ring, [c.cnt], -1, 1, thickness=t)
    # leaf entities: remove interior (text)
    for c in ents:
        x0, y0, x1, y1 = c.box
        leaf = True
        for o in ents:
            if o is c: continue
            a0, b0, a1, b1 = o.box
            if a0 >= x0 and b0 >= y0 and a1 <= x1 and b1 <= y1 and (a1 - a0) * (b1 - b0) < (x1 - x0) * (y1 - y0):
                leaf = False; break
        c.leaf = leaf
        if leaf:
            cv2.drawContours(fillm, [c.cnt], -1, 1, thickness=-1)
    rm = ((ring > 0) | (fillm > 0)).astype(np.uint8)
    lm = ink.copy()
    lm[rm > 0] = 0
    return lm, rm, ents


def skel(lm):
    return skeletonize(lm > 0).astype(np.uint8)


def nearest_on(sk, p, rad=8, excl=None):
    x0, y0 = int(round(p[0])), int(round(p[1]))
    H, W = sk.shape
    best = None; bd = 1e9
    for y in range(max(0, y0 - rad), min(H, y0 + rad + 1)):
        for x in range(max(0, x0 - rad), min(W, x0 + rad + 1)):
            if sk[y, x]:
                if excl is not None and excl(x, y):
                    continue
                d = (x - p[0]) ** 2 + (y - p[1]) ** 2
                if d < bd:
                    bd = d; best = (x, y)
    return best


def walk(sk, start, d0, visited_global=None, maxlen=6000, gap=12):
    """walk along skeleton from start pixel in direction d0; returns list of (x,y)"""
    H, W = sk.shape
    path = [start]
    vis = {start}
    cur = start
    d = np.array(d0, float); d /= (np.linalg.norm(d) + 1e-9)
    for it in range(maxlen):
        x, y = cur
        nb = []
        for dy, dx in OFF8:
            xx, yy = x + dx, y + dy
            if 0 <= xx < W and 0 <= yy < H and sk[yy, xx] and (xx, yy) not in vis:
                nb.append((xx, yy))
        if len(path) >= 6:
            q = np.array(path[-6], float)
            dd = np.array(cur, float) - q
            nn = np.linalg.norm(dd)
            if nn > 0:
                d = 0.5 * d + 0.5 * dd / nn
                d /= np.linalg.norm(d) + 1e-9
        if not nb:
            # gap jump (dashed lines / small breaks)
            found = None
            for t in range(2, gap + 1):
                for s in (0, -1, 1, -2, 2):
                    px = x + d[0] * t - d[1] * s * 0.5 * t / gap * 2
                    py = y + d[1] * t + d[0] * s * 0.5 * t / gap * 2
                    xi, yi = int(round(px)), int(round(py))
                    if 0 <= xi < W and 0 <= yi < H and sk[yi, xi] and (xi, yi) not in vis:
                        found = (xi, yi); break
                if found: break
            if not found:
                break
            nb = [found]
        if len(nb) == 1:
            nxt = nb[0]
        else:
            # choose straightest continuation, looking a few steps ahead
            best = None; bs = -9
            for c in nb:
                v = np.array(c, float) - np.array(cur, float)
                s = float(v @ d) / (np.linalg.norm(v) + 1e-9)
                if s > bs:
                    bs = s; best = c
            nxt = best
        # mark the other neighbours visited to avoid zig-zag
        for c in nb:
            vis.add(c)
        path.append(nxt)
        cur = nxt
    return path


def simplify(pts, eps=1.5):
    a = np.array(pts, np.float32).reshape(-1, 1, 2)
    s = cv2.approxPolyDP(a, eps, False).reshape(-1, 2)
    return s.astype(float)


def end_dir(path, k=10):
    p = np.array(path, float)
    a = p[-1]; b = p[max(0, len(p) - 1 - k)]
    v = a - b; n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0])


def find_entity(cands, p, d, maxd=10.0):
    """entity whose outline is closest to end point p (moving in direction d)"""
    best = None; bd = 1e9; bport = None
    px, py = float(p[0]), float(p[1])
    for c in cands:
        x0, y0, x1, y1 = c.box
        if px < x0 - maxd - 2 or px > x1 + maxd + 2 or py < y0 - maxd - 2 or py > y1 + maxd + 2:
            continue
        sd = cv2.pointPolygonTest(c.cnt, (px, py), True)  # >0 inside
        ad = abs(sd)
        if ad <= maxd and ad < bd:
            bd = ad; best = c
    if best is None:
        return None, None
    # port: march along d until reaching the outline (or back if inside)
    port = np.array([px, py])
    sd = cv2.pointPolygonTest(best.cnt, (px, py), True)
    if sd < 0:
        for t in np.arange(0, 14, 0.5):
            q = (px + d[0] * t, py + d[1] * t)
            if cv2.pointPolygonTest(best.cnt, q, True) >= -1.0:
                port = np.array(q); break
    return best, port


def arrow_width(ink, path, back=(3, 16)):
    """max perpendicular ink width near the end of a path"""
    p = np.array(path, float)
    if len(p) < 4:
        return 0
    d = end_dir(path, 8); nrm = np.array([-d[1], d[0]])
    e = p[-1]
    H, W = ink.shape
    best = 0
    for t in range(back[0], back[1]):
        c = e - d * t
        cnt = 0
        run = 0; mx = 0
        for s in np.arange(-10, 10.5, 1.0):
            q = c + nrm * s
            xi, yi = int(round(q[0])), int(round(q[1]))
            if 0 <= xi < W and 0 <= yi < H and ink[yi, xi]:
                run += 1; mx = max(mx, run)
            else:
                run = 0
        best = max(best, mx)
    return best


def arrow_width_fwd(ink, e, d, rng=(-4, 18)):
    """perpendicular ink width around end point e looking forward along d (the arrowhead may lie beyond the skeleton end)"""
    nrm = np.array([-d[1], d[0]])
    H, W = ink.shape
    ws = []
    for t in range(rng[0], rng[1]):
        c = e + d * t
        run = 0; mx = 0
        for s in np.arange(-10, 10.5, 1.0):
            q = c + nrm * s
            xi, yi = int(round(q[0])), int(round(q[1]))
            if 0 <= xi < W and 0 <= yi < H and ink[yi, xi]:
                run += 1; mx = max(mx, run)
            else:
                run = 0
        ws.append(mx)
    return ws


def circle_ctrl_dir(ink, cx, cy, R):
    H, W = ink.shape
    best = None; bs = -1
    for k in range(120):
        th = 2 * math.pi * k / 120
        u = (math.cos(th), math.sin(th))
        s = 0
        for rr in range(int(R + 3), int(R + 13)):
            xi, yi = int(round(cx + u[0] * rr)), int(round(cy + u[1] * rr))
            if 0 <= xi < W and 0 <= yi < H and ink[yi, xi]:
                s += 1
        if s > bs:
            bs = s; best = th
    return np.array([math.cos(best), math.sin(best)]), bs


DBG = []
def predict_image(img):
    DBG.clear()
    H, W = img.shape[:2]
    ink, cands, circles = detect_shapes(img)
    sw = stroke_width(ink)
    lm, rm, ents = build_line_mask(ink, cands, H, W, sw)
    cands = ents
    sk = skel(lm)
    out = []
    for (cx, cy, rin) in circles:
        R = rin + 2.0  # outer radius
        u_c, sc = circle_ctrl_dir(ink, cx, cy, R)
        if sc < 6:
            DBG.append(((cx, cy), 'weak_ctrl')); continue
        u = -u_c  # from control side through centre to junction
        J0 = np.array([cx, cy]) + u * (R - 0.5)
        excl = lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 < (R - 0.5) ** 2
        # control trace
        st = nearest_on(sk, (cx + u_c[0] * (R + 4), cy + u_c[1] * (R + 4)), rad=4)
        if st is None:
            DBG.append(((cx, cy), 'no_ctrl_start')); continue
        cpath = walk(sk, st, u_c)
        cend = np.array(cpath[-1], float)
        ctrl, cport = find_entity(cands, cend, end_dir(cpath))
        if ctrl is None:
            DBG.append(((cx, cy), 'no_ctrl_entity', cend.tolist())); continue
        # junction snapped to reaction line skeleton
        js = nearest_on(sk, J0, rad=6, excl=lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 < (R - 2) ** 2)
        if js is None:
            DBG.append(((cx, cy), 'no_js')); continue
        J = np.array(js, float)
        # arms: skeleton pixels on ring of radius 12 around J, excluding the circle disc
        arms = []
        pts = []
        for k in range(180):
            th = 2 * math.pi * k / 180
            q = J + 12 * np.array([math.cos(th), math.sin(th)])
            xi, yi = int(round(q[0])), int(round(q[1]))
            if not (0 <= xi < W and 0 <= yi < H):
                continue
            if (xi - cx) ** 2 + (yi - cy) ** 2 < (R + 3) ** 2:
                continue
            win = sk[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2]
            if win.any():
                pts.append(th)
        # cluster angles
        if not pts:
            continue
        pts = sorted(pts)
        groups = [[pts[0]]]
        for t in pts[1:]:
            if t - groups[-1][-1] < 0.2:
                groups[-1].append(t)
            else:
                groups.append([t])
        if len(groups) > 1 and (pts[0] + 2 * math.pi - pts[-1]) < 0.2:
            groups[0] = groups[-1] + groups[0]; groups.pop()
        inputs = []; outputs = []
        for g in groups:
            th = math.atan2(np.mean([math.sin(t) for t in g]), np.mean([math.cos(t) for t in g]))
            d = np.array([math.cos(th), math.sin(th)])
            s0 = nearest_on(sk, J + 12 * d, rad=2)
            if s0 is None:
                continue
            p = walk(sk, s0, d)
            p = [tuple(J.astype(int))] + p
            e = np.array(p[-1], float); de = end_dir(p)
            ent, port = find_entity(cands, e, de)
            if ent is None or ent is ctrl:
                continue
            ws = arrow_width_fwd(lm, e, de, (-int(6 * sw + 6), 3))
            head = max(ws) >= max(5, 2.6 * sw)
            poly = simplify([tuple(J)] + [tuple(q) for q in p[1:]] + [tuple(port)])
            poly[0] = J; poly[-1] = port
            if head:
                outputs.append({'box': [float(v) for v in ent.box], 'path': [[float(a), float(b)] for a, b in poly]})
            else:
                poly = poly[::-1]
                inputs.append({'box': [float(v) for v in ent.box], 'path': [[float(a), float(b)] for a, b in poly]})
        if not outputs:
            DBG.append(((cx, cy), 'no_outputs', len(groups), len(inputs))); continue
        cp = [tuple(cport)] + [tuple(q) for q in cpath[::-1]] + [tuple(J)]
        cpoly = simplify(cp)
        cpoly[0] = cport; cpoly[-1] = J
        out.append({'controller': [float(v) for v in ctrl.box], 'control': 'catalysis', 'junction': [float(J[0]), float(J[1])],
                    'control_path': [[float(a), float(b)] for a, b in cpoly], 'inputs': inputs, 'outputs': outputs})
    return out


# ---------------------------------------------------------------- metric (local report only)
from scipy.optimize import linear_sum_assignment

def resample(p):
    p = np.asarray(p, float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    L = seg.sum()
    n = int(math.ceil(L / 3.0)) + 1
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, L, n)
    x = np.interp(t, cum, p[:, 0]); y = np.interp(t, cum, p[:, 1])
    return np.stack([x, y], 1)

def lcs_match(a, b, tol=6.0):
    n, m = len(a), len(b)
    M = (np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]) <= tol)
    prev = np.zeros(m + 1, np.int32)
    for i in range(n):
        cur = np.zeros(m + 1, np.int32)
        row = M[i]
        for j in range(m):
            if row[j]:
                cur[j + 1] = prev[j] + 1
            else:
                cur[j + 1] = cur[j] if cur[j] > prev[j + 1] else prev[j + 1]
        prev = cur
    return int(prev[m])

def lcs_match_fast(a, b, tol=6.0):
    # vectorised along anti-diagonals is complex; use row loop with numpy maximum.accumulate trick
    n, m = len(a), len(b)
    M = (np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1]) <= tol)
    prev = np.zeros(m + 1, np.int32)
    for i in range(n):
        diag = np.where(M[i], prev[:-1] + 1, 0)
        cand = np.maximum(prev[1:], diag)
        cur = np.concatenate([[0], np.maximum.accumulate(cand)])
        prev = cur
    return int(prev[m])

def path_quality(pp, rp):
    pp = np.asarray(pp, float); rp = np.asarray(rp, float)
    if np.hypot(*(pp[0] - rp[0])) > 6 or np.hypot(*(pp[-1] - rp[-1])) > 6:
        return 0.0
    a = resample(pp); b = resample(rp)
    h = lcs_match_fast(a, b)
    return 2.0 * h / (len(a) + len(b))

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0

def match_parts(P, R, J, side):
    if not P or not R:
        return 0.0
    C = np.zeros((len(P), len(R)))
    for i, p in enumerate(P):
        jp = p['path'][-1] if side == 'in' else p['path'][0]
        if math.hypot(jp[0] - J[0], jp[1] - J[1]) > 6:
            continue
        for k, r in enumerate(R):
            if iou(p['box'], r['box']) >= 0.5:
                C[i, k] = path_quality(p['path'], r['path'])
    ri, ci = linear_sum_assignment(-C)
    return float(C[ri, ci].sum())

def binding_credit(pb, rb):
    if pb['control'] != rb['control']:
        return 0.0
    if iou(pb['controller'], rb['controller']) < 0.5:
        return 0.0
    J = pb['junction']
    if math.hypot(J[0] - rb['junction'][0], J[1] - rb['junction'][1]) > 6:
        return 0.0
    w = 0.0
    cp = pb['control_path']
    if math.hypot(cp[-1][0] - J[0], cp[-1][1] - J[1]) <= 6:
        w += path_quality(cp, rb['control_path'])
    w += match_parts(pb['inputs'], rb['inputs'], J, 'in')
    w += match_parts(pb['outputs'], rb['outputs'], J, 'out')
    p = len(pb['inputs']) + len(pb['outputs']); g = len(rb['inputs']) + len(rb['outputs'])
    return 2.0 * w / (2 + p + g)

def diagram_score(pred, ref, detail=False):
    if len(pred) + len(ref) == 0:
        return 1.0
    if not pred or not ref:
        return (0.0, None) if detail else 0.0
    C = np.zeros((len(pred), len(ref)))
    for i, pb in enumerate(pred):
        for k, rb in enumerate(ref):
            C[i, k] = binding_credit(pb, rb)
    ri, ci = linear_sum_assignment(-C)
    s = 2.0 * C[ri, ci].sum() / (len(pred) + len(ref))
    return (s, C) if detail else s

def score(preds, refs):
    """preds, refs: dict diagram_id/id -> list of bindings"""
    ss = [diagram_score(preds.get(k, []), refs[k]) for k in refs]
    return float(np.mean(ss)), ss


# ---------------------------------------------------------------- output hygiene
def _clip_pt(p, W, H):
    return [float(min(max(p[0], 0.0), W - 1.0)), float(min(max(p[1], 0.0), H - 1.0))]


def _clean_path(path, W, H):
    pts = [_clip_pt(p, W, H) for p in path]
    out = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - out[-1][0]) > 1e-6 or abs(p[1] - out[-1][1]) > 1e-6:
            out.append(p)
    if len(out) < 2:
        return None
    if len(out) > 1024:
        idx = np.linspace(0, len(out) - 1, 1024).round().astype(int)
        out = [out[i] for i in idx]
    L = sum(math.hypot(out[i + 1][0] - out[i][0], out[i + 1][1] - out[i][1]) for i in range(len(out) - 1))
    if L <= 0 or L > 12000:
        return None
    return out


def _clean_box(b, W, H):
    x0, y0, x1, y1 = [float(v) for v in b]
    x0 = min(max(x0, 0.0), W - 2.0); y0 = min(max(y0, 0.0), H - 2.0)
    x1 = min(max(x1, x0 + 1.0), W - 1.0); y1 = min(max(y1, y0 + 1.0), H - 1.0)
    return [x0, y0, x1, y1]


def clean_bindings(B, W, H):
    out = []; seen = set()
    for b in B:
        if not all(np.isfinite(np.array(b['controller'], float))):
            continue
        cp = _clean_path(b['control_path'], W, H)
        if cp is None:
            continue
        nb = {'controller': _clean_box(b['controller'], W, H), 'control': b['control'],
              'junction': _clip_pt(b['junction'], W, H), 'control_path': cp, 'inputs': [], 'outputs': []}
        for side in ('inputs', 'outputs'):
            boxes = set()
            for q in b[side][:64]:
                bx = _clean_box(q['box'], W, H)
                key = tuple(round(v, 3) for v in bx)
                if key in boxes:
                    continue
                pp = _clean_path(q['path'], W, H)
                if pp is None:
                    continue
                boxes.add(key)
                nb[side].append({'box': bx, 'path': pp})
        key = json.dumps(nb, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(nb)
    return out[:256]


# ---------------------------------------------------------------- main
import os, glob, csv
from multiprocessing import Pool
import pandas as pd

OUT_PATH = '/working/submission.csv'


def find_data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [here, os.getcwd(), '/working', '/working/public', '/working/data', '/data', '/kaggle/input']
    for r in roots:
        if os.path.isfile(os.path.join(r, 'test.csv')) and os.path.isfile(os.path.join(r, 'diagrams.jsonl')):
            return r
    for r in ['/working', '/kaggle/input', here, os.getcwd()]:
        hits = sorted(glob.glob(os.path.join(r, '**', 'diagrams.jsonl'), recursive=True))
        for h in hits:
            dd = os.path.dirname(h)
            if os.path.isfile(os.path.join(dd, 'test.csv')):
                return dd
    raise FileNotFoundError('data directory with test.csv and diagrams.jsonl not found')


DATA = None
META = None


def _work(did):
    m = META[did]
    img = cv2.imread(os.path.join(DATA, m['image_path']), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    B = predict_image(img)
    return did, clean_bindings(B, W, H)


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'bindings'])
        for rid, B in rows:
            w.writerow([rid, json.dumps(B, separators=(',', ':'))])


def main():
    global DATA, META
    DATA = find_data_dir()
    META = {}
    with open(os.path.join(DATA, 'diagrams.jsonl')) as f:
        for line in f:
            if line.strip():
                r = json.loads(line); META[r['diagram_id']] = r
    test = pd.read_csv(os.path.join(DATA, 'test.csv'))
    # 1) fallback: valid empty submission first
    write_csv([(rid, []) for rid in test['id']], OUT_PATH)
    print('data dir', DATA, 'test rows', len(test))
    dids = list(test['diagram_id'])
    with Pool(min(8, os.cpu_count() or 1)) as pool:
        res = dict(pool.map(_work, dids))
    rows = [(rid, res[did]) for rid, did in zip(test['id'], test['diagram_id'])]
    write_csv(rows, OUT_PATH)
    nb = sum(len(b) for _, b in rows)
    print('wrote', OUT_PATH, 'rows', len(rows), 'bindings', nb)
    # held-out report on labelled diagrams (no tuning on it)
    lab = []
    for fn in ('train.csv', 'validation.csv'):
        p = os.path.join(DATA, fn)
        if os.path.isfile(p):
            lab.append(pd.read_csv(p).assign(split=fn.split('.')[0]))
    if lab:
        lab = pd.concat(lab)
        with Pool(min(8, os.cpu_count() or 1)) as pool:
            pr = dict(pool.map(_work, list(lab['diagram_id'])))
        sc = {}
        for r in lab.itertuples():
            sc.setdefault(r.split, []).append(diagram_score(pr[r.diagram_id], json.loads(r.bindings)))
        for k, v in sc.items():
            print('labelled %s diagrams %d score %.4f' % (k, len(v), float(np.mean(v))))


if __name__ == '__main__':
    main()
