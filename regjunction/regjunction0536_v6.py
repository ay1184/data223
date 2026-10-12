"""Regulatory junction and connector reconstruction in biochemical diagrams.

Method
  1. Image processing on each diagram (public images only): ink mask against the local background colour,
     entity outlines from enclosed regions, tinted fills and dashed group enclosures, stroke width estimate.
  2. Symbol candidates: enclosed round holes plus a ring filter (ring coverage, empty interior, 8 sectors).
  3. Learned symbol model, trained inside this script: a small CNN (SymNet) classifies each candidate
     crop as a regulatory catalysis terminal or not. Labels come only from the public training and
     validation annotations (junction positions); 8-way dihedral augmentation and test-time averaging.
     It runs on the GPU when one is available (torch.device line in main).
  4. Connector tracing: skeleton graph of the remaining connector ink (spur pruning, crossing merge),
     best-first search from the attachment junction to entity outlines (straightest continuation through
     crossings, dashed-gap bridging), arrowhead blobs mark outputs, unheaded terminals are inputs.
  5. Open triangles (stimulation, with a bar behind the base: necessary stimulation) and end bars
     (inhibition) are found geometrically and traced the same way.
  6. The threshold for accepting symbols is selected inside the script from {0.7, 0.85, 0.5} by 4-fold
     cross-validation over whole labelled diagrams (out-of-fold CNN probabilities, exact task metric of the
     full pipeline); the final model is refit on all labelled diagrams with that threshold.
Data and models: only the supplied public files (images, train.csv, validation.csv, test.csv,
diagrams.jsonl). No pretrained weights, no external data, no hosted APIs, no internet access.
Test images are only read to predict their own row.

Runtime (measured, this exact file, end to end incl. both CNN fits and the validation pass):
  1210 s (20.2 min) wall clock for this exact file, measured on a 4-core CPU without any GPU (5 CNN fits
  on the CPU, nothing else running); that is 22% of the 90-minute cap. The A10G run itself was not measured.
Local out-of-fold score printed by this run (52 labelled diagrams, 4 folds, threshold 0.7): 0.6671.
"""
import os, glob, csv, json, math, random
from multiprocessing import Pool
import numpy as np, pandas as pd, cv2
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize
import torch, torch.nn as nn, torch.nn.functional as F

SEED = 0
EPOCHS = 15
THRS = [0.7, 0.85, 0.5]
N_FOLDS = 4
N_PROC = max(1, min(8, os.cpu_count() or 1))


# ===================================================================================== pipe

INK_T = 215
MIN_INNER_COMP = 1
MIN_DASH_GAPS = 6
DROP_SLIVERS = 1
MEH = [1]
SMALL_HOLE = 8
MERGED_NO_RING = 0
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
    bgc = np.stack([cv2.medianBlur(np.ascontiguousarray(img[:, :, k]), 41) for k in range(3)], 2)
    diff = np.abs(img.astype(np.int16) - bgc.astype(np.int16)).max(axis=2)
    ink = (((diff > 45) & (mn < 235)) | (mn < 110)).astype(np.uint8)
    # holes: enclosed background regions
    bg = 1 - ink
    n, lab, st, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
    holes = []
    circles = []
    region = np.zeros((H, W), np.int32)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < SMALL_HOLE or h < SMALL_HOLE:
            continue
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if w * h > 0.4 * H * W:
            continue
        if w < 8 or h < 8:
            # tiny enclosed round holes: small catalysis circles
            if abs(w - h) <= 2 and a >= 0.55 * w * h:
                circles.append((x + (w - 1) / 2.0, y + (h - 1) / 2.0, (w + h) / 4.0))
            continue
        m = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c0 = max(cs, key=cv2.contourArea)
        mf = np.zeros_like(m); cv2.drawContours(mf, [c0], -1, 1, thickness=-1)
        c = c0 + np.array([x, y])
        fa = float(mf.sum())
        fill = fa / float(w * h)
        if 6 <= w <= 72 and 6 <= h <= 72 and abs(w - h) <= max(2, 0.15 * max(w, h)) and 0.6 < fill < 0.86 and a > 0.85 * fa:
            circles.append((x + (w - 1) / 2.0, y + (h - 1) / 2.0, (w + h) / 4.0))
            continue
        if w < 16 or h < 12:
            continue
        innerm = ((mf > 0) & (m == 0)).astype(np.uint8)
        ncomp = 0
        if innerm.any():
            nci, _, sti, _ = cv2.connectedComponentsWithStats(innerm, connectivity=8)
            ncomp = int(((sti[1:, 4] >= 3) & (sti[1:, 3] < 0.9 * h)).sum())
        holes.append((c, fill, (x, y, w, h), fa - a if ncomp >= MIN_INNER_COMP else 0))
    cands = []
    for c, fill, (x, y, w, h), inner in holes:
        # grow contour by ~1.5px to the border centreline
        cc = Cand(c, 'hole')
        cc.inner = inner
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
        c0 = max(cs, key=cv2.contourArea)
        mf = np.zeros_like(m); cv2.drawContours(mf, [c0], -1, 1, thickness=-1)
        c = c0 + np.array([x, y])
        cc = Cand(c, 'tint')
        cc.inner = float((mf > 0)[ink[y:y + h, x:x + w] > 0].sum())
        cc.box = [x - 1.0, y - 1.0, x + w, y + h]
        cc.fill = float(mf.sum()) / float(w * h)
        cands.append(cc)
    # complexes: tinted fill joined with the member boxes it surrounds
    if True:
        MERGE_EMPTY_HOLES = MEH[0]
        M = (tint > 0).astype(np.uint8)
        filler = M.copy()
        if MERGE_EMPTY_HOLES:
            for c in cands:
                if c.kind == 'hole' and c.inner < max(12, 0.01 * c.area) and c.rect[2] * c.rect[3] < 0.05 * H * W:
                    cv2.drawContours(M, [c.cnt], -1, 1, thickness=-1)
                    cv2.drawContours(filler, [c.cnt], -1, 1, thickness=-1)
        for c in cands:
            if c.kind == 'hole' and c.inner >= max(12, 0.01 * c.area) and c.fill > 0.7:
                cv2.drawContours(M, [c.cnt], -1, 1, thickness=-1)
                cv2.drawContours(M, [c.cnt], -1, 1, thickness=5)
        M = cv2.morphologyEx(M, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n4, lab4, st4, _ = cv2.connectedComponentsWithStats(M, connectivity=8)
        for i in range(1, n4):
            x, y, w, h, a = st4[i]
            if w < 30 or h < 24 or w * h > 0.4 * H * W:
                continue
            if (filler[y:y + h, x:x + w] > 0).sum() < 0.08 * w * h:
                continue
            m = (lab4[y:y + h, x:x + w] == i).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c0 = max(cs, key=cv2.contourArea)
            mf = np.zeros_like(m); cv2.drawContours(mf, [c0], -1, 1, thickness=-1)
            bx = [x - 1.0, y - 1.0, x + w, y + h]
            dup = False
            for o in cands:
                ob = o.box
                ix = max(0, min(ob[2], bx[2]) - max(ob[0], bx[0])); iy = max(0, min(ob[3], bx[3]) - max(ob[1], bx[1]))
                u = (ob[2] - ob[0]) * (ob[3] - ob[1]) + (bx[2] - bx[0]) * (bx[3] - bx[1]) - ix * iy
                if u > 0 and ix * iy / u > 0.85:
                    dup = True; break
            if dup:
                continue
            nmem = 0
            for o in cands:
                if o.kind == 'hole' and o.inner >= max(12, 0.01 * o.area) and o.fill > 0.7:
                    ob = o.box
                    if ob[0] >= bx[0] - 2 and ob[1] >= bx[1] - 2 and ob[2] <= bx[2] + 2 and ob[3] <= bx[3] + 2:
                        nmem += 1
            if nmem < 1:
                continue
            cc = Cand(c0 + np.array([x, y]), 'tint')
            cc.box = bx
            cc.fill = float(mf.sum()) / float(w * h)
            cc.inner = float((mf > 0)[ink[y:y + h, x:x + w] > 0].sum())
            cc.merged = True
            cands.append(cc)
        if DROP_SLIVERS:
            merged = [c for c in cands if getattr(c, 'merged', False)]
            keep = []
            for c in cands:
                if c.kind in ('tint', 'hole') and not getattr(c, 'merged', False) and c.inner < max(12, 0.01 * c.area):
                    inside = False
                    for m_ in merged:
                        b1, b2 = c.box, m_.box
                        if b1[0] >= b2[0] - 3 and b1[1] >= b2[1] - 3 and b1[2] <= b2[2] + 3 and b1[3] <= b2[3] + 3 and c.area < 0.5 * m_.area:
                            inside = True; break
                    if inside:
                        continue
                keep.append(c)
            cands[:] = keep
    # dashed / dotted enclosures: close short gaps along rows and columns, then look for new rectangular holes
    L = 11
    ch = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, L), np.uint8))
    cvv = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((L, 1), np.uint8))
    closed = (ink | ch | cvv).astype(np.uint8)
    n3, lab3, st3, _ = cv2.connectedComponentsWithStats(1 - closed, connectivity=4)
    for i in range(1, n3):
        x, y, w, h, a = st3[i]
        if w < 30 or h < 24 or x == 0 or y == 0 or x + w >= W or y + h >= H or w * h > 0.4 * H * W:
            continue
        m = (lab3[y:y + h, x:x + w] == i).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c0 = max(cs, key=cv2.contourArea)
        mf = np.zeros_like(m); cv2.drawContours(mf, [c0], -1, 1, thickness=-1)
        c = c0 + np.array([x, y])
        fa = float(mf.sum())
        if fa / float(w * h) < 0.9:
            continue
        # a dashed border has many regular gaps; a region closed by bridging one or two line-end gaps is not an enclosure
        cpts = c[:, 0, :]
        dense = []
        for k in range(len(cpts)):
            p0 = cpts[k]; p1 = cpts[(k + 1) % len(cpts)]
            nn = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
            for t in range(nn):
                dense.append((p0 + (p1 - p0) * t / max(1, nn)).round().astype(int))
        dense = np.array(dense)
        # look one pixel outside the hole boundary (on the border stroke)
        cx_, cy_ = x + w / 2.0, y + h / 2.0
        vx = dense[:, 0] - cx_; vy = dense[:, 1] - cy_
        nv = np.hypot(vx, vy) + 1e-6
        bxp = np.clip((dense[:, 0] + np.round(vx / nv * 2)).astype(int), 0, W - 1)
        byp = np.clip((dense[:, 1] + np.round(vy / nv * 2)).astype(int), 0, H - 1)
        on = ink[byp, bxp] > 0
        gaps = int(((~on[1:]) & on[:-1]).sum())
        if gaps < MIN_DASH_GAPS:
            continue
        bx = [x - 1.5, y - 1.5, x + w + 0.5, y + h + 0.5]
        dup = False
        for o in cands:
            ob = o.box
            ix = max(0, min(ob[2], bx[2]) - max(ob[0], bx[0])); iy = max(0, min(ob[3], bx[3]) - max(ob[1], bx[1]))
            u = (ob[2] - ob[0]) * (ob[3] - ob[1]) + (bx[2] - bx[0]) * (bx[3] - bx[1]) - ix * iy
            if u > 0 and ix * iy / u > 0.85:
                dup = True; break
        if dup:
            continue
        cc = Cand(c, 'dash')
        cc.box = bx
        cc.fill = fa / float(w * h)
        cc.inner = fa - a
        cands.append(cc)
    return ink, cands, circles


def stroke_width(ink):
    """most common perpendicular run length of ink (rows and columns)"""
    hist = np.zeros(12)
    for a in (ink, ink.T):
        a = a.astype(np.int8)
        d = np.diff(np.pad(a, ((0, 0), (1, 1))), axis=1)
        st = np.nonzero(d == 1); en = np.nonzero(d == -1)
        L = en[1] - st[1]
        L = L[(L >= 1) & (L <= 11)]
        hist += np.bincount(L, minlength=12)[:12]
    hist[0] = 0
    # weight runs by length so 1px antialias fringes do not dominate
    m = int(np.argmax(hist * np.arange(12) ** 0.5))
    # bimodal drawings (thin connectors plus thick decorations): use the dominant adjacent pair
    pair = hist[1:11] + hist[2:12]
    k = int(np.argmax(pair)) + 1
    alt = (k * hist[k] + (k + 1) * hist[k + 1]) / max(1.0, hist[k] + hist[k + 1])
    if abs(alt - m) > 1.5:
        return float(max(1.0, alt))
    return float(max(1.0, m))


def entity_like(c, H, W):
    x, y, w, h = c.rect
    hull = cv2.convexHull(c.cnt)
    ha = cv2.contourArea(hull)
    sol = c.area / ha if ha > 0 else 0
    c.solidity = sol
    if c.kind == 'hole' and c.inner < max(12, 0.01 * c.area):
        return False
    return sol > 0.93 and c.fill > 0.7 and w < 0.6 * W and h < 0.6 * H


def build_line_mask(ink, cands, H, W, sw):
    ring = np.zeros((H, W), np.uint8)
    fillm = np.zeros((H, W), np.uint8)
    ents = [c for c in cands if entity_like(c, H, W)]
    t = int(max(3, round(2 * sw + 2)))
    for c in ents:
        if getattr(c, 'merged', False) and MERGED_NO_RING:
            continue
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
        if leaf and (c.kind == 'hole' or c.rect[3] < 0.15 * H):
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
ARMDBG = []
def predict_image(img):
    DBG.clear(); ARMDBG.clear()
    H, W = img.shape[:2]
    ink, cands, circles = detect_shapes(img)
    sw = stroke_width(ink)
    lm, rm, ents = build_line_mask(ink, cands, H, W, sw)
    cands = ents
    sk = skel(lm)
    out = []
    keep = []
    for (cx, cy, rin) in circles:
        inside = False
        for c in cands:
            if getattr(c, 'leaf', False) and c.box[0] < cx < c.box[2] and c.box[1] < cy < c.box[3]:
                if cv2.pointPolygonTest(c.cnt, (float(cx), float(cy)), True) > 0:
                    inside = True; break
        if not inside:
            keep.append((cx, cy, rin))
    circles = keep
    for (cx, cy, rin) in circles:
        R = rin + sw  # outer radius
        u_c, sc = circle_ctrl_dir(ink, cx, cy, R)
        if sc < 6:
            DBG.append(((cx, cy), 'weak_ctrl')); continue
        u = -u_c  # from control side through centre to junction
        J0 = np.array([cx, cy]) + u * (R + 0.5 * sw)
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
        Rex = rin + 0.6 * sw
        js = nearest_on(sk, J0, rad=6, excl=lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 < Rex ** 2)
        if js is None:
            DBG.append(((cx, cy), 'no_js')); continue
        J = np.array(js, float)
        # arms: skeleton pixels on rings around J, outside the symbol disc
        pts = []
        for rad in (4, 7, 10, 14):
            for k in range(144):
                th = 2 * math.pi * k / 144
                q = J + rad * np.array([math.cos(th), math.sin(th)])
                xi, yi = int(round(q[0])), int(round(q[1]))
                if not (0 <= xi < W and 0 <= yi < H):
                    continue
                if (xi - cx) ** 2 + (yi - cy) ** 2 < Rex ** 2:
                    continue
                if sk[yi, xi] or (rad > 4 and sk[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].any()):
                    pts.append((th, rad, xi, yi))
        groups = []
        for th, rad, xi, yi in sorted(pts):
            for g in groups:
                dth = abs((th - g['th'] + math.pi) % (2 * math.pi) - math.pi)
                if dth < 0.45:
                    g['m'].append((th, rad, xi, yi)); break
            else:
                groups.append({'th': th, 'm': [(th, rad, xi, yi)]})
        inputs = []; outputs = []
        used = set()
        for g in groups:
            m = sorted(g['m'], key=lambda t: t[1])
            far = m[-1]
            s0 = (far[2], far[3])
            if not sk[s0[1], s0[0]]:
                s0 = nearest_on(sk, (far[2], far[3]), rad=1)
                if s0 is None:
                    continue
            d = np.array(s0, float) - J; d /= np.linalg.norm(d) + 1e-9
            p = walk(sk, s0, d)
            p = [tuple(J.astype(int))] + p
            e = np.array(p[-1], float); de = end_dir(p)
            ent, port = find_entity(cands, e, de)
            ARMDBG.append(((cx, cy), J.tolist(), round(math.degrees(math.atan2(d[1], d[0]))), len(p), e.tolist(), None if ent is None else [round(v) for v in ent.box], ent is ctrl if ent is not None else None))
            if ent is None or ent is ctrl or id(ent) in used:
                continue
            used.add(id(ent))
            ws = arrow_width_fwd(lm, e, de, (-int(6 * sw + 6), 3))
            head = max(ws) >= max(5, 2.6 * sw)
            ARMDBG.append(('ws', max(ws), head))
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


# ===================================================================================== graph

OFF8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class SkelGraph:
    """skeleton -> nodes (clusters of non-degree-2 pixels) and edges (pixel chains)"""

    def __init__(self, sk, spur=6, merge=5):
        self.H, self.W = sk.shape
        self.sk = sk.astype(np.uint8).copy()
        self._build()
        for _ in range(3):
            if not self._prune(spur):
                break
            self._build()
        self._merge(merge)

    def _build(self):
        sk = self.sk
        k = np.ones((3, 3), np.uint8); k[1, 1] = 0
        deg = cv2.filter2D(sk, cv2.CV_16S, k, borderType=cv2.BORDER_CONSTANT) * sk
        nodepx = (sk > 0) & (deg != 2)
        lab, n = ndi.label(nodepx, structure=np.ones((3, 3)))
        self.nodelab = lab
        self.nn = n
        ys, xs = np.nonzero(nodepx)
        self.node_pix = [[] for _ in range(n + 1)]
        for y, x in zip(ys, xs):
            self.node_pix[lab[y, x]].append((x, y))
        self.node_pos = [None] + [np.mean(np.array(p, float), 0) for p in self.node_pix[1:]]
        # edges
        H, W = self.H, self.W
        visited = np.zeros_like(sk, bool)
        edges = []
        adj = [[] for _ in range(n + 1)]
        for nid in range(1, n + 1):
            for (x, y) in self.node_pix[nid]:
                for dy, dx in OFF8:
                    xx, yy = x + dx, y + dy
                    if not (0 <= xx < W and 0 <= yy < H) or not sk[yy, xx]:
                        continue
                    if lab[yy, xx]:
                        continue  # node-node adjacency handled below
                    if visited[yy, xx]:
                        continue
                    chain = [(x, y), (xx, yy)]
                    visited[yy, xx] = True
                    px, py = xx, yy
                    prev = (x, y)
                    end = None
                    while True:
                        nxt = None
                        for ddy, ddx in OFF8:
                            ax, ay = px + ddx, py + ddy
                            if not (0 <= ax < W and 0 <= ay < H) or not sk[ay, ax]:
                                continue
                            if (ax, ay) == prev or (ax, ay) in chain[-3:]:
                                continue
                            if lab[ay, ax]:
                                if lab[ay, ax] != nid or len(chain) > 3:
                                    end = lab[ay, ax]; nxt = (ax, ay)
                                    break
                                continue
                            if not visited[ay, ax]:
                                nxt = (ax, ay)
                                break
                        if nxt is None:
                            break
                        chain.append(nxt)
                        if end is not None:
                            break
                        visited[nxt[1], nxt[0]] = True
                        prev = (px, py)
                        px, py = nxt
                    if end is None:
                        continue
                    eid = len(edges)
                    edges.append({'a': nid, 'b': end, 'pix': chain})
                    adj[nid].append((eid, 0))
                    if end != nid:
                        adj[end].append((eid, 1))
                    else:
                        adj[end].append((eid, 1))
        self.edges = edges
        self.adj = adj
        # free loops (no node) are ignored

    def _prune(self, spur):
        changed = False
        for e in self.edges:
            a, b = e['a'], e['b']
            da, db = len(self.adj[a]), len(self.adj[b])
            L = len(e['pix'])
            if L < spur and ((da == 1 and db >= 3) or (db == 1 and da >= 3)):
                tip = a if da == 1 else b
                for (x, y) in e['pix']:
                    if self.nodelab[y, x] in (0, tip):
                        self.sk[y, x] = 0
                changed = True
        return changed

    def _merge(self, merge):
        # union nodes connected by very short edges
        par = list(range(self.nn + 1))

        def f(i):
            while par[i] != i:
                par[i] = par[par[i]]; i = par[i]
            return i
        for e in self.edges:
            if len(e['pix']) <= merge and e['a'] != e['b']:
                par[f(e['a'])] = f(e['b'])
        self.root = [f(i) for i in range(self.nn + 1)]
        groups = {}
        for i in range(1, self.nn + 1):
            groups.setdefault(self.root[i], []).append(i)
        self.mnode_pos = {}
        for r, mem in groups.items():
            pts = np.concatenate([np.array(self.node_pix[m], float) for m in mem], 0)
            self.mnode_pos[r] = pts.mean(0)
        self.madj = {r: [] for r in groups}
        for eid, e in enumerate(self.edges):
            ra, rb = self.root[e['a']], self.root[e['b']]
            if ra == rb and len(e['pix']) <= merge:
                continue
            self.madj[ra].append((eid, 0))
            self.madj[rb].append((eid, 1))
        # pixel -> edge index map
        self.pix_edge = -np.ones((self.H, self.W), np.int32)
        self.pix_idx = np.zeros((self.H, self.W), np.int32)
        for eid, e in enumerate(self.edges):
            for k, (x, y) in enumerate(e['pix']):
                if self.nodelab[y, x] == 0:
                    self.pix_edge[y, x] = eid; self.pix_idx[y, x] = k

    def edge_pts(self, eid, from_side):
        p = self.edges[eid]['pix']
        return p if from_side == 0 else p[::-1]

    def tangent(self, pts, k=10):
        """direction leaving pts[0] along the chain"""
        p = np.array(pts, float)
        j = min(len(p) - 1, k)
        v = p[j] - p[0]
        n = np.linalg.norm(v)
        return v / n if n > 0 else np.array([1.0, 0.0])

    def node_of(self, eid, side):
        e = self.edges[eid]
        return self.root[e['a'] if side == 0 else e['b']]


def turn(d_in, d_out):
    c = float(np.clip(d_in @ d_out, -1, 1))
    return math.degrees(math.acos(c))


# ===================================================================================== rings


def ring_kernel(r, w):
    s = int(math.ceil(r + w + 1))
    yy, xx = np.mgrid[-s:s + 1, -s:s + 1]
    d = np.hypot(xx, yy)
    k = (np.abs(d - r) <= w / 2.0).astype(np.float32)
    return k / k.sum()


def disk_kernel(r):
    s = int(math.ceil(r + 1))
    yy, xx = np.mgrid[-s:s + 1, -s:s + 1]
    k = (np.hypot(xx, yy) <= r).astype(np.float32)
    return k / max(1.0, k.sum())


def detect_rings(ink, sw, rmin=4.0, rmax=30.0, cov_t=0.72, int_t=0.33, sec_t=0.45):
    """ring-shaped symbols: centre line radius r; returns list of (cx, cy, rin, score)"""
    H, W = ink.shape
    f = ink.astype(np.float32)
    found = []
    rr = rmin
    w = max(1.5, 0.8 * sw)
    while rr <= rmax:
        cov = cv2.filter2D(f, -1, ring_kernel(rr, w), borderType=cv2.BORDER_CONSTANT)
        ri = rr - 0.5 * sw - 1.5
        if ri < 1.5:
            rr += 1.0; continue
        inn = cv2.filter2D(f, -1, disk_kernel(ri), borderType=cv2.BORDER_CONSTANT)
        sc = cov - 0.8 * inn
        mx = cv2.dilate(sc, np.ones((5, 5), np.uint8))
        cand = (sc >= mx - 1e-6) & (cov >= cov_t) & (inn <= int_t)
        ys, xs = np.nonzero(cand)
        for x, y in zip(xs, ys):
            # sector check
            ok = True; secs = []
            for s in range(8):
                hit = 0; tot = 0
                for a in np.linspace(2 * math.pi * s / 8, 2 * math.pi * (s + 1) / 8, 7, endpoint=False):
                    px = int(round(x + rr * math.cos(a))); py = int(round(y + rr * math.sin(a)))
                    tot += 1
                    if 0 <= px < W and 0 <= py < H and ink[max(0, py - 1):py + 2, max(0, px - 1):px + 2].any():
                        hit += 1
                secs.append(hit / tot)
            if min(secs) < sec_t:
                continue
            found.append((float(x), float(y), rr - 0.5 * sw, float(sc[y, x]), rr))
        rr += 1.0
    # nms
    found.sort(key=lambda t: -t[3])
    keep = []
    for f_ in found:
        if all(math.hypot(f_[0] - k[0], f_[1] - k[1]) > 0.8 * max(f_[4], k[4]) for k in keep):
            keep.append(f_)
    return keep


# ===================================================================================== symbols2


def detect_triangles(ink, sw):
    """open triangular terminals (stimulation): enclosed small triangular holes; returns apex, base mid"""
    H, W = ink.shape
    bg = (1 - ink).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 5 or h < 5 or w > 70 or h > 70 or a < 12:
            continue
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        m = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        ca = cv2.contourArea(c)
        if ca < 10:
            continue
        hull = cv2.convexHull(c)
        if ca / max(1.0, cv2.contourArea(hull)) < 0.85:
            continue
        tri = cv2.minEnclosingTriangle(c.astype(np.float32))[1]
        if tri is None:
            continue
        tri = tri.reshape(3, 2)
        ta = abs(cv2.contourArea(tri.astype(np.float32)))
        if ta <= 0 or ca / ta < 0.75:
            continue
        # fill against bbox: triangles ~0.5
        if not (0.3 < ca / float(w * h) < 0.7):
            continue
        tri = tri + np.array([x, y], np.float32)
        out.append(tri)
    res = []
    for tri in out:
        best = None
        for k in range(3):
            v = tri[k]; o = [tri[j] for j in range(3) if j != k]
            mid = (o[0] + o[1]) / 2.0
            u = mid - v; L = np.linalg.norm(u)
            if L < 3:
                continue
            u = u / L
            # control line beyond the base, outward
            hit = 0
            for t in range(int(sw) + 2, int(sw) + 12):
                q = mid + u * t
                xi, yi = int(round(q[0])), int(round(q[1]))
                if 0 <= xi < W and 0 <= yi < H and ink[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].any():
                    hit += 1
            if best is None or hit > best[0]:
                best = (hit, v.copy(), mid.copy(), L)
        if best is None or best[0] < 6:
            continue
        hit, apex, mid, L = best
        res.append((apex, mid, L))
    return res


def has_bar(ink, p, d, sw, minlen):
    """ink run perpendicular to d through p, length >= minlen"""
    n = np.array([-d[1], d[0]])
    H, W = ink.shape
    best = 0
    for along in (-1.0, 0.0, 1.0, 2.0):
        c = p + d * along
        run = 0; mx = 0
        for s in np.arange(-3 * minlen, 3 * minlen + 0.5, 1.0):
            q = c + n * s
            xi, yi = int(round(q[0])), int(round(q[1]))
            if 0 <= xi < W and 0 <= yi < H and ink[yi, xi]:
                run += 1; mx = max(mx, run)
            else:
                run = 0
        best = max(best, mx)
    return best >= minlen, best


# ===================================================================================== pipe3
USE_EXTRA = True
CROSS_PEN = 3.0
BRANCH_DC = 1.3
MAX_PER_ARM = 3
DROP_REVERSIBLE = 1.0
NEAR_NODE_ARMS = 1.0
DEDUPE = 1.0
KEEP_NECESSARY = 0.0
PREFER_LARGE = 1.0
ALT_DIRS = 0.0
PLATEAU = 0.0
SPUR_A = 2.0
MERGE_B = 3.0
ENT_A = 6.0
RECT_SNAP = 0.0
RAY_INPUT = 1.0
RAY_OUTPUT = 1.0
RAY_OUT_HEAD = 1.0
RAY_BOTH = 0.0
RAY_MULTI_IN = 0.0
RAY_CTRL = 0.0
RECLASS = 0.0
CTRL_CROSSCHECK = 1.0
CC_COVER = 0.9
CC_RATIO = 1.5
USE_RINGS = False

DBG = []


def find_entity2(cands, p, d, maxd, exclude=()):
    best = None; bd = 1e9
    px, py = float(p[0]), float(p[1])
    for c in cands:
        if id(c) in exclude:
            continue
        x0, y0, x1, y1 = c.box
        if px < x0 - maxd - 2 or px > x1 + maxd + 2 or py < y0 - maxd - 2 or py > y1 + maxd + 2:
            continue
        sd = cv2.pointPolygonTest(c.cnt, (px, py), True)
        ad = abs(sd) if sd < 0 else (0.0 if sd < 4 else sd)
        # prefer the innermost outline when inside (members of complexes) only if close to its border
        if ad <= maxd and (ad < bd - 1.5 or (ad <= bd + 1.5 and best is not None and c.area > best.area and ad < bd + 1.5 and abs(ad - bd) <= 1.5 and PREFER_LARGE)):
            bd = min(ad, bd); best = c
    if best is None:
        return None, None
    port = np.array([px, py])
    sd = cv2.pointPolygonTest(best.cnt, (px, py), True)
    if sd < -1.0:
        for t in np.arange(0, maxd + 2, 0.5):
            q = (px + d[0] * t, py + d[1] * t)
            if cv2.pointPolygonTest(best.cnt, q, True) >= -1.5:
                port = np.array(q); break
    return best, port


class Ctx:
    pass


def prep_base(img):
    H, W = img.shape[:2]
    ink, cands, hole_circles = detect_shapes(img)
    sw = stroke_width(ink)
    lm, rm, ents = build_line_mask(ink, cands, H, W, sw)
    sw = stroke_width(lm)
    B = Ctx()
    B.H, B.W, B.img, B.ink, B.ents, B.sw, B.lm, B.hole_circles = H, W, img, ink, ents, sw, lm, hole_circles
    return B


def _inside_leaf(B, cx, cy):
    for c in B.ents:
        if getattr(c, 'leaf', False) and c.box[0] < cx < c.box[2] and c.box[1] < cy < c.box[3]:
            if cv2.pointPolygonTest(c.cnt, (float(cx), float(cy)), True) > 0:
                return True
    return False


CROP = 40


def circle_candidates(B, use_rings=True):
    """ring-shaped symbol candidates (centre x, y, inner radius) and normalised crops"""
    sw = B.sw
    cl = [(x, y, rin) for (x, y, rin) in B.hole_circles]
    if use_rings:
        for (x, y, rin, sc_, rr) in detect_rings(B.ink, sw):
            if all(math.hypot(x - a, y - b_) > 0.6 * (rin + r2) for (a, b_, r2) in cl):
                cl.append((x, y, rin))
    cl = [c for c in cl if not _inside_leaf(B, c[0], c[1])]
    gray = cv2.cvtColor(B.img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    inkf = B.ink.astype(np.float32)
    crops = np.zeros((len(cl), 2, CROP, CROP), np.float32)
    for i, (x, y, rin) in enumerate(cl):
        half = 2.0 * (rin + sw) + 2
        sc = CROP / (2 * half)
        M = np.array([[sc, 0, CROP / 2.0 - sc * x], [0, sc, CROP / 2.0 - sc * y]], np.float32)
        crops[i, 0] = cv2.warpAffine(inkf, M, (CROP, CROP), flags=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR, borderValue=0)
        crops[i, 1] = 1.0 - cv2.warpAffine(gray, M, (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=1.0)
    return cl, crops


def circle_ctrl_dirs(ink, cx, cy, R, k=2):
    H, W = ink.shape
    prof = np.zeros(120)
    for j in range(120):
        th = 2 * math.pi * j / 120
        u = (math.cos(th), math.sin(th))
        s = 0
        for rr in range(int(R + 3), int(R + 13)):
            xi, yi = int(round(cx + u[0] * rr)), int(round(cy + u[1] * rr))
            if 0 <= xi < W and 0 <= yi < H and ink[yi, xi]:
                s += 1
        prof[j] = s
    order = np.argsort(-prof, kind='stable')
    out = []
    for j in order:
        if prof[j] < 5:
            break
        if all(min(abs(j - o), 120 - abs(j - o)) >= 20 for o, _ in out):
            jj = float(j)
            if PLATEAU:
                # centre of the plateau of equal maximal ink around j
                lo = 0
                while lo < 10 and prof[(j - lo - 1) % 120] == prof[j]:
                    lo += 1
                hi = 0
                while hi < 10 and prof[(j + hi + 1) % 120] == prof[j]:
                    hi += 1
                jj = j + (hi - lo) / 2.0
            out.append((jj, prof[j]))
        if len(out) >= k:
            break
    return [(np.array([math.cos(2 * math.pi * j / 120), math.sin(2 * math.pi * j / 120)]), sc) for j, sc in out]


def prep_finish(B, circles):
    H, W, ink, sw, lm = B.H, B.W, B.ink, B.sw, B.lm
    syms = []
    yy, xx = np.mgrid[0:H, 0:W]
    lm2 = lm.copy()
    for gi, (cx, cy, rin) in enumerate(circles):
        R = rin + sw
        dirs = circle_ctrl_dirs(ink, cx, cy, R, k=2 if ALT_DIRS else 1)
        if not dirs:
            continue
        r0 = int(R + 3)
        x0, x1 = max(0, int(cx) - r0), min(W, int(cx) + r0 + 1)
        y0, y1 = max(0, int(cy) - r0), min(H, int(cy) + r0 + 1)
        dd = np.hypot(xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy)
        keep = np.zeros_like(dd, bool)
        for u_c, sc in dirs:
            u = -u_c
            J0 = np.array([cx, cy]) + u * (rin + 1.5 * sw)
            keep |= np.hypot(xx[y0:y1, x0:x1] - J0[0], yy[y0:y1, x0:x1] - J0[1]) <= sw + 2
            syms.append({'type': 'catalysis', 'c': np.array([cx, cy]), 'rin': rin, 'R': R, 'u': u, 'J0': J0, 'grp': ('c', gi)})
        band = (dd <= rin + sw + 0.5) & ~keep
        lm2[y0:y1, x0:x1][band] = 0
    if USE_EXTRA:
        for apex, mid, L in detect_triangles(ink, sw):
            if _inside_leaf(B, apex[0], apex[1]):
                continue
            u = (apex - mid) / max(1e-6, np.linalg.norm(apex - mid))
            # bar right behind the base -> necessary stimulation
            nb_, _ = has_bar(ink, mid - u * (sw + 3), u, sw, max(4, 0.6 * L))
            typ = 'necessary_stimulation' if nb_ else 'stimulation'
            if typ == 'necessary_stimulation' and not KEEP_NECESSARY:
                continue
            cen = (apex + 2 * mid) / 3.0
            r = int(L + sw + 3)
            x0, x1 = max(0, int(cen[0]) - r), min(W, int(cen[0]) + r + 1)
            y0, y1 = max(0, int(cen[1]) - r), min(H, int(cen[1]) + r + 1)
            dc = np.hypot(xx[y0:y1, x0:x1] - cen[0], yy[y0:y1, x0:x1] - cen[1])
            dj = np.hypot(xx[y0:y1, x0:x1] - apex[0], yy[y0:y1, x0:x1] - apex[1])
            band = (dc <= 0.75 * L + sw) & (dj > sw + 2)
            lm2[y0:y1, x0:x1][band] = 0
            syms.append({'type': typ, 'c': cen, 'rin': 0.5 * L, 'R': 0.5 * L + sw, 'u': u, 'J0': apex + u * 0.5 * sw,
                         'a0': mid - u * (sw + 2), 'excl': 0.4 * L})
    ks = 2 * int(round(sw)) + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    core = cv2.morphologyEx(lm2, cv2.MORPH_OPEN, ker)
    blob = cv2.dilate(core, ker) & lm2
    nb, blab, bst, bcen = cv2.connectedComponentsWithStats(blob, connectivity=8)
    heads = np.zeros_like(lm2)
    good = [i for i in range(1, nb) if bst[i, 4] >= max(12, 2.0 * sw * sw) and max(bst[i, 2], bst[i, 3]) <= 12 * sw + 20]
    for i in good:
        heads[blab == i] = 1
    lm3 = lm2.copy()
    lm3[cv2.dilate(heads, np.ones((3, 3), np.uint8)) > 0] = 0
    sk = skeletonize(lm3 > 0).astype(np.uint8)
    G = SkelGraph(sk, spur=int(SPUR_A * sw + 5), merge=int(sw + MERGE_B))
    if USE_EXTRA:
        for r_, mem in [(r_, None) for r_ in G.madj]:
            if len(G.madj[r_]) != 1:
                continue
            e2, s2 = G.madj[r_][0]
            pts = G.edge_pts(e2, s2)
            if len(pts) < 3 * sw + 6:
                continue
            e = np.array(pts[0], float)
            q = np.array(pts[min(len(pts) - 1, int(3 * sw + 6))], float)
            d = e - q; d /= np.linalg.norm(d) + 1e-9
            # skip arrowhead ends
            rr = int(2 * sw + 4)
            if heads[max(0, int(e[1]) - rr):int(e[1]) + rr + 1, max(0, int(e[0]) - rr):int(e[0]) + rr + 1].any():
                continue
            ok, blen = has_bar(lm2, e + d * 1.0, d, sw, 3 * sw + 4)
            if not ok:
                continue
            # reaction line ahead across a small gap
            hitp = None
            for t in range(2, int(4 * sw + 10)):
                p2 = e + d * t
                xi, yi = int(round(p2[0])), int(round(p2[1]))
                if not (0 <= xi < W and 0 <= yi < H):
                    break
                if t > sw + 2 and G.pix_edge[yi, xi] >= 0 and G.pix_edge[yi, xi] != e2:
                    hitp = np.array([xi, yi], float); break
            if hitp is None:
                continue
            syms.append({'type': 'inhibition', 'c': e, 'rin': 0.0, 'R': sw, 'u': d, 'J0': hitp, 'a0': e - d * (sw + 2),
                         'excl': 0.0, 'bar_end': (e2, s2)})
    C = Ctx()
    C.H, C.W, C.ink, C.ents, C.sw, C.lm, C.lm3, C.heads, C.blab, C.sk, C.G, C.syms = H, W, ink, B.ents, sw, lm, lm3, heads, blab, G.sk, G, syms
    return C


def prep(img):
    B = prep_base(img)
    cl = [c for c in B.hole_circles if not _inside_leaf(B, c[0], c[1])]
    return prep_finish(B, cl)


def walk_graph(C, eid, k, direction, maxsteps=40):
    """walk from pixel index k of edge eid; direction +1 towards end 'b', -1 towards 'a'.
    returns pixel list and end descriptor"""
    G = C.G
    pix = G.edges[eid]['pix']
    path = list(pix[k::direction]) if direction == 1 else list(pix[k::-1])
    used = {eid}
    side_arrive = 1 if direction == 1 else 0  # node at which we arrive: b if +1
    for _ in range(maxsteps):
        node = G.node_of(eid, side_arrive)
        p = np.array(path, float)
        j = max(0, len(p) - 11)
        din = p[-1] - p[j]
        n = np.linalg.norm(din)
        din = din / n if n > 0 else np.array([1.0, 0.0])
        cands = []
        for (e2, s2) in G.madj[node]:
            if e2 == eid and s2 == side_arrive:
                continue
            if e2 in used:
                continue
            pts = G.edge_pts(e2, s2)
            t = G.tangent(pts, 10)
            cands.append((turn(din, t), e2, s2, pts))
        if not cands:
            return path, ('end', G.mnode_pos[node], din)
        cands.sort(key=lambda c: c[0])
        if len(cands) == 1 or cands[0][0] < 50:
            if len(cands) > 1 and cands[0][0] >= 50:
                return path, ('tee', G.mnode_pos[node], din)
            a, e2, s2, pts = cands[0]
            if a > 100 and len(cands) == 1:
                return path, ('end', G.mnode_pos[node], din)
            path.extend(pts[1:])
            used.add(e2)
            eid = e2
            side_arrive = 1 - s2
            continue
        return path, ('tee', G.mnode_pos[node], din)
    return path, ('end', np.array(path[-1], float), din)


def _jump(C, e, din):
    """dashed / broken connector: next skeleton pixel ahead in a narrow cone"""
    G = C.G
    L = int(4 * C.sw + 12)
    for t in range(2, L):
        for sgn in (0, 1, -1, 2, -2):
            off = sgn * 0.18 * t
            q = e + din * t + np.array([-din[1], din[0]]) * off
            x, y = int(round(q[0])), int(round(q[1]))
            if 0 <= x < C.W and 0 <= y < C.H and G.pix_edge[y, x] >= 0:
                return G.pix_edge[y, x], G.pix_idx[y, x], (x, y)
    return None


def trace_search(C, eid, k, direction, exclude=(), max_exp=300, want_head=None):
    """best-first search over the skeleton graph from (eid,k,direction); returns list of
    (cost, path, ent, port, head) for branches ending at an entity"""
    G = C.G
    import heapq
    pix = G.edges[eid]['pix']
    path0 = list(pix[k:]) if direction == 1 else list(pix[k::-1])
    heap = [(0.0, 0, path0, eid, 1 if direction == 1 else 0, frozenset([eid]), 0)]
    res = []
    cnt = 0; uid = 1
    while heap and cnt < max_exp:
        cost, _, path, e, side_arr, used, jumps = heapq.heappop(heap)
        cnt += 1
        node = G.node_of(e, side_arr)
        p = np.array(path, float)
        j = max(0, len(p) - 11)
        din = p[-1] - p[j]; n = np.linalg.norm(din)
        din = din / n if n > 0 else np.array([1.0, 0.0])
        nxt = []
        for (e2, s2) in G.madj[node]:
            if e2 == e and s2 == side_arr: continue
            if e2 in used: continue
            pts = G.edge_pts(e2, s2)
            nxt.append((turn(din, G.tangent(pts, 10)), e2, s2, pts))
        # terminal check at this node
        ent, port, head = resolve_end(C, path, din, exclude=exclude)
        if ent is not None and (not nxt or min(a for a, *_ in nxt) > 30):
            res.append((cost, path, ent, port, head))
            continue
        if not nxt:
            jp = _jump(C, p[-1], din) if jumps < 30 else None
            if jp is not None:
                e2, k2, q = jp
                pix2 = G.edges[e2]['pix']
                # continue away from the gap
                a = np.array(pix2[max(0, k2 - 3)], float); b = np.array(pix2[min(len(pix2) - 1, k2 + 3)], float)
                dn = 1 if (b - a) @ din >= 0 else -1
                p2 = list(pix2[k2:]) if dn == 1 else list(pix2[k2::-1])
                heapq.heappush(heap, (cost + 0.3, uid, path + p2, e2, 1 if dn == 1 else 0, used | {e2}, jumps + 1)); uid += 1
            continue
        for a, e2, s2, pts in nxt:
            if a > 110: continue
            c2 = cost + (a / 90.0) ** 2 + (0.15 if len(nxt) > 1 else 0)
            if len(nxt) >= 3 and a > 45:
                c2 += CROSS_PEN  # turning at a crossing: geometric crossings do not join
            heapq.heappush(heap, (c2, uid, path + list(pts[1:]), e2, 1 - s2, used | {e2}, jumps)); uid += 1
    res.sort(key=lambda r: r[0])
    return res


def rect_snap(box, p, d):
    """point where the line through p with direction d crosses the rectangle outline nearest to p"""
    x0, y0, x1, y1 = box
    ts = []
    if abs(d[0]) > 1e-6:
        for xb in (x0, x1):
            t = (xb - p[0]) / d[0]
            y = p[1] + t * d[1]
            if y0 - 1e-6 <= y <= y1 + 1e-6:
                ts.append(t)
    if abs(d[1]) > 1e-6:
        for yb in (y0, y1):
            t = (yb - p[1]) / d[1]
            x = p[0] + t * d[0]
            if x0 - 1e-6 <= x <= x1 + 1e-6:
                ts.append(t)
    if not ts:
        return None
    t = min(ts, key=abs)
    return np.array([p[0] + t * d[0], p[1] + t * d[1]])


def resolve_end(C, path, din, exclude=()):
    """at the end of a traced connector: arrowhead? which entity? port"""
    e = np.array(path[-1], float)
    sw = C.sw
    # arrowhead blob near the end
    r = int(2 * sw + 4)
    x0, x1 = max(0, int(e[0]) - r), min(C.W, int(e[0]) + r + 1)
    y0, y1 = max(0, int(e[1]) - r), min(C.H, int(e[1]) + r + 1)
    win = C.blab[y0:y1, x0:x1] * C.heads[y0:y1, x0:x1]
    head = False
    tip = e
    if win.any():
        ids, cnt = np.unique(win[win > 0], return_counts=True)
        bid = ids[np.argmax(cnt)]
        ys, xs = np.nonzero(C.blab == bid)
        dist = np.hypot(xs - e[0], ys - e[1])
        if dist.max() > sw:
            head = True
            j = np.argmax(dist)
            tip = np.array([xs[j], ys[j]], float)
            v = tip - e; nv = np.linalg.norm(v)
            if nv > 0: din = v / nv
    ent, port = find_entity2(C.ents, tip, din, maxd=(ENT_A + 2 * sw) if head else (ENT_A - 1 + 2 * sw), exclude=exclude)
    if ent is not None and not head and len(path) > 3 * sw + 16:
        # skeleton ends hook near outlines: re-derive the port from the connector direction before the end
        k0 = int(sw + 3)
        pg = np.array(path[-1 - k0], float)
        pb = np.array(path[max(0, len(path) - 1 - k0 - 12)], float)
        dg = pg - pb; ng = np.linalg.norm(dg)
        if ng > 0:
            dg = dg / ng
            for t in np.arange(0, 4 * sw + 16, 0.5):
                q = (float(pg[0] + dg[0] * t), float(pg[1] + dg[1] * t))
                if cv2.pointPolygonTest(ent.cnt, q, True) >= -1.5:
                    port = np.array(q); break
    if ent is not None and RECT_SNAP and getattr(ent, 'fill', 0) >= 0.93:
        dd_ = din
        if not head and len(path) > 3 * sw + 16:
            k0 = int(sw + 3)
            pg = np.array(path[-1 - k0], float); pb = np.array(path[max(0, len(path) - 1 - k0 - 12)], float)
            v_ = pg - pb
            if np.linalg.norm(v_) > 0:
                dd_ = v_ / np.linalg.norm(v_)
        q = rect_snap(ent.box, port, dd_)
        if q is not None and np.linalg.norm(q - port) <= 2 * sw + 4:
            port = q
    return ent, port, head


def poly(pts, eps=1.5):
    s = simplify([tuple(map(float, q)) for q in pts], eps)
    return s


def ray_entity(C, J, v, sw, exclude, maxlen=600, cover=0.8):
    hits = 0; tot = 0; gap = 0
    for t in range(int(sw + 2), maxlen):
        q = J + v * t
        xi, yi = int(round(q[0])), int(round(q[1]))
        if not (0 <= xi < C.W and 0 <= yi < C.H):
            return None
        if t % 2 == 0:
            ent, port = find_entity2(C.ents, q, v, maxd=1.5, exclude=exclude)
            if ent is not None:
                return (ent, port, t) if tot > 0 and hits / tot >= cover else None
        on = C.lm[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].any()
        tot += 1; hits += on
        gap = 0 if on else gap + 1
        if gap > 3 * sw + 6:
            return None
    return None


def head_near(C, p, sw):
    rr = int(3 * sw + 8)
    return bool(C.heads[max(0, int(p[1]) - rr):int(p[1]) + rr + 1, max(0, int(p[0]) - rr):int(p[0]) + rr + 1].any())


def predict_image(img, C=None):
    DBG.clear()
    if C is None:
        C = prep(img)
    G = C.G
    out = []
    done_grp = set()
    for sym in C.syms:
        if sym.get('grp') is not None and sym['grp'] in done_grp:
            continue
        c, u, rin, R = sym['c'], sym['u'], sym['rin'], sym['R']
        u_c = -u
        sw = C.sw
        # ---- control line: skeleton pixel nearest to the attachment point
        a0 = sym['a0'] if 'a0' in sym else c + u_c * (rin + sw + 2)
        best = None; bd = 1e9
        r = int(sw + 6)
        for y in range(int(a0[1]) - r, int(a0[1]) + r + 1):
            for x in range(int(a0[0]) - r, int(a0[0]) + r + 1):
                if 0 <= x < C.W and 0 <= y < C.H and G.pix_edge[y, x] >= 0:
                    dd = (x - a0[0]) ** 2 + (y - a0[1]) ** 2
                    if dd < bd:
                        bd = dd; best = (x, y)
        direct = None
        if best is None:
            # controller right next to the symbol: no connector pixels left between them
            f_ = ray_entity(C, c + u_c * (rin + sw), u_c, sw, set(), cover=0.7) if RAY_CTRL else None
            if f_ is None:
                DBG.append((tuple(c), 'no_ctrl_start')); continue
            direct = (f_[0], f_[1])
        if direct is not None:
            ctrl, cport = direct
            cpath = [tuple(map(float, cport))]
        else:
          eid = G.pix_edge[best[1], best[0]]; k = G.pix_idx[best[1], best[0]]
          pix = G.edges[eid]['pix']
          # choose direction away from the circle
          pa = np.array(pix[max(0, k - 5)], float); pb = np.array(pix[min(len(pix) - 1, k + 5)], float)
          dirn = 1 if np.linalg.norm(pb - c) > np.linalg.norm(pa - c) else -1
          rs = trace_search(C, eid, k, dirn)
          rs = [x for x in rs if not x[4]] or rs
          if not rs:
              f_ = ray_entity(C, c + u_c * (rin + sw), u_c, sw, set(), cover=0.7) if RAY_CTRL else None
              if f_ is None:
                  DBG.append((tuple(c), 'no_ctrl_entity', best)); continue
              ctrl, cport = f_[0], f_[1]
              cpath = [tuple(map(float, cport))]
          else:
              _, cpath, ctrl, cport, _ = rs[0]
              if CTRL_CROSSCHECK:
                  f_ = ray_entity(C, c + u_c * (rin + sw), u_c, sw, set(), cover=CC_COVER)
                  if f_ is not None and f_[0] is not ctrl and len(cpath) > CC_RATIO * f_[2] + 10:
                      ctrl, cport = f_[0], f_[1]
                      cpath = [tuple(map(float, cport))]
        # ---- junction on the reaction line
        J0 = sym['J0']
        best = None; bd = 1e9
        r = int(sw + 5)
        for y in range(int(J0[1]) - r, int(J0[1]) + r + 1):
            for x in range(int(J0[0]) - r, int(J0[0]) + r + 1):
                if 0 <= x < C.W and 0 <= y < C.H and C.sk[y, x] and (G.pix_edge[y, x] >= 0 or G.root[G.nodelab[y, x]] > 0):
                    ex_ = sym['excl'] if 'excl' in sym else rin + sw * 0.5
                    if (x - c[0]) ** 2 + (y - c[1]) ** 2 < ex_ ** 2:
                        continue
                    dd = (x - J0[0]) ** 2 + (y - J0[1]) ** 2
                    if dd < bd:
                        bd = dd; best = (x, y)
        if best is None:
            DBG.append((tuple(c), 'no_js')); continue
        J = np.array(best, float)
        # ---- arms from J
        arms = []
        eid = G.pix_edge[best[1], best[0]]
        if eid >= 0:
            k = G.pix_idx[best[1], best[0]]
            arms = [(eid, k, 1), (eid, k, -1)]
        else:
            # J on a node: every edge at that node
            nl = G.nodelab[best[1], best[0]]
            node = G.root[nl]
            for (e2, s2) in G.madj[node]:
                pts = G.edges[e2]['pix']
                arms.append((e2, 0 if s2 == 0 else len(pts) - 1, 1 if s2 == 0 else -1))
        # nodes right next to J: their edges are arms too (J often sits a few pixels off a corner or branch node)
        if NEAR_NODE_ARMS:
            seen_e = {(a_[0], a_[2]) for a_ in arms}
            rj = 2 * sw + 4
            for node, pos in G.mnode_pos.items():
                if abs(pos[0] - J[0]) > rj or abs(pos[1] - J[1]) > rj:
                    continue
                if math.hypot(pos[0] - J[0], pos[1] - J[1]) > rj:
                    continue
                for (e2, s2) in G.madj[node]:
                    pts = G.edges[e2]['pix']
                    k2, dn2 = (0, 1) if s2 == 0 else (len(pts) - 1, -1)
                    # do not walk back into the symbol
                    q = np.array(pts[min(len(pts) - 1, 8)] if s2 == 0 else pts[max(0, len(pts) - 9)], float)
                    if np.linalg.norm(q - c) < R + sw + 2:
                        continue
                    if (e2, dn2) in seen_e or (e2, -dn2) in seen_e and len(pts) < 2:
                        continue
                    seen_e.add((e2, dn2))
                    arms.append((e2, k2, dn2))
        inputs, outputs = [], []
        usedent = {id(ctrl)}
        for (e2, k2, dn) in arms:
            rs = trace_search(C, e2, k2, dn, exclude=usedent)
            if not rs:
                DBG.append((tuple(c), 'arm_noent', G.edges[e2]['pix'][k2], ''))
                continue
            took = 0
            for (cst, p, ent, port, head) in rs:
              if took >= MAX_PER_ARM or cst > rs[0][0] + BRANCH_DC or id(ent) in usedent:
                  continue
              took += 1
              usedent.add(id(ent))
              pl = poly([J] + p[1:] + [port])
              pl[0] = J; pl[-1] = port
              item = {'box': [float(v) for v in ent.box], 'path': [[float(a), float(b)] for a, b in pl]}
              if head:
                  outputs.append(item)
              else:
                  item['path'] = item['path'][::-1]
                  inputs.append(item)
        # arrowheads that start right at the junction (no connector pixels between J and the head)
        rr = int(2 * sw + 4)
        y0, y1 = max(0, int(J[1]) - rr), min(C.H, int(J[1]) + rr + 1)
        x0, x1 = max(0, int(J[0]) - rr), min(C.W, int(J[0]) + rr + 1)
        win = C.blab[y0:y1, x0:x1] * C.heads[y0:y1, x0:x1]
        for bid in np.unique(win[win > 0]):
            ys, xs = np.nonzero(C.blab == bid)
            dist = np.hypot(xs - J[0], ys - J[1])
            if dist.min() > 2 * sw + 3:
                continue
            j = int(np.argmax(dist))
            tip = np.array([xs[j], ys[j]], float)
            v = tip - J; nv = np.linalg.norm(v)
            if nv < 2 * sw:
                continue
            v = v / nv
            # the head must point away from J and away from the symbol
            if v @ u_c > 0.5:
                continue
            ent, port = find_entity2(C.ents, tip, v, maxd=6 + 2 * sw, exclude=usedent)
            if ent is None:
                continue
            usedent.add(id(ent))
            outputs.append({'box': [float(q) for q in ent.box], 'path': [[float(J[0]), float(J[1])], [float(port[0]), float(port[1])]]})
        if RECLASS and not outputs and len(inputs) >= 2:
            hh = [head_near(C, np.array(q_['path'][0]), sw) for q_ in inputs]
            if sum(hh) == 1:
                k_ = hh.index(True)
                q_ = inputs.pop(k_)
                q_['path'] = q_['path'][::-1]
                outputs.append(q_)
        if RAY_BOTH and not outputs and not inputs:
            tdir = np.array([-u[1], u[0]])
            res_ = []
            for sg in (1, -1):
                f_ = ray_entity(C, J, sg * tdir, sw, usedent)
                if f_ is not None:
                    res_.append((sg, f_, head_near(C, f_[1], sw)))
            if len(res_) == 2 and res_[0][2] != res_[1][2]:
                for sg, (ent, port, t), hd in res_:
                    usedent.add(id(ent))
                    if hd:
                        outputs.append({'box': [float(q_) for q_ in ent.box], 'path': [[float(J[0]), float(J[1])], [float(port[0]), float(port[1])]]})
                    else:
                        inputs.append({'box': [float(q_) for q_ in ent.box], 'path': [[float(port[0]), float(port[1])], [float(J[0]), float(J[1])]]})
        if RAY_OUTPUT and not outputs and (len(inputs) == 1 or (RAY_MULTI_IN and len(inputs) >= 1)):
            pi = np.array(inputs[0]['path'][max(0, len(inputs[0]['path']) - 2)], float)
            vi = pi - J; nvi = np.linalg.norm(vi)
            if nvi > 0:
                vf = -vi / nvi
                hits = 0; tot = 0; gap = 0; found = None
                for t in range(int(sw + 2), 600):
                    q = J + vf * t
                    xi, yi = int(round(q[0])), int(round(q[1]))
                    if not (0 <= xi < C.W and 0 <= yi < C.H):
                        break
                    ent, port = find_entity2(C.ents, q, vf, maxd=1.5, exclude=usedent) if t % 2 == 0 else (None, None)
                    if ent is not None:
                        found = (ent, port, t); break
                    on = C.lm[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].any()
                    tot += 1; hits += on
                    gap = 0 if on else gap + 1
                    if gap > 3 * sw + 6:
                        break
                if found is not None and tot > 0 and hits / tot >= 0.8:
                    ent, port, t = found
                    hd = True
                    if RAY_OUT_HEAD:
                        rr = int(3 * sw + 8)
                        hd = bool(C.heads[max(0, int(port[1]) - rr):int(port[1]) + rr + 1, max(0, int(port[0]) - rr):int(port[0]) + rr + 1].any())
                    if hd:
                        outputs.append({'box': [float(q_) for q_ in ent.box], 'path': [[float(J[0]), float(J[1])], [float(port[0]), float(port[1])]]})
        if not outputs:
            DBG.append((tuple(c), 'no_outputs', len(arms), len(inputs))); continue
        if RAY_INPUT and not inputs and len(outputs) == 1:
            # the reaction line usually continues straight through J on the input side
            po = np.array(outputs[0]['path'][min(1, len(outputs[0]['path']) - 1)], float)
            vo = po - J; nvo = np.linalg.norm(vo)
            if nvo > 0:
                vb = -vo / nvo
                hits = 0; tot = 0; gap = 0; found = None
                for t in range(int(sw + 2), 600):
                    q = J + vb * t
                    xi, yi = int(round(q[0])), int(round(q[1]))
                    if not (0 <= xi < C.W and 0 <= yi < C.H):
                        break
                    ent, port = find_entity2(C.ents, q, vb, maxd=1.5, exclude=usedent) if t % 2 == 0 else (None, None)
                    if ent is not None:
                        found = (ent, port, t); break
                    on = C.lm[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].any()
                    tot += 1; hits += on
                    gap = 0 if on else gap + 1
                    if gap > 3 * sw + 6:
                        break
                if found is not None and tot > 0 and hits / tot >= 0.8:
                    ent, port, t = found
                    inputs.append({'box': [float(q_) for q_ in ent.box], 'path': [[float(port[0]), float(port[1])], [float(J[0]), float(J[1])]]})
        if DROP_REVERSIBLE and not inputs and len(outputs) >= 2:
            DBG.append((tuple(c), 'reversible')); continue
        cp = poly([cport] + cpath[::-1] + [J])
        cp[0] = cport; cp[-1] = J
        if DEDUPE:
            dup = False
            for o in out:
                if o['control'] == sym['type'] and abs(o['junction'][0] - J[0]) <= 4 and abs(o['junction'][1] - J[1]) <= 4 \
                        and max(abs(a_ - b_) for a_, b_ in zip(o['controller'], ctrl.box)) < 3:
                    dup = True; break
            if dup:
                DBG.append((tuple(c), 'duplicate')); continue
        if sym.get('grp') is not None:
            done_grp.add(sym['grp'])
        out.append({'controller': [float(v) for v in ctrl.box], 'control': sym['type'], 'junction': [float(J[0]), float(J[1])],
                    'control_path': [[float(a), float(b)] for a, b in cp], 'inputs': inputs, 'outputs': outputs})
    return out


# ===================================================================================== cnn


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class SymNet(nn.Module):
    def __init__(self, cin=2, ncls=1, w=32):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                                 nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        self.b1 = blk(cin, w); self.b2 = blk(w, 2 * w); self.b3 = blk(2 * w, 4 * w)
        self.head = nn.Sequential(nn.Linear(4 * w * 2, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, ncls))

    def forward(self, x):
        x = self.b1(x); x = F.max_pool2d(x, 2)
        x = self.b2(x); x = F.max_pool2d(x, 2)
        x = self.b3(x)
        x = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)
        return self.head(x)


def dihedral(x, k):
    if k >= 4:
        x = torch.flip(x, dims=[3])
    return torch.rot90(x, k % 4, dims=[2, 3])


def train_net(X, y, device, epochs=20, bs=128, lr=2e-3, seed=0, ncls=1):
    set_seed(seed)
    net = SymNet(cin=X.shape[1], ncls=ncls).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(X)
    steps = epochs * max(1, (n + bs - 1) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    Xt = torch.from_numpy(X).to(device); yt = torch.from_numpy(y.astype(np.float32 if ncls == 1 else np.int64)).to(device)
    if ncls == 1:
        pos = float(y.sum()); neg = float(n - y.sum())
        pw = torch.tensor([min(10.0, neg / max(1.0, pos))], device=device)
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    net.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs].to(device)
            xb = Xt[idx]; yb = yt[idx]
            k = int(torch.randint(0, 8, (1,), generator=g))
            xb = dihedral(xb, k)
            out = net(xb)
            if ncls == 1:
                loss = F.binary_cross_entropy_with_logits(out[:, 0], yb, pos_weight=pw)
            else:
                loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sch.step()
    net.eval()
    return net


@torch.no_grad()
def predict_net(net, X, device, bs=512, tta=True):
    Xt = torch.from_numpy(X).to(device)
    outs = []
    for i in range(0, len(X), bs):
        xb = Xt[i:i + bs]
        ks = range(8) if tta else [0]
        p = sum(torch.sigmoid(net(dihedral(xb, k))[:, 0]) for k in ks) / len(ks)
        outs.append(p.cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros(0, np.float32)


# ===================================================================================== metric

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



# =====================================================================================  output hygiene
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
        vals = np.array(list(b['controller']) + list(b['junction']), float)
        if not np.all(np.isfinite(vals)):
            continue
        cp = _clean_path(b['control_path'], W, H)
        if cp is None:
            continue
        nb = {'controller': _clean_box(b['controller'], W, H), 'control': b['control'],
              'junction': _clip_pt(b['junction'], W, H), 'control_path': cp, 'inputs': [], 'outputs': []}
        for side in ('inputs', 'outputs'):
            boxes = set()
            for q in b[side][:64]:
                if not np.all(np.isfinite(np.array(q['box'], float))):
                    continue
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


# =====================================================================================  data + main
OUT_PATH = '/working/submission.csv'
DATA = None
META = None


def find_data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [here, os.getcwd(), '/working', '/working/public', '/working/data', '/data', '/kaggle/input']
    for r in roots:
        if os.path.isfile(os.path.join(r, 'test.csv')) and os.path.isfile(os.path.join(r, 'diagrams.jsonl')):
            return r
    for r in ['/working', '/kaggle/input', here, os.getcwd()]:
        for h in sorted(glob.glob(os.path.join(r, '**', 'diagrams.jsonl'), recursive=True)):
            dd = os.path.dirname(h)
            if os.path.isfile(os.path.join(dd, 'test.csv')):
                return dd
    raise FileNotFoundError('data directory with test.csv and diagrams.jsonl not found')


def load_img_path(did):
    m = META[did]
    img = cv2.imread(os.path.join(DATA, m['image_path']), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(m['image_path'])
    return img


def _cands_job(args):
    """stage 1 for one diagram: candidate symbols, normalised crops, labels (labelled diagrams only)"""
    did, bindings = args
    img = load_img_path(did)
    B = prep_base(img)
    cl, crops = circle_candidates(B, use_rings=True)
    lab = np.zeros(len(cl), np.int64)
    if bindings is not None:
        ref = [b for b in bindings if b['control'] == 'catalysis']
        for k, (x, y, rin) in enumerate(cl):
            for b in ref:
                if math.hypot(b['junction'][0] - x, b['junction'][1] - y) <= rin + 1.5 * B.sw + 7:
                    lab[k] = 1; break
    return did, cl, crops, lab


def _final_job(args):
    """stage 2 for one diagram: accepted symbols -> traced bindings at each threshold"""
    did, cl, p, thrs = args
    img = load_img_path(did)
    H, W = img.shape[:2]
    B = prep_base(img)
    outs = []
    for t in thrs:
        acc = [c for c, pp in zip(cl, p) if pp >= t]
        C = prep_finish(B, acc)
        outs.append(clean_bindings(predict_image(None, C), W, H))
    return did, outs


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'bindings'])
        for rid, B in rows:
            w.writerow([rid, json.dumps(B, separators=(',', ':'))])


def main():
    global DATA, META
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA = find_data_dir()
    META = {}
    with open(os.path.join(DATA, 'diagrams.jsonl')) as f:
        for line in f:
            if line.strip():
                r = json.loads(line); META[r['diagram_id']] = r
    test = pd.read_csv(os.path.join(DATA, 'test.csv'))
    write_csv([(rid, []) for rid in test['id']], OUT_PATH)  # valid fallback first
    print('data', DATA, 'device', device, 'test rows', len(test), flush=True)
    tr = pd.read_csv(os.path.join(DATA, 'train.csv'))
    va = pd.read_csv(os.path.join(DATA, 'validation.csv'))
    jobs = [(r.diagram_id, json.loads(r.bindings)) for r in tr.itertuples()] + \
           [(r.diagram_id, json.loads(r.bindings)) for r in va.itertuples()] + \
           [(did, None) for did in test['diagram_id']]
    with Pool(N_PROC) as pool:
        cres = {r[0]: r for r in pool.map(_cands_job, jobs)}
    def stack(ids):
        X = np.concatenate([cres[i][2] for i in ids if len(cres[i][2])], 0)
        y = np.concatenate([cres[i][3] for i in ids if len(cres[i][3])], 0)
        return X, y
    tr_ids = list(tr['diagram_id']); va_ids = list(va['diagram_id'])
    lab_ids = tr_ids + va_ids
    lref = {r.diagram_id: json.loads(r.bindings) for r in pd.concat([tr, va]).itertuples()}
    # ---- held-out selection: K-fold over whole labelled diagrams, out-of-fold symbol probabilities,
    #      then the exact metric of the full pipeline at each candidate threshold
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(lab_ids))
    fold_of = {lab_ids[j]: int(k % N_FOLDS) for k, j in enumerate(perm)}
    oof = {}
    for f in range(N_FOLDS):
        fit_ids = [i for i in lab_ids if fold_of[i] != f]
        X, y = stack(fit_ids)
        net = train_net(X, y, device, epochs=EPOCHS, seed=SEED + f)
        for i in lab_ids:
            if fold_of[i] == f:
                oof[i] = predict_net(net, cres[i][2], device) if len(cres[i][2]) else np.zeros(0)
        print('fold %d fitted on %d candidates (%d positive)' % (f, len(X), int(y.sum())), flush=True)
    jobs = [(i, cres[i][1], oof[i], THRS) for i in lab_ids]
    with Pool(N_PROC) as pool:
        lres = dict(pool.map(_final_job, jobs))
    vs = []
    for k, t in enumerate(THRS):
        sc = [diagram_score(lres[i][k], lref[i]) for i in lab_ids]
        vs.append(float(np.mean(sc)))
        print('out-of-fold labelled diagrams %d  threshold %.2f  score %.4f  (train %.4f, validation %.4f)' % (
            len(lab_ids), t, vs[-1], float(np.mean(sc[:len(tr_ids)])), float(np.mean(sc[len(tr_ids):]))), flush=True)
    kbest = int(np.argmax(vs))
    thr = THRS[kbest]
    print('selected threshold', thr, flush=True)
    # ---- final model on all labelled diagrams
    X, y = stack(tr_ids + va_ids)
    net = train_net(X, y, device, epochs=EPOCHS, seed=SEED)
    tids = list(test['diagram_id'])
    probs = {i: (predict_net(net, cres[i][2], device) if len(cres[i][2]) else np.zeros(0)) for i in tids}
    nonfinite = int(sum((~np.isfinite(p)).sum() for p in probs.values()))
    print('non-finite symbol probabilities', nonfinite, flush=True)
    if nonfinite:
        raise SystemExit('non-finite model output')
    jobs = [(i, cres[i][1], probs[i], [thr]) for i in tids]
    with Pool(N_PROC) as pool:
        tres = dict(pool.map(_final_job, jobs))
    rows = [(rid, tres[did][0]) for rid, did in zip(test['id'], test['diagram_id'])]
    write_csv(rows, OUT_PATH)
    print('wrote', OUT_PATH, 'rows', len(rows), 'bindings', sum(len(b) for _, b in rows), flush=True)


if __name__ == '__main__':
    main()
