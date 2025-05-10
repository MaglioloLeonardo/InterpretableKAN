#!/usr/bin/env python3
"""
Batch scorer – multithread con progressi per‑cento
=================================================
Per ogni heat‑map vengono calcolate 6 metriche × varianti.
**Sei thread**, uno per metrica, lavorano in parallelo; ciascun thread
stampa (stdout + `batch_scorer.log`) **ogni punto percentuale** di avanzamento:

```
[Thread‑140261055047424] HeatxEdge – 34 % (done/total) | path
```

`Segmentation_heat_loop` salva un intero (#cluster, 1‑10).

Nuova opzione `--types`: `nonorm`, `norm_[0,1]`, `abs_ordinal` per specificare
il tipo di trasformazione del heatmap applicata a tutte le metriche.
"""
from __future__ import annotations
import argparse
import logging
import os
import re
import struct
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

###############################################################################
# Config                                                                      #
###############################################################################
ROOT = Path("/home/magliolo/KAN/ConvolutionalKans/Convolutional-KANs/results")
AVAILABLE_VARIANTS = ["None", "L2"]
MODEL_DIRS = {"Standard_LeNet5": (0, 0), "KaNet5": (5, 3)}
TYPES_MAP = {"featuremap": "FeatureMap", "gradcam": "GradCAM"}
SUFFIX = {"featuremap": "_fmap_up.npy", "gradcam": "_gcam_up.npy"}
TOTAL_EPOCHS = 50

DATA_DIR = os.path.expanduser("~/.cache/emnist/gzip/")
TRAIN_IDX = os.path.join(DATA_DIR, "emnist-byclass-train-images-idx3-ubyte")
TEST_IDX = os.path.join(DATA_DIR, "emnist-byclass-test-images-idx3-ubyte")

###############################################################################
# Log setup                                                                   #
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("batch_scorer.log", "w"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("batch_scorer")
log_lock = threading.Lock()

def log_progress(msg: str):
    with log_lock:
        log.info(msg)
        print(msg, flush=True)

###############################################################################
# EMNIST loader                                                               #
###############################################################################

def read_idx_header(fp: str):
    with open(fp, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
    return n, r, c

TRAIN_N, IMG_H, IMG_W = read_idx_header(TRAIN_IDX)
TEST_N, _, _ = read_idx_header(TEST_IDX)

def read_idx_image(fp: str, idx: int) -> np.ndarray:
    with open(fp, "rb") as f:
        f.seek(16 + idx * IMG_H * IMG_W)
        return np.frombuffer(f.read(IMG_H * IMG_W), dtype=np.uint8).reshape(IMG_H, IMG_W)

def get_emnist_image(idx: int) -> np.ndarray:
    return read_idx_image(
        TRAIN_IDX if idx < TRAIN_N else TEST_IDX,
        idx if idx < TRAIN_N else idx - TRAIN_N
    )

###############################################################################
# Image utilities                                                             #
###############################################################################

def mask_hierarchy_clean(bin_img):
    cnts, hier = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(bin_img, np.uint8)
    if hier is not None:
        for i, h in enumerate(hier[0]):
            if h[3] != -1:
                cv2.drawContours(m, cnts, i, 255, -1)
    hole = m.astype(bool) & (bin_img == 0)
    hole[[0,-1],:] = hole[:,[0,-1]] = False
    return hole


def convexity_defects_opening_mask(img_gray):
    _, b = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    b = (b // 255).astype(np.uint8)
    d = cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), 1)
    cnts, _ = cv2.findContours(d*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros_like(b)
    cnt = cnts[0]
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) <= 3:
        return np.zeros_like(b)
    defects = cv2.convexityDefects(cnt, hull)
    m = np.zeros_like(b)
    if defects is not None:
        for s,e,f,_ in defects[:,0]:
            cv2.fillConvexPoly(m, np.array([cnt[s][0], cnt[f][0], cnt[e][0]]), 1)
    e = cv2.Canny(img_gray, 30, 100)
    e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), 1)
    m[e>0] = 0
    return m


def vcr_mask(orig_n):
    h,w = orig_n.shape
    lvccr = np.zeros(w, np.float32)
    for j in range(w):
        col = (orig_n[:,j] >= 0.1).astype(int)
        edges = np.concatenate(([col[0]], col[:-1] != col[1:], [True]))
        idxs = np.where(edges)[0]
        runs = np.diff(idxs)[::2]
        lvccr[j] = (runs.max() if runs.size else 0) / h
    mask_col = lvccr >= np.percentile(lvccr, 90)
    return np.repeat(mask_col[None,:], h, axis=0)


def cluster_quantile(grad, mask, K=10):
    lbl = np.zeros_like(grad, dtype=int)
    vals = grad[mask]
    if vals.size == 0:
        return lbl
    thr = [np.percentile(vals, 100*i/K) for i in range(1,K)]
    seg = np.digitize(vals, thr)
    lbl[mask] = seg
    return lbl

###############################################################################
# Scoring functions per metric                                                #
###############################################################################

def common_features(orig_n):
    edge = np.abs(cv2.Sobel(orig_n, cv2.CV_32F, 1, 0, ksize=3)); edge /= edge.max() or 1
    corner = cv2.cornerMinEigenVal(orig_n, 2, 3); corner /= corner.max() or 1
    _, bin_char = cv2.threshold((orig_n*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    loop_m = mask_hierarchy_clean(bin_char)
    orl_m = convexity_defects_opening_mask((orig_n*255).astype(np.uint8))
    vcr_m = vcr_mask(orig_n)
    return edge, corner, loop_m, orl_m, vcr_m


def heatxedge(h, e, *rest):      return h * e

def heatxcorner(h, _, c, *rest): return h * c

def heatxloop(h, _, __, l, *rest): return h * l

def heatxorl(h, _, __, ___, o, *rest): return h * o

def heatxvcr(h, _, __, ___, ____, v): return h * v

def segmentation(h, _, __, l, *rest):
    lbl = cluster_quantile(h * l, l, K=10)
    return np.array(len(np.unique(lbl[lbl>0])), dtype=np.int32)

###############################################################################
# Scoring functions per metric                                                #
###############################################################################

# Mappa metriche
SCORES = {
    "HeatxEdge":       heatxedge,
    "HeatxCorner":     heatxcorner,
    "HeatxLoop":       heatxloop,
    "HeatxORL_Mask":   heatxorl,
    "HeatxVCR_Mask":   heatxvcr,
    "Segmentation_heat_loop": segmentation,
}

###############################################################################
# Transform types                                                              #
###############################################################################

def abs_ordinal_transform(heat, *_):
    abs_heat = np.abs(heat)
    flat = abs_heat.flatten()
    order = np.argsort(flat)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, flat.size + 1, dtype=np.int32)
    return ranks.reshape(abs_heat.shape)


def abs_zscore_transform(raw, *_):
    # 1) modulo per eliminare il segno
    abs_raw = np.abs(raw)
    # 2) z-score sul valore assoluto
    mean = abs_raw.mean()
    std = abs_raw.std() if abs_raw.std() != 0 else 1.0
    z = (abs_raw - mean) / std
    # 3) min-max normalization su [0,1] di z
    minz, maxz = z.min(), z.max()
    denom = (maxz - minz) if (maxz - minz) != 0 else 1.0
    return (z - minz) / denom

TRANSFORMS = {
    "nonorm":       lambda raw, norm, *_: raw,
    "norm_[0,1]":   lambda raw, _, *__: (
        (lambda hmin, hmax: (raw - hmin) / ((hmax - hmin) if (hmax - hmin) != 0 else 1.0))
    )(raw.min(), raw.max()),
    "abs_ordinal":  abs_ordinal_transform,
    "z-score_[0,1]": abs_zscore_transform,
}

###############################################################################
# Heatmap list                                                                #
###############################################################################

def list_heatmaps(variants: list[str]) -> list[tuple[Path,int]]:
    out = []
    rx = re.compile(r"idx_(\d+)")
    for var in variants:
        for model,(g,s) in MODEL_DIRS.items():
            for htype,suf in SUFFIX.items():
                base = ROOT / f"results_{var}_SGD_lr0.01_{g}_{s}" / model / TYPES_MAP[htype]
                for epoch in range(1, TOTAL_EPOCHS+1):
                    d = base / f"epoch_{epoch}" / "extracted_random_examples" / "Upsampled"
                    if not d.exists(): continue
                    for fp in d.glob(f"*{suf}"):
                        m = rx.search(fp.stem)
                        if m:
                            out.append((fp, int(m.group(1))))
    return out

###############################################################################
# Worker per metrica                                                          #
###############################################################################

def metric_worker(metric_name: str, func, types: list[str], variants: list[str]):
    tid = threading.get_ident()
    heatmaps = list_heatmaps(variants)
    total = len(heatmaps) * len(types)
    done = 0
    last_pct = -1
    for fp, idx in heatmaps:
        heat_raw = np.load(fp).astype(np.float32)
        heat_norm = heat_raw / heat_raw.max() if heat_raw.max() else heat_raw
        orig = get_emnist_image(idx)
        orig_n = np.rot90(np.fliplr(orig), k=1).astype(np.float32) / 255.0
        feat = common_features(orig_n)
        for t in types:
            heat = TRANSFORMS[t](heat_raw, heat_norm, *feat)
            result = func(heat, *feat)
            out_dir = fp.parent / metric_name / t
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / fp.name
            np.save(out_path, result.astype(np.float32) if hasattr(result, "ndim") else result)
            done += 1
            pct = int(done * 100 / total)
            if pct > last_pct:
                last_pct = pct
                log_progress(f"[Thread-{tid}] {metric_name} – {pct}% ({done}/{total}) | {out_path.relative_to(ROOT)}")

###############################################################################
# Main                                                                        #
###############################################################################

def main(selected_metrics: dict[str, callable], selected_types: list[str], variants: list[str]):
    n = len(list_heatmaps(variants)) * len(selected_types)
    log.info("Heat‑map totali: %d (metriche: %s, tipi: %s, variants: %s)", n, list(selected_metrics.keys()), selected_types, variants)
    with ThreadPoolExecutor(max_workers=len(selected_metrics)) as exe:
        for name, fn in selected_metrics.items():
            exe.submit(metric_worker, name, fn, selected_types, variants)
    log.info("Terminato.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch scorer – multithread con opzioni metrics, types e variants")
    parser.add_argument(
        "--metrics", nargs='+', choices=list(SCORES.keys()),
        default=list(SCORES.keys()),
        help="Seleziona le metriche da calcolare (default: tutte)")
    parser.add_argument(
        "--variants", nargs='+', choices=AVAILABLE_VARIANTS,
        default=AVAILABLE_VARIANTS,
        help="Seleziona le varianti da processare (default: tutte)")
    parser.add_argument(
        "--types", nargs='+', choices=list(TRANSFORMS.keys()),
        default=list(TRANSFORMS.keys()),
        help="Seleziona i tipi di trasformazione (default: tutte)")
    args = parser.parse_args()
    VARIANTS = args.variants
    TYPES = args.types
    METRICS = {name: SCORES[name] for name in args.metrics}
    main(METRICS, TYPES, VARIANTS)
