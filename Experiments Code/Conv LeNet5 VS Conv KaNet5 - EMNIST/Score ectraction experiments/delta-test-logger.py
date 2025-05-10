#!/usr/bin/env python3
"""
Heat-map and Batch-score quality analysis (len-mismatch tolerant)
================================================================
* Scansione → metriche (heat-map + batch scorer) → CSV.
* Mantiene delta = 0 per l’ipotesi “equal”.

CLI
----
--variant {None|L2|all}        : analizza solo None, solo L2 o entrambe
--quick-test                   : epoca 1, prime 10 istanze (debug veloce)
--only-abs-ordinal             : esegue solo le metriche aggiuntive abs_ordinal senza raw e norm_[0,1]
--only-z-score                 : esegue solo le metriche aggiuntive z-score_[0,1] senza raw, norm e abs_ordinal
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import threading
import time
from itertools import product
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from scipy.stats import t as t_dist, ttest_rel
from skimage.morphology import skeletonize

# ————————————— configurazione generale —————————————
ROOT = Path("/home/magliolo/KAN/ConvolutionalKans/Convolutional-KANs/results")

TYPES  = {"featuremap": "FeatureMap",
          "gradcam":    "GradCAM"}

SUFFIX = {"featuremap": "_fmap_up.npy",
          "gradcam":    "_gcam_up.npy"}

VARIANTS_ALL = ["None", "L2"]

MODEL_DIRS   = {"Standard_LeNet5": (0, 0),
                "KaNet5":         (5, 3)}

MODELS       = list(MODEL_DIRS.keys())
TOTAL_EPOCHS = 50

ADDITIONAL_METRICS = {
    "HeatxEdge": "heatxedge",
    "HeatxCorner": "heatxcorner",
    "HeatxLoop": "heatxloop",
    "HeatxORL_Mask": "heatxorl_mask",
    "HeatxVCR_Mask": "heatxvcr_mask",
    "Segmentation_heat_loop": "segmentation_heat_loop",
}

TEST_METRICS = list(ADDITIONAL_METRICS.values())

# ora gestiamo i tipi raw, norm, abs_ordinal e zscore
METRIC_TYPES = ["raw", "norm", "abs_ordinal", "zscore"]

P_UPPER = 0.05
rng = np.random.default_rng(0)

# ————————————— logging semplificato —————————————
_log_lock = threading.Lock()

def configure_logging(logfile: str, level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(logfile, mode="w"),
            logging.StreamHandler()
        ]
    )

def log_print(level: str, msg: str):
    with _log_lock:
        getattr(logging, level)(msg)
        print(msg, flush=True)

# ————————————— heartbeat —————————————
def heartbeat(pool: mp.pool.Pool):
    log_print("info", "Heartbeat thread started")
    while True:
        if not any(p.is_alive() for p in pool._pool):
            break
        log_print("info", "Heartbeat: estrazione metriche in corso…")
        time.sleep(60)

# ————————————— helper punteggi pixel —————————————
def blur_score(img: np.ndarray) -> np.ndarray:
    return np.clip(1 - 4 * (img - 0.5) ** 2, 0, 1)

def corner_score(gray: np.ndarray) -> np.ndarray:
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    s = skeletonize((b > 0).astype(np.uint8)).astype(np.uint8)
    k = np.array([[1,1,1],[1,10,1],[1,1,1]])
    c = convolve(s, k, mode="constant", cval=0)
    r = np.maximum(c-10,0)
    return np.zeros_like(r) if r.max()==0 else r/r.max()

def delta_percent(a: float, b: float) -> float:
    return np.nan if b==0 else (a-b)/abs(b)*100

# ————————————— paired t-test —————————————
def paired_ttest(a: np.ndarray, b: np.ndarray, alt: str) -> Tuple[float,float]:
    if len(a)==0 or len(a)!=len(b):
        return np.nan, np.nan
    if len(a)==1:
        return np.nan, delta_percent(a[0], b[0])
    if alt == "equal":
        return ttest_rel(a, b, nan_policy="omit").pvalue, 0.0

    sign, low, high = 1, -100.0, 100.0
    best_p = best_d = np.nan
    for _ in range(100):
        mid = (low + high) / 2
        f = 1 + sign * mid
        diff = a.mean() - f * b.mean()
        var = a.var(ddof=1) + f*f*b.var(ddof=1) - 2*f*np.cov(a,b,ddof=1)[0,1]
        if var == 0:
            break
        t = diff / (np.sqrt(var) / np.sqrt(len(a)))
        p = 1 - t_dist.cdf(t, len(a)-1)
        best_p, best_d = p, sign * mid * 100
        if 0.01 <= p <= P_UPPER:
            break
        if p > P_UPPER:
            high = mid
        else:
            low = mid
    return best_p, best_d

# ————————————— container runtime —————————————
FILES: dict[tuple[int,str,str,str,str], Path] = {}
VALID_PREFIXES: dict[int, set[str]] = {}
EPOCH_RANGE: range = range(1)

# ————————————— metric task (Pool) —————————————
def metric_task(params: tuple[int,str,str,str,bool,bool,bool]):
    epoch, variant, model, htype, quick, only_abs, only_z = params
    log_print("info", f"[metric_task] start epoch={epoch}, variant={variant}, model={model}, type={htype}")
    rows = []
    for prefix in VALID_PREFIXES.get(epoch, set()):
        path = FILES.get((epoch, prefix, variant, model, htype))
        if path is None:
            continue
        try:
            heat = np.load(path, mmap_mode="r").astype(np.float32)
        except Exception as exc:
            log_print("error", f"Metric load FAILED {path}: {exc}")
            continue

        vmax = heat.max()
        fimg = heat / vmax if vmax else heat
        gray = (fimg * 255).astype(np.uint8)
        valid = heat > 0
        if not valid.any():
            continue

        b = blur_score(fimg)
        c = corner_score(gray)

        row = {
            "epoch": epoch, "variant": variant, "model": model, "map_type": htype,
            "raw_corner": (heat * c).mean(),
            "raw_blur": (heat * b)[valid].mean(),
            "norm_corner": (fimg * c).mean(),
            "norm_blur": (fimg * b).mean(),
        }
        # includiamo abs_ordinal e z-score in base alle opzioni
        for m_name, col in ADDITIONAL_METRICS.items():
            if only_z:
                subs = [("z-score_[0,1]", "zscore")]
            elif only_abs:
                subs = [("abs_ordinal", "abs_ordinal")]
            else:
                subs = [
                    ("nonorm", "raw"),
                    ("norm_[0,1]", "norm"),
                    ("abs_ordinal", "abs_ordinal"),
                    ("z-score_[0,1]", "zscore"),
                ]
            for sub, flag in subs:
                fp = path.parent / m_name / sub / path.name
                if fp.exists():
                    d = np.load(fp, mmap_mode="r")
                    row[f"{flag}_{col}"] = float(d) if d.ndim == 0 else d.sum(dtype=np.float64)

        rows.append(row)
        if quick and epoch == 1 and len(rows) >= 10:
            break

    return rows

# ————————————— scan files —————————————
def scan_files(variants: list[str], quick: bool):
    global FILES, VALID_PREFIXES
    FILES.clear()
    count: dict[tuple[int,str], int] = {}
    exp = len(variants) * len(MODELS) * len(TYPES)

    for v, m, ht in product(variants, MODEL_DIRS, TYPES):
        g, s = MODEL_DIRS[m]
        base = ROOT / f"results_{v}_SGD_lr0.01_{g}_{s}" / m / TYPES[ht]
        for ep in EPOCH_RANGE:
            fold = base / f"epoch_{ep}" / "extracted_random_examples" / "Upsampled"
            if not fold.exists():
                continue
            for f in fold.glob(f"*{SUFFIX[ht]}"):
                pre = "_".join(f.stem.split("_")[:-2])
                FILES[(ep, pre, v, m, ht)] = f
                count[(ep, pre)] = count.get((ep, pre), 0) + 1

    VALID_PREFIXES = {
        ep: {p for (ep_, p), c in count.items() if ep_ == ep and c == exp}
        for ep in EPOCH_RANGE
    }
    if quick and 1 in VALID_PREFIXES:
        VALID_PREFIXES[1] = set(list(VALID_PREFIXES[1])[:10])
        for ep in list(VALID_PREFIXES):
            if ep != 1:
                VALID_PREFIXES[ep] = set()

# ————————————— length equalizer —————————————
def equalize_lengths(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(a) == len(b):
        return a, b
    m = min(len(a), len(b))
    if m == 0:
        return a[:0], b[:0]
    if len(a) > len(b):
        a = np.delete(a, rng.choice(len(a), len(a) - m, False))
    else:
        b = np.delete(b, rng.choice(len(b), len(b) - m, False))
    return a, b

# ————————————— CSV worker (thread) —————————————
class PercentMeter:
    def __init__(self, total: int):
        self.total = max(total, 1)
        self.done = 0
        self.last_pct = -1
        self.last_time = time.time()
        self.lock = threading.Lock()

    def tick(self, thread_name: str, last_file: str):
        with self.lock:
            self.done += 1
            pct = int(self.done * 100 / self.total)
            now = time.time()
            if pct > self.last_pct or now - self.last_time >= 60:
                self.last_pct, self.last_time = pct, now
                rem = self.total - self.done
                log_print("info",
                          f"[{thread_name}] {pct}% "
                          f"({self.done}/{self.total}) – ultimo: {last_file} "
                          f"| mancanti: {rem}")

def csv_worker(task):
    v, ht, flag, met, alt, df, meter = task
    log_print("info", f"[csv_worker] start variant={v}, map={ht}, flag={flag}, metric={met}, alt={alt}")
    model_a, model_b = "Standard_LeNet5", "KaNet5"
    sym = {"equal": "=", "greater": ">"}
    col = f"{flag}_{met}"

    # base directory per i file di score grezzi e CSV, con struttura scores/<metric>
    base_dir = Path("scores") / met
    base_dir.mkdir(parents=True, exist_ok=True)

    if col not in df.columns:
        meter.tick(threading.current_thread().name, f"SKIP-{v}-{ht}-{met}-{flag}")
        return

    rows = []
    # loop per epoca
    for ep in EPOCH_RANGE:
        ma = (df.variant == v) & (df.map_type == ht) & (df.model == model_a) & (df.epoch == ep)
        mb = (df.variant == v) & (df.map_type == ht) & (df.model == model_b) & (df.epoch == ep)
        a_vals, b_vals = equalize_lengths(df.loc[ma, col].values, df.loc[mb, col].values)

        # salvataggio vettori .npy in sottocartelle <metric>/<model>
        epoch_str = f"epoch_{ep}"
        model_a_dir = base_dir / model_a
        model_a_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_a_dir / f"{v}_{ht}_{col}_{model_a}_{epoch_str}.npy", a_vals)
        model_b_dir = base_dir / model_b
        model_b_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_b_dir / f"{v}_{ht}_{col}_{model_b}_{epoch_str}.npy", b_vals)

        # statistica
        if len(a_vals) == 0:
            p, d, sign = np.nan, np.nan, "NaN"
        else:
            p, d = paired_ttest(a_vals, b_vals, alt)
            sign = "positive" if d > 0 else "negative" if d < 0 else "zero"
        rows.append({"epoch": ep, "p_value": p, "delta": d, "delta_sign": sign})

    # totale di tutte le epoche
    a_tot = df.loc[(df.variant == v) & (df.map_type == ht) & (df.model == model_a), col].values
    b_tot = df.loc[(df.variant == v) & (df.map_type == ht) & (df.model == model_b), col].values
    a_tot, b_tot = equalize_lengths(a_tot, b_tot)

    model_a_dir = base_dir / model_a
    np.save(model_a_dir / f"{v}_{ht}_{col}_{model_a}_all.npy", a_tot)
    model_b_dir = base_dir / model_b
    np.save(model_b_dir / f"{v}_{ht}_{col}_{model_b}_all.npy", b_tot)

    if len(a_tot):
        p_tot, d_tot = paired_ttest(a_tot, b_tot, alt)
        sign_tot = "positive" if d_tot > 0 else "negative" if d_tot < 0 else "zero"
    else:
        p_tot, d_tot, sign_tot = np.nan, np.nan, "NaN"
    rows.append({"epoch": "Total", "p_value": p_tot, "delta": d_tot, "delta_sign": sign_tot})

    fname = base_dir / f"results_{v}_{ht}_{flag}_{met}_{model_a}_vs_{model_b}_{sym[alt]}.csv"
    pd.DataFrame(rows).to_csv(fname, index=False, na_rep="NaN", float_format="%.8e")
    meter.tick(threading.current_thread().name, str(fname))

# ————————————— main —————————————
def main(variants: list[str], quick: bool, only_abs: bool, only_z: bool):
    global EPOCH_RANGE
    EPOCH_RANGE = range(1, 2) if quick else range(1, TOTAL_EPOCHS + 1)

    configure_logging("heatmap_analyser.log", level=logging.INFO)
    log_print("info", "Program started")

    scan_files(variants, quick)

    params_mp = [
        (e, v, m, t, quick, only_abs, only_z)
        for e, v, m, t in product(EPOCH_RANGE, variants, MODELS, TYPES)
    ]
    pool = mp.Pool(cpu_count())
    threading.Thread(target=heartbeat, args=(pool,), daemon=True).start()

    recs = []
    try:
        for sub in pool.imap_unordered(metric_task, params_mp):
            recs.extend(sub)
    finally:
        pool.close()
        pool.join()

    if not recs:
        log_print("error", "Nessuna metrica raccolta.")
        return

    df = pd.DataFrame.from_records(recs)

    if only_abs:
        flags_list = ["abs_ordinal"]
    elif only_z:
        flags_list = ["zscore"]
    else:
        flags_list = METRIC_TYPES

    combos = [
        (v, ht, flag, met, alt)
        for v, ht, flag, met, alt in product(variants, TYPES, flags_list, TEST_METRICS, ["equal", "greater"])
    ]
    meter = PercentMeter(total=len(combos))
    tasks = [(v, ht, flag, met, alt, df, meter) for (v, ht, flag, met, alt) in combos]

    with ThreadPoolExecutor(max_workers=len(tasks)) as exe:
        futures = [exe.submit(csv_worker, t) for t in tasks]
        for _ in as_completed(futures):
            pass

    log_print("info", "Analisi completata.")

# ————————————— entrypoint —————————————
if __name__ == "__main__":
    mp.set_start_method("fork")
    parser = argparse.ArgumentParser(description="Heat-map + Batch-score analyser")
    parser.add_argument("--variant", choices=["None", "L2", "all"], default="all")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument(
        "--only-abs-ordinal",
        action="store_true",
        dest="only_abs_ordinal",
        help="Esegue solo le metriche aggiuntive abs_ordinal senza raw e norm_[0,1]"
    )
    parser.add_argument(
        "--only-z-score",
        action="store_true",
        dest="only_z_score",
        help="Esegue solo le metriche aggiuntive z-score_[0,1] senza raw, norm e abs_ordinal"
    )
    args = parser.parse_args()

    selected = VARIANTS_ALL if args.variant == "all" else [args.variant]
    main(selected, args.quick_test, args.only_abs_ordinal, args.only_z_score)
