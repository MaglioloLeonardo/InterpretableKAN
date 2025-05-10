import os
import re
import random
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import struct
from skimage.morphology import closing, square

# --- ROOT fisso per heatmap files ---
ROOT = Path("/home/magliolo/KAN/ConvolutionalKans/Convolutional-KANs/results")
VARIANTS = ["None"]
MODEL_DIRS = {"Standard_LeNet5": (0, 0), "KaNet5": (5, 3)}
MODEL_NAMES = {"Standard_LeNet5": "LeNet5 No Norm", "KaNet5": "KaNet5 No Norm"}
TYPES = {"featuremap": "FeatureMap", "gradcam": "GradCAM"}
SUFFIX = {"featuremap": "_fmap_up.npy", "gradcam": "_gcam_up.npy"}
TOTAL_EPOCHS = 50
HEATMAP_CMAP = plt.get_cmap('jet')

# Percorso dati idx EMNIST
DATA_DIR = '/home/magliolo/.cache/emnist/gzip/'
TRAIN_PATH = os.path.join(DATA_DIR, 'emnist-byclass-train-images-idx3-ubyte')
TEST_PATH  = os.path.join(DATA_DIR, 'emnist-byclass-test-images-idx3-ubyte')

# --- Directory di output e upscaling ---
OUTPUT_DIR = Path.cwd() / "saved_images"
UPSCALE_FACTOR = 8

def read_idx_header(fp: str):
    with open(fp, 'rb') as f:
        _, num, rows, cols = struct.unpack('>IIII', f.read(16))
    return num, rows, cols

train_n, IMG_ROWS, IMG_COLS = read_idx_header(TRAIN_PATH)
test_n, _, _               = read_idx_header(TEST_PATH)

def read_idx_image(fp: str, idx: int) -> np.ndarray:
    with open(fp, 'rb') as f:
        f.seek(16 + idx * IMG_ROWS * IMG_COLS)
        buf = f.read(IMG_ROWS * IMG_COLS)
    return np.frombuffer(buf, dtype=np.uint8).reshape(IMG_ROWS, IMG_COLS)

def get_emnist_image(idx: int) -> np.ndarray:
    if idx < train_n:
        return read_idx_image(TRAIN_PATH, idx)
    else:
        return read_idx_image(TEST_PATH, idx - train_n)

def mask_hierarchy_clean(bin_img):
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bin_img, np.uint8)
    if hierarchy is None: return mask.astype(bool)
    for i,h in enumerate(hierarchy[0]):
        if h[3] != -1:
            cv2.drawContours(mask, contours, i, 255, -1)
    hole = mask.astype(bool) & (bin_img==0)
    hole[0,:]=hole[-1,:]=hole[:,0]=hole[:,-1]=False
    return hole

def convexity_defects_opening_mask(img_gray, dilate_iter=1, dilate_kernel_size=3):
    _, b = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    b = (b//255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilate_kernel_size,)*2)
    d = cv2.dilate(b,k,iterations=dilate_iter)
    cnts,_ = cv2.findContours(d*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cnts: return np.zeros_like(b)
    cnt = cnts[0]
    hull = cv2.convexHull(cnt,returnPoints=False)
    if hull is None or len(hull)<=3: return np.zeros_like(b)
    defects = cv2.convexityDefects(cnt,hull)
    mask = np.zeros_like(b)
    if defects is not None:
        for s,e,f,_ in defects[:,0]:
            tri = np.array([cnt[s][0],cnt[f][0],cnt[e][0]])
            cv2.fillConvexPoly(mask,tri,1)
    e = cv2.Canny(img_gray,30,100)
    e = cv2.dilate(e,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),iterations=1)
    mask[e>0]=0
    return mask

def cluster_quantile(grad, mask, K=3):
    lbl = np.zeros_like(grad,dtype=int)
    vals = grad[mask]
    if vals.size==0: return lbl
    thr = [np.percentile(vals,100*i/K) for i in range(1,K)]
    seg = np.digitize(vals,thr)
    lbl[mask] = seg
    return lbl

def collect_heatmap_files():
    out = {ht:[] for ht in SUFFIX}
    for var in VARIANTS:
        for m,(g,s) in MODEL_DIRS.items():
            for ht,suf in SUFFIX.items():
                base = ROOT/f"results_{var}_SGD_lr0.01_{g}_{s}"/m/TYPES[ht]
                for e in range(1,TOTAL_EPOCHS+1):
                    d = base/f"epoch_{e}"/"extracted_random_examples"/"Upsampled"
                    if d.exists(): out[ht]+=list(d.glob(f"*{suf}"))
    return out

def upscale(img):
    h,w = img.shape[:2]
    return cv2.resize(img,(w*UPSCALE_FACTOR,h*UPSCALE_FACTOR),interpolation=cv2.INTER_CUBIC)

def save_image_uint8(img, cmap, filename:Path):
    if img.dtype.kind=='f':
        img = np.clip(img,0,1)
    else:
        img = img.astype(np.float32)/255
    cmap_fn = plt.get_cmap(cmap) if isinstance(cmap,str) else cmap
    if cmap_fn:
        gray = img if img.ndim==2 else img[...,0]
        rgba = cmap_fn(gray)
        rgb = (rgba[...,:3]*255).astype(np.uint8)
    else:
        rgb = (img*255).astype(np.uint8) if img.ndim==3 else (img*255).astype(np.uint8)
    if rgb.ndim==3 and rgb.shape[2]==3:
        rgb = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(filename),rgb)

def show_comparison_and_save(sample_size=3):
    files = collect_heatmap_files()
    for ht,flist in files.items():
        fm = {m:[f for f in flist if f"/{m}/" in f.as_posix()] for m in MODEL_DIRS}
        idxs = {m:{int(re.search(r'idx_(\d+)',f.stem).group(1)) for f in fps} for m,fps in fm.items()}
        common = set.intersection(*idxs.values())
        for idx in random.sample(list(common),min(sample_size,len(common))):
            raw = get_emnist_image(idx)
            orig = np.rot90(np.fliplr(raw),k=1)
            orig_n = orig.astype(np.float32)/255
            h,w = orig_n.shape

            # calcolo LVCCR con controllo runs.size
            lvccr = np.zeros(w)
            for j in range(w):
                col = (orig_n[:,j]>=0.1).astype(int)
                edges = np.concatenate(([col[0]], (col[:-1]!=col[1:]), [True]))
                idxs_change = np.where(edges)[0]
                runs = np.diff(idxs_change)[::2]
                L = runs.max() if runs.size>0 else 0
                lvccr[j] = L/h
            vcr = lvccr >= np.percentile(lvccr,90)

            _,bin_char = cv2.threshold((orig_n*255).astype(np.uint8),0,255,cv2.THRESH_OTSU)
            loop_m = mask_hierarchy_clean(bin_char)
            orl_m  = convexity_defects_opening_mask((orig_n*255).astype(np.uint8))

            orig_rgb     = np.stack([orig_n]*3,-1)
            overlay_vcr  = orig_rgb.copy(); overlay_vcr[:,vcr,:]=[1,0,0]
            overlay_loop = np.stack([bin_char==255]*3,-1).astype(float); overlay_loop[loop_m]=[1,0,0]
            overlay_orl  = orig_rgb.copy(); overlay_orl[orl_m==1]=[1,0,0]

            edge   = np.abs(cv2.Sobel(orig_n,cv2.CV_32F,1,0,ksize=3)); edge/=edge.max() or 1
            corner = cv2.cornerMinEigenVal(orig_n,2,3); corner/=corner.max() or 1

            base = OUTPUT_DIR/ht/f"idx_{idx}"
            for model in MODEL_DIRS:
                out_m = base/model; out_m.mkdir(parents=True,exist_ok=True)
                fp = next(f for f in fm[model] if f"idx_{idx}" in f.stem)
                heat = np.rot90(np.fliplr(np.load(fp).astype(np.float32)),k=1); heat/=heat.max() or 1

                lbl3 = cluster_quantile(heat*loop_m,loop_m,3)
                cluster_img = np.zeros_like(np.stack([heat*loop_m]*3,-1))
                for v in np.unique(lbl3):
                    if v>0: cluster_img[lbl3==v]=[(230/255,25/255,75/255),
                                                   (60/255,180/255,75/255),
                                                   (255/255,225/255,25/255)][v-1]
                heat_orl = heat*orl_m

                variants = {
                    "Orig_EMNIST":           orig,
                    "Heat_norm":             heat,
                    "Corner":                corner,
                    "Edge":                  edge,
                    "Heat×Edge":             heat*edge,
                    "Heat×Corner":           heat*corner,
                    "Orig_RGB":              orig_rgb,
                    "Orig+ORL_overlay":      overlay_orl,
                    "Orig+loop_detection":   overlay_loop,
                    "Heat×Loop":             heat*loop_m,
                    "Segmentation_heat_loop":cluster_img,
                    "Heat×ORL_Mask":         heat_orl,
                    "Orig×Heat":             orig_n*heat,
                    "Orig+VCR_Mask":         overlay_vcr,
                    "FeatureMap×VCR_Mask":   heat*vcr,
                    "GradCAM×VCR_Mask":      heat*vcr,
                }

                for name,img in variants.items():
                    cmap = None
                    if name in ("Orig_EMNIST","Corner","Edge","Heat×Loop","Heat×ORL_Mask"):
                        cmap = 'gray'
                    elif name not in ("Orig_RGB","Orig+ORL_overlay","Orig+loop_detection"):
                        cmap = 'jet'
                    save_image_uint8(upscale(img), cmap, out_m/f"{model}_{name}.png")

if __name__ == "__main__":
    random.seed(42)
    show_comparison_and_save(sample_size=3)
