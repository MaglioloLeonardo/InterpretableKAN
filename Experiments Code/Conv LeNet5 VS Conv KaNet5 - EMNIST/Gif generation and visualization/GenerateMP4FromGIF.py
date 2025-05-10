


import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image

###############################################################################
# Configuration constants
###############################################################################
FPS = 10                    # frames per second in the final combined GIF
TOTAL_GIF_DIR = "results/mean/upsampled/total_gifs"  # output folder
SEPARATOR_VERTICAL = 2      # px for vertical white line between columns
SEPARATOR_HORIZONTAL = 4    # px for horizontal white line between rows

###############################################################################
# Utility: read frames from a GIF or fallback black frames if missing/empty
###############################################################################

# -------------- UTILS --------------
def get_emnist_class_mapping():
    """
    Returns a dict {0:'0', 1:'1', ..., 9:'9', 10:'A', ..., 61:'z'}
    """
    chars = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    return {i: ch for i, ch in enumerate(chars)}

def read_gif_frames(gif_path: str):
    """
    Reads frames from the specified GIF into a list of np.array frames (RGB).
    If 'gif_path' is not found or is empty, returns [10×10 black].
    """
    if not os.path.exists(gif_path):
        print(f"[WARN] Missing sub-GIF: {gif_path}, fallback black frame.")
        return [np.zeros((10,10,3), dtype=np.uint8)]
    frames = []
    reader = imageio.get_reader(gif_path)
    for frame in reader:
        # Some GIFs can be paletted, ensure we convert to RGB np.uint8
        arr = np.array(frame)
        if arr.ndim == 2:
            # grayscale => make it RGB
            arr = np.stack([arr]*3, axis=2)
        elif arr.shape[2] == 4:
            # RGBA => drop alpha
            arr = arr[...,:3]
        frames.append(arr)
    reader.close()
    if not frames:
        print(f"[WARN] Empty sub-GIF: {gif_path}, fallback black frame.")
        return [np.zeros((10,10,3), dtype=np.uint8)]
    return frames

###############################################################################
# Utility: place two frames side by side, 2 px white vertical separator
###############################################################################
def side_by_side_with_white_bar(frame_left: np.ndarray,
                                frame_right: np.ndarray,
                                sep: int = SEPARATOR_VERTICAL) -> np.ndarray:
    """
    Places 'frame_left' and 'frame_right' side by side with a vertical white bar.
    The new image has shape [max(H_left,H_right), W_left+sep+W_right, 3].
    The bar is 'sep' px wide, fully white.
    """
    pil_left = Image.fromarray(frame_left)
    pil_right= Image.fromarray(frame_right)
    Wl, Hl = pil_left.size
    Wr, Hr = pil_right.size
    newW = Wl + sep + Wr
    newH = max(Hl, Hr)

    combined = Image.new("RGB", (newW, newH), color=(0,0,0))
    # paste left
    combined.paste(pil_left, (0,0))
    # vertical white bar
    for y in range(newH):
        for x in range(Wl, Wl+sep):
            combined.putpixel((x,y), (255,255,255))
    # paste right
    combined.paste(pil_right, (Wl+sep,0))

    return np.array(combined)

###############################################################################
# Utility: stack multiple frames vertically with a 4 px horizontal white bar
###############################################################################
def stack_vertical_with_white_bar(frames_row: list[np.ndarray],
                                  sep: int = SEPARATOR_HORIZONTAL) -> np.ndarray:
    """
    Stacks frames_row top→bottom, each separated by 'sep' px of white horizontally.
    The final shape is [sum_of_heights + some bars, max_width, 3].
    """
    # gather sizes
    widths = [f.shape[1] for f in frames_row]
    maxW = max(widths)
    heights= [f.shape[0] for f in frames_row]
    totalH= sum(heights) + (len(frames_row)-1)*sep

    # create black
    combined = Image.new("RGB",(maxW,totalH),(0,0,0))
    y_off= 0
    for i, fr in enumerate(frames_row):
        pil_fr = Image.fromarray(fr)
        combined.paste(pil_fr, (0, y_off))
        y_off += fr.shape[0]
        if i < len(frames_row)-1:
            # 4 px white line
            for xx in range(maxW):
                for yy in range(y_off, y_off+sep):
                    combined.putpixel((xx,yy),(255,255,255))
            y_off+= sep
    return np.array(combined)

###############################################################################
# The main: combine 8 sub-GIFs (4 rows × 2 columns) into one final GIF
###############################################################################
def combine_8_subgifs_to_4x2_gif(subgif_paths: list[str],
                                 output_gif: str,
                                 fps: int = FPS):
    """
    subgif_paths must have exactly 8 paths in the order:
      0 => GradCAM L2 base
      1 => FeatureMap L2 base
      2 => GradCAM None base
      3 => FeatureMap None base
      4 => GradCAM L2 overlay
      5 => FeatureMap L2 overlay
      6 => GradCAM None overlay
      7 => FeatureMap None overlay

    We'll produce one big GIF with 4 rows, each row is 2 sub-GIF side by side,
    and between consecutive rows there's a 4 px horizontal white bar.
    The final frames are determined by the max # of frames among the 8 sub-GIFs.
    For subgif i, if we exceed its #frames, we clamp to last frame.
    We'll write the final GIF at 'fps' frames/sec => ~(#frames / fps) seconds total.
    """
    # If it already exists, skip
    if os.path.exists(output_gif):
        print(f"Skipping final combine because exists: {output_gif}")
        return

    # read all sub-gifs frames
    subgifs_frames = []
    lengths= []
    for path in subgif_paths:
        frames_list = read_gif_frames(path)
        subgifs_frames.append(frames_list)
        lengths.append(len(frames_list))
    max_frames = max(lengths)

    writer = imageio.get_writer(output_gif, mode='I', fps=fps)
    print(f"[INFO] Creating combined 4×2 => {output_gif}")

    # For each global frame i
    for i in range(max_frames):
        # row0 => side_by_side( subgif0[i], subgif1[i] )
        # row1 => side_by_side( subgif2[i], subgif3[i] )
        # row2 => side_by_side( subgif4[i], subgif5[i] )
        # row3 => side_by_side( subgif6[i], subgif7[i] )

        row_imgs = []
        for row_idx in range(4):
            left_idx = row_idx*2
            right_idx= row_idx*2 + 1
            left_frames = subgifs_frames[left_idx]
            right_frames= subgifs_frames[right_idx]
            # clamp i to last index if out of range
            iL= i if i < len(left_frames) else len(left_frames)-1
            iR= i if i < len(right_frames) else len(right_frames)-1
            row_img = side_by_side_with_white_bar(left_frames[iL], right_frames[iR])
            row_imgs.append(row_img)

        # stack row0..row3 vertically
        final_frame = stack_vertical_with_white_bar(row_imgs)
        writer.append_data(final_frame)
    writer.close()
    print(f"[DONE] {output_gif}")

###############################################################################
# MAIN
###############################################################################
def main():
    # 1) The EMNIST classes => 62 total
    # We'll produce one final 4×2 GIF per class
    # Sub-GIF paths must be in:
    #   results/mean/upsampled/<gradcam|featuremap>/<L2|None>/<base|overlay>/Combined/class_{cls}_{ch}.gif
    # in the order:
    #   0 => gradcam/L2/base  => Combined
    #   1 => featuremap/L2/base => Combined
    #   2 => gradcam/None/base => Combined
    #   3 => featuremap/None/base => Combined
    #   4 => gradcam/L2/overlay => Combined
    #   5 => featuremap/L2/overlay => Combined
    #   6 => gradcam/None/overlay => Combined
    #   7 => featuremap/None/overlay => Combined

    mapping = get_emnist_class_mapping()
    num_classes= 62

    # final output folder
    os.makedirs(TOTAL_GIF_DIR, exist_ok=True)

    def path_subgif(tipo, norm, mode, cls, ch):
        # example:
        #   results/mean/upsampled/gradcam/L2/base/Combined/class_0_0.gif
        return os.path.join(
            "results","mean","upsampled", tipo,
            norm, mode, "Combined",
            f"class_{cls}_{ch}.gif"
        )

    for cls in range(num_classes):
        ch = mapping[cls]
        out_gif = os.path.join(TOTAL_GIF_DIR, f"class_{cls}_{ch}.gif")
        # subgif order
        subgifs = [
            path_subgif("gradcam","L2","base",    cls, ch),
            path_subgif("featuremap","L2","base", cls, ch),
            path_subgif("gradcam","None","base",  cls, ch),
            path_subgif("featuremap","None","base", cls, ch),
            path_subgif("gradcam","L2","overlay",    cls, ch),
            path_subgif("featuremap","L2","overlay", cls, ch),
            path_subgif("gradcam","None","overlay",  cls, ch),
            path_subgif("featuremap","None","overlay", cls, ch),
        ]
        combine_8_subgifs_to_4x2_gif(
            subgif_paths=subgifs,
            output_gif=out_gif,
            fps=FPS
        )

    print("All final 4x2 combined GIFs created in:", TOTAL_GIF_DIR)

if __name__ == "__main__":
    main()
