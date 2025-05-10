import os
import re
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor

ALPHA_OVERLAY = 0.3

###################################################
# 1. Funzioni di utilità
###################################################
def get_epoch_from_folder(folder_name):
    """
    Cerca 'epoch_<num>' nel nome della cartella e restituisce il numero.
    """
    m = re.search(r"epoch_(\d+)", folder_name)
    return int(m.group(1)) if m else None

def get_sorted_epoch_folders(input_folder):
    """
    Ritorna l'elenco di sottocartelle 'epoch_xxx' ordinate per x.
    Se la cartella non esiste, ritorna [].
    """
    if not os.path.exists(input_folder):
        print(f"[WARN] Cartella non trovata: {input_folder}")
        return []
    subdirs = [
        f for f in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, f)) and f.startswith("epoch_")
    ]
    subdirs.sort(key=lambda x: get_epoch_from_folder(x))
    return subdirs

def compute_durations(num_images, total_duration=10.0, max_duration=1.0, min_duration=0.1):
    """
    Genera un vettore di durate in secondi per i frame, di lunghezza num_images,
    in modo che la somma delle durate sia total_duration (es. 10 secondi).
    Viene applicato un andamento "quadratico" per variare le durate.
    """
    if num_images < 1:
        return []
    durations = max_duration - (np.linspace(0, 1, num_images)**2) * (max_duration - min_duration)
    scale = total_duration / np.sum(durations)
    return durations * scale

def get_emnist_class_mapping():
    """Mappa 0..61 → '0'..'9','A'..'Z','a'..'z'."""
    chars = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    return {i: ch for i, ch in enumerate(chars)}

def draw_text_with_border(draw, position, text, font,
                          fill=(255, 255, 255), border_fill="black", border_width=1):
    """
    Disegna testo in colore bianco con un piccolo contorno nero.
    """
    x, y = position
    for dx in [-border_width, 0, border_width]:
        for dy in [-border_width, 0, border_width]:
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=border_fill)
    draw.text((x, y), text, font=font, fill=fill)

def get_mean_image(class_id):
    """
    Carica l'immagine media di E-MNIST per la classe data, se esiste.
    Ritorna None se mancante, in modo da saltare l'overlay.
    """
    path = os.path.join("results", "E-MNIST-PROCESSING", "Total", "Mean",
                        f"EMNIST_total_mean_class_{class_id}.png")
    if not os.path.exists(path):
        print(f"[WARN] Mean image assente: {path}")
        return None
    try:
        pil_img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Errore nell'apertura dell'immagine media {path}: {e}")
        return None
    return np.array(pil_img).astype(np.float32) / 255.0

def overlay_mean_on_frame(frame, mean_img):
    """
    Sovrappone 'frame' su mean_img con trasparenza ALPHA_OVERLAY.
    Riduce l'intensità del canale blu (x0.1).
    """
    if frame.shape[2] == 4:
        frame = frame[..., :3]
    heatmap = frame.astype(np.float32) / 255.0
    h, w = heatmap.shape[:2]

    # Ridimensiona mean_img a (w, h)
    base = np.array(Image.fromarray((mean_img * 255).astype(np.uint8)).resize((w, h))).astype(np.float32) / 255.0

    # Sopprime blu
    heatmap[:, :, 2] *= 0.1
    # Usa il canale rosso come intensità
    intensity = heatmap[:, :, 0]
    alpha_mask = np.where(intensity < 0.3, 0.0, ALPHA_OVERLAY)[..., np.newaxis]

    out = (1 - alpha_mask) * base + alpha_mask * heatmap
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)

###################################################
# 2. Creazione delle GIF singole
###################################################
def create_gif_from_images(image_paths, durations, output_path,
                           overlay_texts=None, net_text="", norm_text="",
                           category_text=None):
    """
    Crea la GIF base (durata totale 10 secondi).
    Se il file esiste già, salta la creazione.
    """
    if os.path.exists(output_path):
        print(f"Salto perché esiste già: {output_path}")
        return

    writer = imageio.get_writer(output_path, mode='I')
    font = ImageFont.load_default()
    for i, img_path in enumerate(image_paths):
        try:
            frame = imageio.imread(img_path)
        except Exception as e:
            print(f"[WARN] Errore nella lettura dell'immagine {img_path}: {e}")
            continue

        # Ruota e ribalta l'immagine se necessario
        pil_img = Image.fromarray(frame).transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
        draw = ImageDraw.Draw(pil_img)

        ep_text = overlay_texts[i] if (overlay_texts and i < len(overlay_texts)) else ""
        # In alto a sinistra
        if ep_text:
            draw_text_with_border(draw, (10, 10), ep_text, font)
            # In alto a destra se category_text è definito
            if category_text:
                rt = f"{category_text} - {ep_text}"
                bbox = draw.textbbox((0, 0), rt, font=font)
                tw = bbox[2] - bbox[0]
                draw_text_with_border(draw, (pil_img.width - tw - 10, 10), rt, font)

        # In alto al centro: net_text e norm_text
        if net_text and norm_text:
            center_str = f"{norm_text} - {net_text}"
            bc = draw.textbbox((0, 0), center_str, font=font)
            cw = bc[2] - bc[0]
            cx = (pil_img.width - cw) // 2
            draw_text_with_border(draw, (cx, 10), center_str, font)

        writer.append_data(np.array(pil_img), {'duration': durations[i]})
    writer.close()
    print(f"GIF base salvata => {output_path}")

def generate_overlay_version(gif_path, class_id, out_overlay,
                             overlay_texts=None, net_text="", norm_text="",
                             category_text=None):
    """
    Dalla GIF base genera la versione overlay (aggiungendo l'immagine media)
    se non esiste, salvando il file con suffisso '_sovrapposta'.
    """
    base_name = os.path.basename(gif_path)
    root, ext = os.path.splitext(base_name)
    out_name = f"{root}_sovrapposta{ext}"
    out_path = os.path.join(out_overlay, out_name)

    if os.path.exists(out_path):
        print(f"Salto overlay perché esiste già: {out_path}")
        return

    mean_img = get_mean_image(class_id)
    if mean_img is None:
        print("[WARN] Niente mean_img => skip overlay.")
        return

    os.makedirs(out_overlay, exist_ok=True)
    frames_in = []
    try:
        reader = imageio.get_reader(gif_path)
        for fr in reader:
            frames_in.append(np.array(fr))
        reader.close()
    except Exception as e:
        print(f"[WARN] Errore nella lettura della GIF base {gif_path}: {e}")
        return

    if not frames_in:
        print(f"[WARN] GIF base vuota: {gif_path}")
        return

    durations = compute_durations(len(frames_in), total_duration=10.0)
    if overlay_texts is None:
        overlay_texts = ["" for _ in frames_in]

    writer = imageio.get_writer(out_path, mode='I')
    font = ImageFont.load_default()
    for i, fr in enumerate(frames_in):
        over = overlay_mean_on_frame(fr, mean_img)
        pil_img = Image.fromarray(over)
        draw = ImageDraw.Draw(pil_img)
        ep_text = overlay_texts[i]
        if ep_text:
            draw_text_with_border(draw, (10, 10), ep_text, font)
            if category_text:
                rt = f"{category_text} - {ep_text}"
                bb = draw.textbbox((0, 0), rt, font=font)
                tw = bb[2] - bb[0]
                draw_text_with_border(draw, (pil_img.width - tw - 10, 10), rt, font)
        if net_text and norm_text:
            cs = f"{norm_text} - {net_text}"
            bc = draw.textbbox((0, 0), cs, font=font)
            cw = bc[2] - bc[0]
            cx = (pil_img.width - cw) // 2
            draw_text_with_border(draw, (cx, 10), cs, font)
        writer.append_data(np.array(pil_img), {'duration': durations[i]})
    writer.close()
    print(f"Overlay GIF salvata => {out_path}")

def generate_individual_gif_for_class(model_type, norm, class_id,
                                      base_input_folder, output_folder,
                                      mapping, image_type="gradcam"):
    """
    Crea la GIF base per la classe (se non esiste) e ritorna (path, epoche).
    Il nome della GIF base è: class_<class_id>_<letter>.gif
    mentre la relativa overlay sarà: class_<class_id>_<letter>_sovrapposta.gif.
    """
    if model_type == "Standard":
        net_text = "LeNet5"
        prefix = "Standard_LeNet5_train"
    else:
        net_text = "KaNet5"
        prefix = "KaNet5_test"

    norm_text = "No-Norm" if norm == "None" else "L2"
    if image_type == "gradcam":
        cat_text = "GRADCAM"
        suffix = "_mean_gradcam_class_"
    else:
        cat_text = "FEATUREMAP"
        suffix = "_mean_featuremap_class_"
    suffix += str(class_id) + ".png"

    # Raccoglie i PNG (uno per epoch)
    ep_folders = get_sorted_epoch_folders(base_input_folder)
    paths = []
    ep_nums = []
    for f in ep_folders:
        ep = get_epoch_from_folder(f)
        if ep is None:
            continue
        path_png = os.path.join(base_input_folder, f, "Upsampled", "Mean", prefix + suffix)
        if os.path.exists(path_png):
            ep_nums.append(ep)
            paths.append(path_png)
        else:
            print(f"[WARN] Immagine non trovata per la classe {class_id} nell'epoca {ep}: {path_png}")

    if not paths:
        print(f"[WARN] Nessuna immagine per cls={class_id} in {base_input_folder}")
        return None, None

    durations = compute_durations(len(paths), total_duration=10.0)
    overlay_texts = [f"Epoch: {e}" for e in ep_nums]

    letter = mapping.get(class_id, "NA")
    out_name = f"class_{class_id}_{letter}.gif"
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, out_name)
    if os.path.exists(out_path):
        print(f"Esiste {out_path}, skip singola base.")
        return out_path, ep_nums

    create_gif_from_images(paths, durations, out_path,
                           overlay_texts=overlay_texts,
                           net_text=net_text,
                           norm_text=norm_text,
                           category_text=cat_text)
    return out_path, ep_nums

###################################################
# 3. GIF COMBINATE (Standard + Kan)
###################################################
def combine_two_gifs_side_by_side(std_gif, kan_gif, out_path, epoch_numbers, class_id=None):
    """
    Giustappone orizzontalmente i frame di std_gif (a sinistra) e kan_gif (a destra),
    separandoli con un separatore verticale di 2px bianco.
    Se out_path esiste già, salta.
    """
    if os.path.exists(out_path):
        print(f"Esiste {out_path}, skip combined.")
        return
    try:
        std_rd = imageio.get_reader(std_gif)
        kan_rd = imageio.get_reader(kan_gif)
    except Exception as e:
        print(f"[WARN] Errore nell'apertura delle GIF per la combinazione: {e}")
        return

    fr_std = list(std_rd)
    fr_kan = list(kan_rd)
    std_rd.close()
    kan_rd.close()

    n_frames = max(len(fr_std), len(fr_kan))
    durations = compute_durations(n_frames, total_duration=10.0)

    writer = imageio.get_writer(out_path, mode='I')
    sep_width = 2

    for i in range(n_frames):
        i_s = i if i < len(fr_std) else len(fr_std) - 1
        i_k = i if i < len(fr_kan) else len(fr_kan) - 1
        left_frame = np.array(fr_std[i_s])
        right_frame = np.array(fr_kan[i_k])

        pil_left = Image.fromarray(left_frame)
        pil_right = Image.fromarray(right_frame)
        Wl, Hl = pil_left.size
        Wr, Hr = pil_right.size

        newW = Wl + sep_width + Wr
        newH = max(Hl, Hr)
        combined = Image.new("RGB", (newW, newH), (0, 0, 0))
        combined.paste(pil_left, (0, 0))

        # Disegna separatore verticale (2px bianco)
        for y in range(newH):
            for x in range(Wl, Wl + sep_width):
                combined.putpixel((x, y), (255, 255, 255))

        combined.paste(pil_right, (Wl + sep_width, 0))
        writer.append_data(np.array(combined), {'duration': durations[i]})
    writer.close()
    print(f"Combined GIF salvata => {out_path}")

###################################################
# 4. Generazione della GIF TOTAL (griglia 2x4)
###################################################
def combine_total_gif_for_class(cls, mapping, base_gradcam, base_featuremap, out_total_dir):
    """
    Per la classe data, combina le GIF "combined" (già generate) in una griglia di 4 righe e 2 colonne:
    
      Row0: [GradCAM_composta_L2_base]  –  [Featuremap_composta_L2_base]
      Row1: [GradCAM_composta_None_base]  –  [Featuremap_composta_None_base]
      Row2: [GradCAM_composta_L2_overlay] –  [Featuremap_composta_L2_overlay]
      Row3: [GradCAM_composta_None_overlay] – [Featuremap_composta_None_overlay]
      
    Le colonne sono separate da un separatore verticale di 2px bianco,
    le righe da un separatore orizzontale di 4px bianco.
    """
    letter = mapping.get(cls, "NA")
    # I file base hanno nome "class_<cls>_<letter>.gif"
    filename_base = f"class_{cls}_{letter}.gif"
    # I file overlay hanno nome "class_{cls}_{letter}_sovrapposta.gif"
    filename_overlay = f"class_{cls}_{letter}_sovrapposta.gif"
    
    paths = {
        0: {  # Row 0: L2 base
            "left": os.path.join(base_gradcam, "L2", "base", "Combined", filename_base),
            "right": os.path.join(base_featuremap, "L2", "base", "Combined", filename_base)
        },
        1: {  # Row 1: None base
            "left": os.path.join(base_gradcam, "None", "base", "Combined", filename_base),
            "right": os.path.join(base_featuremap, "None", "base", "Combined", filename_base)
        },
        2: {  # Row 2: L2 overlay
            "left": os.path.join(base_gradcam, "L2", "overlay", "Combined", filename_overlay),
            "right": os.path.join(base_featuremap, "L2", "overlay", "Combined", filename_overlay)
        },
        3: {  # Row 3: None overlay
            "left": os.path.join(base_gradcam, "None", "overlay", "Combined", filename_overlay),
            "right": os.path.join(base_featuremap, "None", "overlay", "Combined", filename_overlay)
        }
    }
    
    # Verifica che tutti i file esistano
    for r in paths:
        for pos in ["left", "right"]:
            if not os.path.exists(paths[r][pos]):
                print(f"[WARN] File mancante per total GIF, classe {cls}: {paths[r][pos]}")
                return

    # Carica i frame per ogni cella
    cell_frames = {}  # {r: {"left": [...], "right": [...]} }
    cell_sizes = {}   # {r: {"left": (w,h), "right": (w,h)} }
    max_frames_per_row = {}
    for r in paths:
        cell_frames[r] = {}
        cell_sizes[r] = {}
        for pos in ["left", "right"]:
            try:
                reader = imageio.get_reader(paths[r][pos])
                frames = [np.array(fr) for fr in reader]
                reader.close()
            except Exception as e:
                print(f"[WARN] Errore nella lettura di {paths[r][pos]}: {e}")
                return
            if not frames:
                print(f"[WARN] GIF vuota: {paths[r][pos]}")
                return
            cell_frames[r][pos] = frames
            # Prendi la dimensione del primo frame
            im = Image.fromarray(frames[0])
            cell_sizes[r][pos] = im.size
        max_frames_per_row[r] = max(len(cell_frames[r]["left"]), len(cell_frames[r]["right"]))
    total_frames = max(max_frames_per_row.values())

    # Determina la larghezza massima della colonna sinistra e destra (su tutte le righe)
    max_left_width = max(cell_sizes[r]["left"][0] for r in cell_sizes)
    max_right_width = max(cell_sizes[r]["right"][0] for r in cell_sizes)

    total_frames_list = []
    durations = compute_durations(total_frames, total_duration=10.0)
    for i in range(total_frames):
        row_images = []
        row_widths = []
        for r in range(4):
            # Seleziona il frame (se i supera il numero disponibile, prendi l'ultimo)
            left_frames = cell_frames[r]["left"]
            right_frames = cell_frames[r]["right"]
            left_idx = i if i < len(left_frames) else len(left_frames) - 1
            right_idx = i if i < len(right_frames) else len(right_frames) - 1
            left_img = Image.fromarray(left_frames[left_idx])
            right_img = Image.fromarray(right_frames[right_idx])
            # Pad a sinistra (se necessario)
            if left_img.width < max_left_width:
                new_left = Image.new("RGB", (max_left_width, left_img.height), (0, 0, 0))
                new_left.paste(left_img, (0, 0))
                left_img = new_left
            if right_img.width < max_right_width:
                new_right = Image.new("RGB", (max_right_width, right_img.height), (0, 0, 0))
                new_right.paste(right_img, (0, 0))
                right_img = new_right
            # Altezza della riga = max(altezza sinistra, destra)
            row_height = max(left_img.height, right_img.height)
            if left_img.height < row_height:
                new_left = Image.new("RGB", (left_img.width, row_height), (0, 0, 0))
                new_left.paste(left_img, (0, 0))
                left_img = new_left
            if right_img.height < row_height:
                new_right = Image.new("RGB", (right_img.width, row_height), (0, 0, 0))
                new_right.paste(right_img, (0, 0))
                right_img = new_right
            # Combina sinistra e destra con separatore verticale di 2px bianco
            combined_width = left_img.width + 2 + right_img.width
            combined_img = Image.new("RGB", (combined_width, row_height), (0, 0, 0))
            combined_img.paste(left_img, (0, 0))
            for x in range(left_img.width, left_img.width + 2):
                for y in range(row_height):
                    combined_img.putpixel((x, y), (255, 255, 255))
            combined_img.paste(right_img, (left_img.width + 2, 0))
            row_images.append(combined_img)
            row_widths.append(combined_width)
        # Allinea tutte le righe alla larghezza massima
        final_row_width = max(row_widths)
        for idx, img in enumerate(row_images):
            if img.width < final_row_width:
                new_img = Image.new("RGB", (final_row_width, img.height), (0, 0, 0))
                new_img.paste(img, (0, 0))
                row_images[idx] = new_img
        # Combina le righe verticalmente con separatore orizzontale di 4px bianco
        total_height = sum(img.height for img in row_images) + 3 * 4  # 3 separatori fra 4 righe
        total_img = Image.new("RGB", (final_row_width, total_height), (0, 0, 0))
        current_y = 0
        for idx, img in enumerate(row_images):
            total_img.paste(img, (0, current_y))
            current_y += img.height
            if idx < 3:
                # Disegna separatore orizzontale di 4px bianco
                for y in range(current_y, current_y + 4):
                    for x in range(final_row_width):
                        total_img.putpixel((x, y), (255, 255, 255))
                current_y += 4
        total_frames_list.append(np.array(total_img))
    # Scrive la GIF total
    os.makedirs(out_total_dir, exist_ok=True)
    out_total = os.path.join(out_total_dir, f"class_{cls}_{letter}_total.gif")
    writer = imageio.get_writer(out_total, mode='I')
    for i, frame in enumerate(total_frames_list):
        writer.append_data(frame, {'duration': durations[i]})
    writer.close()
    print(f"Total GIF salvata => {out_total}")

###################################################
# 5. Parallel Helper
###################################################
def process_individual_class(args):
    """
    Esegue generate_individual_gif_for_class per la GIF base
    e poi crea la versione overlay.
    """
    (tipo, norm, model, cls,
     in_folder, out_base, out_overlay,
     mapping, img_type) = args

    base_gif, ep_nums = generate_individual_gif_for_class(
        model, norm, cls, in_folder, out_base,
        mapping, image_type=img_type
    )
    if base_gif and ep_nums:
        overlay_texts = [f"Epoch: {ep}" for ep in ep_nums]
        net_text = "LeNet5" if model == "Standard" else "KaNet5"
        n_text = "No-Norm" if norm == "None" else "L2"
        cat_text = "GRADCAM" if img_type == "gradcam" else "FEATUREMAP"
        generate_overlay_version(base_gif, cls, out_overlay,
                                 overlay_texts=overlay_texts,
                                 net_text=net_text, norm_text=n_text,
                                 category_text=cat_text)
    return (cls, base_gif)

###################################################
# MAIN
###################################################
def main_generate_gifs():
    """
    1) Genera le GIF singole (base e overlay) per gradcam e featuremap,
       per entrambe le modalità di normalizzazione (L2, None) e modelli (Standard, Kan).
    2) Combina le GIF (Standard vs Kan) per le versioni base e overlay.
    3) Genera le GIF TOTAL, composte da 4 righe (2 per base e 2 per overlay) disposte in griglia.
       Tutte le GIF avranno una durata totale di 10 secondi.
    """
    mapping = get_emnist_class_mapping()
    norms = ["L2", "None"]
    models = ["Standard", "Kan"]
    num_classes = 62

    base_paths_gradcam = {
        ("L2", "Standard"): "results/results_L2_SGD_lr0.01_0_0/Standard_LeNet5/GradCAM",
        ("L2", "Kan"):      "results/results_L2_SGD_lr0.01_5_3/KaNet5/GradCAM",
        ("None", "Standard"): "results/results_None_SGD_lr0.01_0_0/Standard_LeNet5/GradCAM",
        ("None", "Kan"):      "results/results_None_SGD_lr0.01_5_3/KaNet5/GradCAM"
    }
    base_paths_featuremap = {
        ("L2", "Standard"): "results/results_L2_SGD_lr0.01_0_0/Standard_LeNet5/FeatureMap",
        ("L2", "Kan"):      "results/results_L2_SGD_lr0.01_5_3/KaNet5/FeatureMap",
        ("None", "Standard"): "results/results_None_SGD_lr0.01_0_0/Standard_LeNet5/FeatureMap",
        ("None", "Kan"):      "results/results_None_SGD_lr0.01_5_3/KaNet5/FeatureMap"
    }

    base_output_gradcam = "results/mean/upsampled/gradcam"
    base_output_featuremap = "results/mean/upsampled/featuremap"

    # --- 1) Generazione in parallelo delle GIF singole ---
    def process_individual(tipo, norm, model):
        if tipo == "gradcam":
            in_folder = base_paths_gradcam[(norm, model)]
            out_dir = base_output_gradcam
            img_type = "gradcam"
        else:
            in_folder = base_paths_featuremap[(norm, model)]
            out_dir = base_output_featuremap
            img_type = "featuremap"

        out_base = os.path.join(out_dir, norm, "base", model)
        out_overlay = os.path.join(out_dir, norm, "overlay", model)
        os.makedirs(out_base, exist_ok=True)
        os.makedirs(out_overlay, exist_ok=True)

        print(f"\n[{tipo.upper()}] Norm={norm}, Model={model}, folder={in_folder}")

        args_list = []
        for cls in range(num_classes):
            args_list.append((tipo, norm, model, cls, in_folder, out_base, out_overlay, mapping, img_type))

        with ProcessPoolExecutor() as executor:
            for c, base_gif in executor.map(process_individual_class, args_list):
                if base_gif:
                    print(f"  {tipo.upper()} - classe {c} ({mapping[c]}) => {base_gif}")

    # --- 2) Combina: Standard vs Kan ---
    def process_combined(tipo, norm):
        if tipo == "gradcam":
            out_dir = base_output_gradcam
            in_paths = base_paths_gradcam
        else:
            out_dir = base_output_featuremap
            in_paths = base_paths_featuremap

        # Directory per le GIF base
        std_dir_base = os.path.join(out_dir, norm, "base", "Standard")
        kan_dir_base = os.path.join(out_dir, norm, "base", "Kan")
        comb_base = os.path.join(out_dir, norm, "base", "Combined")
        os.makedirs(comb_base, exist_ok=True)

        # Directory per le GIF overlay
        std_dir_ov = os.path.join(out_dir, norm, "overlay", "Standard")
        kan_dir_ov = os.path.join(out_dir, norm, "overlay", "Kan")
        comb_ov = os.path.join(out_dir, norm, "overlay", "Combined")
        os.makedirs(comb_ov, exist_ok=True)

        # Ottiene i numeri degli epoch (basandosi sui dati di Kan)
        base_in_kan = in_paths[(norm, "Kan")]
        ep_folders = get_sorted_epoch_folders(base_in_kan)
        ep_nums = [get_epoch_from_folder(f) for f in ep_folders]

        for cls in range(num_classes):
            ch = mapping[cls]
            # GIF base
            bname = f"class_{cls}_{ch}.gif"
            std_gif = os.path.join(std_dir_base, bname)
            kan_gif = os.path.join(kan_dir_base, bname)
            comb_gif = os.path.join(comb_base, bname)
            if os.path.exists(std_gif) and os.path.exists(kan_gif):
                combine_two_gifs_side_by_side(
                    std_gif, kan_gif, comb_gif, ep_nums, class_id=cls
                )
            else:
                print(f"[WARN] Per la combinazione base, mancano le GIF per la classe {cls} ({bname}).")

            # GIF overlay
            oname = f"class_{cls}_{ch}_sovrapposta.gif"
            std_ovgif = os.path.join(std_dir_ov, oname)
            kan_ovgif = os.path.join(kan_dir_ov, oname)
            comb_ovgif = os.path.join(comb_ov, oname)
            if os.path.exists(std_ovgif) and os.path.exists(kan_ovgif):
                combine_two_gifs_side_by_side(
                    std_ovgif, kan_ovgif, comb_ovgif, ep_nums, class_id=cls
                )
            else:
                print(f"[WARN] Per la combinazione overlay, mancano le GIF per la classe {cls} ({oname}).")

    # --- 3) Genera le GIF TOTAL (griglia 2x4) ---
    def process_total():
        # Definiamo la directory per le GIF total
        out_total = os.path.join("results", "mean", "upsampled", "total")
        os.makedirs(out_total, exist_ok=True)
        for cls in range(num_classes):
            combine_total_gif_for_class(cls, mapping, base_output_gradcam, base_output_featuremap, out_total)

    # Esecuzione:
    # a) Genera le GIF singole
    for norm in ["L2", "None"]:
        for model in ["Standard", "Kan"]:
            process_individual("gradcam", norm, model)
    for norm in ["L2", "None"]:
        for model in ["Standard", "Kan"]:
            process_individual("featuremap", norm, model)

    # b) Combina le GIF (Standard vs Kan)
    for norm in ["L2", "None"]:
        process_combined("gradcam", norm)
        process_combined("featuremap", norm)

    # c) Genera le GIF TOTAL
    process_total()

if __name__ == "__main__":
    main_generate_gifs()
