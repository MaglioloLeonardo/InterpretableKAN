#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd

def format_number(x):
    return f"{x:.2f}"

def analyze_csv(csv_file, model_a, model_b, p_eps=0.0):
    """
    Estrae:
      - Delta@50
      - Comparison@50
      - Delta_range_epochs (escludendo righe con p_value <= p_eps)
      - Significance (in base a p_value Total)
      - Has_delta_changed_sign
    
    p_eps: soglia minima di p_value per includere la riga nel calcolo del range
    """
    df = pd.read_csv(csv_file)
    # converto epoch a numerico e filtro
    df['epoch_num'] = pd.to_numeric(df['epoch'], errors='coerce')
    df_num = df.dropna(subset=['epoch_num']).copy()
    # converto p_value a float
    df_num['p_value'] = pd.to_numeric(df_num['p_value'], errors='coerce')
    
    # 1) Delta@50
    row50 = df_num[df_num['epoch_num'] == 50]
    if row50.empty:
        raise RuntimeError(f"Epoch 50 non trovata in {csv_file}")
    d50 = float(row50.iloc[0]['delta'])
    
    # 2) Has delta changed sign?
    pos = (df_num['delta'] > 0).any()
    neg = (df_num['delta'] < 0).any()
    has_changed = pos and neg
    
    # 3) Delta_range_epochs, escludendo p_value <= p_eps
    df_for_range = df_num[df_num['p_value'] > p_eps]
    if df_for_range.empty:
        # se dopo il filtro non resta nulla, ripiego su tutte le epoche
        df_for_range = df_num
    drange = float(df_for_range['delta'].max() - df_for_range['delta'].min())
    
    # 4) Confronto a 50
    if d50 > 0:
        comparison = f"{model_a} > {model_b} by {format_number(d50)}%"
    elif d50 < 0:
        comparison = f"{model_b} > {model_a} by {format_number(abs(d50))}%"
    else:
        comparison = "equal"
    
    # 5) Significance sul totale
    total = df[df['epoch'] == 'Total']
    if total.empty:
        raise RuntimeError(f"Riga Total non trovata in {csv_file}")
    p_tot = float(total.iloc[0]['p_value'])
    if 0.01 <= p_tot <= 0.05:
        significance = "0.01<=p<=0.05"
    else:
        significance = "p<0.01 or p>0.05"
    
    return {
        'Delta@50': format_number(d50),
        'Comparison@50': comparison,
        'Delta_range_epochs': format_number(drange),
        'Significance': significance,
        'Has_delta_changed_sign': has_changed
    }

def build_table(root_scores, variant, map_type, model_a, model_b):
    metrics = [
        "heatxedge", "heatxcorner", "heatxloop",
        "heatxorl_mask", "heatxvcr_mask", "segmentation_heat_loop"
    ]
    flags = ["raw", "norm", "abs_ordinal", "zscore"]
    rows = []

    for metric in metrics:
        for norm in flags:
            # provo prima 'greater', poi '='
            found = None
            for sym in ['greater', 'equal']:
                pattern = os.path.join(
                    root_scores,
                    metric,
                    f"results_{variant}_{map_type}_{norm}_{metric}_{model_a}_vs_{model_b}_{'>' if sym=='greater' else '='}.csv"
                )
                matches = glob.glob(pattern)
                if matches:
                    found = matches[0]
                    break
            if not found:
                continue

            try:
                info = analyze_csv(found, model_a, model_b, p_eps=0.0)
            except Exception as e:
                # salto i file che danno errore
                continue

            rows.append({
                'Metric': metric,
                'Normalization type': norm,
                **info
            })

    out_name = f"table_{variant}_{map_type}.csv"
    pd.DataFrame(rows).to_csv(out_name, index=False)
    print(f"Creato: {out_name}")

def main(root_scores, variant, model_a, model_b):
    for map_type in ("featuremap", "gradcam"):
        build_table(root_scores, variant, map_type, model_a, model_b)

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Estrae Delta@50, Comparison, Delta_range_epochs, "
                    "Significance e Has_delta_changed_sign; "
                    "esclude righe con p_value=0 dal range"
    )
    p.add_argument("--root-scores", required=True,
                   help="Cartella base contenente scores/<metric>/")
    p.add_argument("--variant", default="L2",
                   help="Variant, es. None o L2")
    p.add_argument("--model-a", default="Standard_LeNet5",
                   help="Primo modello (es. Standard_LeNet5)")
    p.add_argument("--model-b", default="KaNet5",
                   help="Secondo modello (es. KaNet5)")
    args = p.parse_args()
    main(args.root_scores, args.variant, args.model_a, args.model_b)
