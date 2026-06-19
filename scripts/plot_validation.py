#!/usr/bin/env python3
"""
plot_validation_fixed.py
Fully compatible with your CIFAR-10 CSV logs:
epoch, train_acc, test_acc, mi_proxy, align_mi, curv_proxy, idim_proxy, eff_ratio

Generates:
  - figs/cifar_validation_combined.png
  - tables/cifar_summary_full.csv + .tex
Usage:
  python plot_validation_fixed.py --logdir logs --outdir figs --tablesdir tables
"""
import os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from matplotlib import gridspec

# ---------- Utilities ----------
def infer_fraction(fname):
    fname = fname.lower()
    for k,v in {'20':0.2,'40':0.4,'60':0.6,'80':0.8,'100':1.0}.items():
        if f'frac{k}' in fname or f'_{k}_' in fname:
            return v
    return None

def load_csv(p):
    df = pd.read_csv(p)
    # normalize names
    df.columns = [c.strip().lower() for c in df.columns]
    # sanity check
    required = ['epoch','test_acc','align_mi','eff_ratio']
    for r in required:
        if r not in df.columns:
            raise ValueError(f"{p} missing required column: {r}")
    return df

def compute_metrics(df, last_window=10, sat_delta=0.002):
    """Compute summary metrics per CSV"""
    df = df.sort_values('epoch')
    final = df.iloc[-1]
    res = {
        'final_acc': final['test_acc'],
        'final_align': final['align_mi'],
        'mean_eff': df['eff_ratio'].mean(),
        'max_eff': df['eff_ratio'].max(),
        'corr_acc_align': float(np.corrcoef(df['test_acc'], df['align_mi'])[0,1]),
    }
    # detect saturation (rolling mean stable)
    roll = df['test_acc'].rolling(last_window, min_periods=1).mean()
    diffs = np.abs(np.diff(roll))
    sat_epoch = df['epoch'].iloc[-1]
    for i in range(len(diffs)-last_window):
        if np.all(diffs[i:i+last_window] < sat_delta):
            sat_epoch = df['epoch'].iloc[i+last_window]
            break
    res['saturation_epoch'] = sat_epoch
    return res

# ---------- Main pipeline ----------
def main(args):
    files = glob(os.path.join(args.logdir, "*.csv"))
    if not files:
        print("No CSV files found in", args.logdir)
        return

    summary = {}
    for f in files:
        frac = infer_fraction(f)
        if frac is None:
            print("Skipping (no fraction inferred):", f)
            continue
        df = load_csv(f)
        m = compute_metrics(df)
        summary.setdefault(frac, []).append((f, m, df))

    # aggregate across possible multiple runs
    agg = {}
    for frac, items in summary.items():
        ms = [x[1] for x in items]
        agg[frac] = {k: np.mean([m[k] for m in ms]) for k in ms[0].keys()}
        for k in ms[0].keys():
            agg[frac][k+'_std'] = np.std([m[k] for m in ms])

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.tablesdir, exist_ok=True)

    # ---------- Save LaTeX table ----------
    tex_path = os.path.join(args.tablesdir, "cifar_summary_full.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{CIFAR-10 summary across training fractions. Mean (std) across seeds.}\n")
        f.write("\\label{tab:cifar_summary_full}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Frac & Final Acc & Align MI & Mean Eff & Max Eff & Corr(acc,align) & Sat. Epoch\\\\\n\\midrule\n")
        for frac in sorted(agg.keys()):
            a = agg[frac]
            f.write(f"{frac:.2f} & {a['final_acc']:.3f} & {a['final_align']:.4f} & "
                    f"{a['mean_eff']:.2f} & {a['max_eff']:.2f} & "
                    f"{a['corr_acc_align']:.3f} & {int(a['saturation_epoch'])}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"[✓] Wrote table → {tex_path}")

    # ---------- Plot combined ----------
    fig = plt.figure(figsize=(14,10))
    gs = gridspec.GridSpec(3, 2)
    fracs = sorted(agg.keys())
    for i, frac in enumerate(fracs):
        _, _, df = summary[frac][0]
        ax = fig.add_subplot(gs[i//2, i%2])
        ax.plot(df['epoch'], df['test_acc'], 'b-', label='acc')
        ax.set_ylabel('Accuracy', color='b')
        ax.set_xlabel('Epoch')
        ax2 = ax.twinx()
        ax2.plot(df['epoch'], df['align_mi'], 'r--', label='alignMI')
        ax2.set_ylabel('alignMI', color='r')
        ax.set_title(f"Fraction={frac:.2f}")
    # bottom row: efficiency and correlation
    axb = fig.add_subplot(gs[2,0])
    mean_eff = [agg[f]['mean_eff'] for f in fracs]
    max_eff = [agg[f]['max_eff'] for f in fracs]
    axb.bar(np.arange(len(fracs))-0.2, mean_eff, 0.4, label='mean_eff')
    axb.bar(np.arange(len(fracs))+0.2, max_eff, 0.4, label='max_eff')
    axb.set_xticks(np.arange(len(fracs)))
    axb.set_xticklabels([f"{f:.2f}" for f in fracs])
    axb.set_ylabel("Efficiency")
    axb.legend()

    axc = fig.add_subplot(gs[2,1])
    axc.plot(fracs, [agg[f]['corr_acc_align'] for f in fracs], marker='o')
    axc.set_xlabel("Fraction")
    axc.set_ylabel("Corr(acc,align)")
    axc.set_title("Coupling across data fractions")
    plt.tight_layout()
    outfig = os.path.join(args.outdir, "cifar_validation_combined.png")
    plt.savefig(outfig, dpi=300)
    print(f"[✓] Saved figure → {outfig}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="logs")
    p.add_argument("--outdir", default="figs")
    p.add_argument("--tablesdir", default="tables")
    args = p.parse_args()
    main(args)