"""
vgib_plots.py

Reads CSV logs produced by vgib_experiments.py and
generates summary tables and core diagnostic plots:

- Swiss-roll: curvature–information Pareto, baseline vs V-GIB
- Fashion-MNIST: accuracy and curvature dynamics
- CIFAR-10: accuracy vs data fraction, efficiency vs fraction, etc.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_logs(base_dir="./logs"):
    swiss = pd.read_csv(os.path.join(base_dir, "swiss", "swiss_roll_results.csv"))
    fashion = pd.read_csv(os.path.join(base_dir, "fashion", "fashion_results.csv"))
    cifar = pd.read_csv(os.path.join(base_dir, "cifar", "cifar10_results.csv"))
    return swiss, fashion, cifar


def swiss_plots(df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    # final epoch per config
    final = df.groupby(
        ["model", "seed", "noise_std", "beta", "gamma", "z_dim"],
        as_index=False
    ).last()

    # Pareto frontier: val_acc vs curv for each model
    plt.figure(figsize=(6, 4))
    for model in ["random", "mlp", "vib", "vgib"]:
        sub = final[final["model"] == model]
        if sub.empty:
            continue
        plt.scatter(sub["curv"], sub["val_acc"], label=model, alpha=0.8)
    plt.xlabel("Curvature proxy")
    plt.ylabel("Validation accuracy")
    plt.title("Swiss-roll: curvature–information trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "swiss_pareto.pdf"), bbox_inches="tight")
    plt.close()

    # Simple correlation summary
    corr = final[["val_acc", "curv", "idim", "mi_proxy"]].corr()
    print("\n[Swiss-roll] correlation matrix:")
    print(corr.round(3))


def fashion_plots(df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    # keep seed=0, model=vgib as main curve
    vgib = df[(df["model"] == "vgib") & (df["seed"] == 0)]
    baseline = df[(df["model"] == "baseline") & (df["seed"] == 0)]

    plt.figure(figsize=(6, 4))
    plt.plot(vgib["epoch"], vgib["val_acc"], label="V-GIB")
    plt.plot(baseline["epoch"], baseline["val_acc"], label="Baseline")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Fashion-MNIST: baseline vs V-GIB")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fashion_acc.pdf"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(vgib["epoch"], vgib["curv"])
    plt.xlabel("Epoch")
    plt.ylabel("Curvature proxy")
    plt.title("Fashion-MNIST: curvature dynamics (V-GIB)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fashion_curv.pdf"), bbox_inches="tight")
    plt.close()

    corr = vgib[["val_acc", "curv", "idim", "mi_proxy"]].corr()
    print("\n[Fashion-MNIST] correlation matrix (V-GIB, seed=0):")
    print(corr.round(3))


def cifar_plots(df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    # final epoch per run
    final = df.groupby(
        ["model", "seed", "label_frac"], as_index=False
    ).last()

    # average over seeds
    summary = final.groupby(["model", "label_frac"], as_index=False).agg({
        "val_acc": ["mean", "std"],
        "curv": ["mean"],
        "idim": ["mean"],
        "efficiency": ["mean"],
        "wall_time": ["mean"],
    })
    summary.columns = [
        "model", "label_frac",
        "val_acc_mean", "val_acc_std",
        "curv_mean", "idim_mean",
        "eff_mean", "time_mean",
    ]
    print("\n[CIFAR-10] summary (final epoch):")
    print(summary.round(4))

    # accuracy vs label fraction
    plt.figure(figsize=(6, 4))
    for model in ["baseline", "vib", "vgib", "laplacian"]:
        sub = summary[summary["model"] == model]
        if sub.empty:
            continue
        plt.errorbar(
            sub["label_frac"],
            sub["val_acc_mean"],
            yerr=sub["val_acc_std"],
            marker="o",
            capsize=3,
            label=model,
        )
    plt.xlabel("Label fraction")
    plt.ylabel("Test accuracy")
    plt.title("CIFAR-10: accuracy vs labeled fraction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cifar_acc_vs_frac.pdf"), bbox_inches="tight")
    plt.close()

    # efficiency vs label fraction
    plt.figure(figsize=(6, 4))
    for model in ["baseline", "vib", "vgib", "laplacian"]:
        sub = summary[summary["model"] == model]
        if sub.empty:
            continue
        plt.plot(sub["label_frac"], sub["eff_mean"], marker="o", label=model)
    plt.xlabel("Label fraction")
    plt.ylabel("Mean efficiency (acc / align proxy)")
    plt.title("CIFAR-10: interpretive efficiency vs fraction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cifar_eff_vs_frac.pdf"), bbox_inches="tight")
    plt.close()

    # wall-clock vs model
    plt.figure(figsize=(6, 4))
    bar_data = summary.groupby("model", as_index=False)["time_mean"].mean()
    plt.bar(bar_data["model"], bar_data["time_mean"])
    plt.ylabel("Mean wall time per epoch (s)")
    plt.title("CIFAR-10: computational overhead by model")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cifar_time_bar.pdf"), bbox_inches="tight")
    plt.close()


def main():
    logs_dir = "./logs"
    figs_dir = "./figs"
    swiss, fashion, cifar = load_logs(base_dir=logs_dir)

    swiss_plots(swiss, out_dir=os.path.join(figs_dir, "swiss"))
    fashion_plots(fashion, out_dir=os.path.join(figs_dir, "fashion"))
    cifar_plots(cifar, out_dir=os.path.join(figs_dir, "cifar"))

    print("\nPlots written under", figs_dir)


if __name__ == "__main__":
    main()
