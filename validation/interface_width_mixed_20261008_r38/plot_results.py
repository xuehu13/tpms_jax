"""Reproduce the r36-r38 figures from frozen result files; no mechanics solve."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    data = args.data or Path(__file__).resolve().parents[1]
    read = lambda name: json.loads((data / name / "result.json").read_text(encoding="utf-8"))
    width = read("interface_width_20261008_r37")
    mixed = read("interface_width_mixed_20261008_r38")
    rows = width["cases"]
    shell = mixed["shell_stiffness_N_per_mm"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    args.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.1, 5.0), constrained_layout=True)
    x = np.array([r["interface_10_90_mm"] for r in rows])
    y = np.array([100*r["relative_to_shell"] for r in rows])
    ax.plot(x, y, "o-", color="#24557a", lw=1.8, label="Original 27-point equilibrium")
    xm = [.025, .05]
    ym = [100*mixed["relative_to_shell"], 100*mixed["same_rule_width005_relative_to_shell"]]
    ax.plot(xm, ym, "s--", color="#a45c29", lw=1.8, label="Same selected-12 / rest-27 equilibrium")
    for a,b in zip(x,y):
        ax.annotate(f"{b:.4f}%", (a,b), xytext=(0, -20), textcoords="offset points", ha="center", color="#24557a")
    for a,b in zip(xm,ym):
        ax.annotate(f"{b:.4f}%", (a,b), xytext=(0, 10), textcoords="offset points", ha="center", color="#a45c29")
    ax.axhline(0, color="#666666", lw=1, label=f"Matched shell: K = {shell:.6f} N/mm")
    ax.set(xlabel="10-90% interface width (mm); narrower = steeper", ylabel="Initial stiffness difference from matched shell (%)",
           title="diverse_04, N32: narrowing helps modestly, while integration changes the apparent benefit",
           xticks=x, ylim=(-.4, 9.2))
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="lower right", fontsize=9.6)
    fig.savefig(args.output / "width_response.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), constrained_layout=True)
    z = np.linspace(.14, .36, 800)
    colors = ["#24557a", "#347e59", "#a45c29"]
    for r,c in zip(rows, colors):
        w = r["interface_10_90_mm"]
        phi = 1/(1+np.exp((z-.25)/(w/(2*np.log(9)))))
        axes[0].plot(z, phi, color=c, label=f"width {w:g} mm")
    axes[0].axvline(.25, color="#666666", ls=":", lw=1)
    axes[0].set(xlabel="Distance to mid-surface d (mm)", ylabel="Occupancy phi", title="Same nominal boundary d = t/2 = 0.25 mm")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=.2)
    local = width["local_integration"]
    values = [100*r["dense12_relative_change_scaled_by_own_full_energy"] for r in local]
    axes[1].bar(range(3), values, color=colors, width=.58)
    for i,v in enumerate(values):
        axes[1].text(i,v+.12,f"{v:.4f}%",ha="center",fontsize=10)
    axes[1].set(xticks=range(3), xticklabels=[f"{r['width_mm']:g}" for r in local], ylim=(0,5.4),
                xlabel="10-90% interface width (mm)", ylabel="Selected energy increase / own full energy (%)",
                title="Fixed-state dense check: 1941 frozen cells only")
    axes[1].grid(axis="y",alpha=.2)
    fig.suptitle("A sharper transition is more demanding to integrate; these bars are not stiffness corrections", fontsize=12)
    fig.savefig(args.output / "transition_integration.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
