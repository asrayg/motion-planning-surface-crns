"""
Generate plots from sweep.csv for the paper.
"""

import sys
import csv
import os
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Need matplotlib and numpy: pip install matplotlib numpy")
    sys.exit(1)


def load_sweep(path="data/sweep.csv"):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["reached"] != "True":
                continue
            rows.append({
                "n_param": int(row["n_param"]),
                "density": float(row["density"]),
                "seed": int(row["seed"]),
                "n_traversable": int(row["n_traversable"]),
                "d_shortest": int(row["d_shortest"]),
                "path_length": int(row["path_length"]) if row["path_length"] else None,
                "reactions": int(row["reactions"]),
            })
    return rows


def plot_reactions_vs_n(rows, out_path):
    """Plot reactions vs n (traversable sites), one series per density."""
    fig, ax = plt.subplots(figsize=(6, 4))
    by_density = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_density[r["density"]][r["n_traversable"]].append(r["reactions"])

    markers = {"0.0": "o", "0.1": "s", "0.2": "^", "0.3": "D"}
    for density in sorted(by_density.keys()):
        ns = sorted(by_density[density].keys())
        means = [np.mean(by_density[density][n]) for n in ns]
        stds = [np.std(by_density[density][n]) for n in ns]
        marker = markers.get(f"{density:.1f}", "o")
        ax.errorbar(ns, means, yerr=stds, marker=marker, capsize=3,
                    label=fr"$\rho={density:.1f}$", markersize=6,
                    linewidth=1.2)

    # Plot reference line: y = c * n
    all_ns = sorted({r["n_traversable"] for r in rows})
    ref_ns = np.array(all_ns)
    c_fit = np.mean([r["reactions"] / r["n_traversable"] for r in rows])
    ax.plot(ref_ns, c_fit * ref_ns, "k--", alpha=0.4,
            label=fr"$y \propto n$ (slope $\approx$ {c_fit:.2f})")

    ax.set_xlabel(r"Traversable sites $n = |G|$")
    ax.set_ylabel("Expected reactions to termination")
    ax.set_title(r"Scaling of $\mathbb{E}[T]$ vs. $n$")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_reactions_vs_d(rows, out_path):
    """Plot reactions vs shortest-path distance, one series per density."""
    fig, ax = plt.subplots(figsize=(6, 4))
    by_density = defaultdict(list)
    for r in rows:
        by_density[r["density"]].append((r["d_shortest"], r["reactions"]))

    colors = {"0.0": "C0", "0.1": "C1", "0.2": "C2", "0.3": "C3"}
    for density in sorted(by_density.keys()):
        points = sorted(by_density[density])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, s=15, alpha=0.5,
                   color=colors.get(f"{density:.1f}", "gray"),
                   label=fr"$\rho={density:.1f}$")

    ax.set_xlabel(r"Shortest-path distance $d = \mathrm{dist}_G(s, t)$")
    ax.set_ylabel("Reactions to termination")
    ax.set_title(r"Reactions vs. $d$")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_path_optimality(rows, out_path):
    """Plot path_length / d (optimality ratio) across instances."""
    fig, ax = plt.subplots(figsize=(6, 4))
    by_density = defaultdict(list)
    for r in rows:
        if r["d_shortest"] > 0 and r["path_length"] is not None:
            ratio = r["path_length"] / r["d_shortest"]
            by_density[r["density"]].append((r["n_traversable"], ratio))

    colors = {"0.0": "C0", "0.1": "C1", "0.2": "C2", "0.3": "C3"}
    for density in sorted(by_density.keys()):
        points = sorted(by_density[density])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, s=15, alpha=0.5,
                   color=colors.get(f"{density:.1f}", "gray"),
                   label=fr"$\rho={density:.1f}$")

    ax.axhline(1.0, color="k", linestyle="--", alpha=0.4,
               label="Optimal ($L = d$)")
    ax.set_xlabel(r"Traversable sites $n$")
    ax.set_ylabel(r"Path length / shortest distance ($L / d$)")
    ax.set_title(r"Path optimality ratio for base construction")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_ratio_vs_n_plus_d(rows, out_path):
    """Sanity check: reactions / (n + d) should be roughly constant."""
    fig, ax = plt.subplots(figsize=(6, 4))
    by_density = defaultdict(list)
    for r in rows:
        n_plus_d = r["n_traversable"] + r["d_shortest"]
        ratio = r["reactions"] / n_plus_d
        by_density[r["density"]].append((r["n_traversable"], ratio))

    colors = {"0.0": "C0", "0.1": "C1", "0.2": "C2", "0.3": "C3"}
    for density in sorted(by_density.keys()):
        points = sorted(by_density[density])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, s=15, alpha=0.5,
                   color=colors.get(f"{density:.1f}", "gray"),
                   label=fr"$\rho={density:.1f}$")

    ax.set_xlabel(r"Traversable sites $n$")
    ax.set_ylabel(r"Reactions / $(n + d)$")
    ax.set_title(r"Empirical $\Theta(n + d)$ scaling")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def print_summary_table(rows):
    """Print a LaTeX-ready summary table."""
    by_config = defaultdict(list)
    for r in rows:
        key = (r["n_param"], r["density"])
        by_config[key].append(r)

    print("\n% LaTeX summary table")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{rrrrrrr}")
    print(r"\toprule")
    print(r"Grid & $\rho$ & Trials & Mean $n$ & Mean $d$ & Mean reactions & Mean $L/d$ \\")
    print(r"\midrule")
    for (n_param, density), trials in sorted(by_config.items()):
        n_traversable = np.mean([t["n_traversable"] for t in trials])
        d = np.mean([t["d_shortest"] for t in trials])
        reactions = np.mean([t["reactions"] for t in trials])
        l_over_d = np.mean([t["path_length"] / t["d_shortest"]
                            for t in trials
                            if t["d_shortest"] > 0 and t["path_length"]])
        print(f"${n_param}\\times{n_param}$ & {density:.1f} & "
              f"{len(trials)} & {n_traversable:.1f} & {d:.1f} & "
              f"{reactions:.1f} & {l_over_d:.3f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Empirical results across grid sizes and obstacle densities.}")
    print(r"\label{tab:sweep-results}")
    print(r"\end{table}")


def main():
    os.makedirs("figures", exist_ok=True)
    rows = load_sweep()
    print(f"Loaded {len(rows)} successful trials.")

    plot_reactions_vs_n(rows, "figures/scaling_n.png")
    plot_reactions_vs_d(rows, "figures/scaling_d.png")
    plot_ratio_vs_n_plus_d(rows, "figures/scaling_ratio.png")
    plot_path_optimality(rows, "figures/path_optimality.png")
    print_summary_table(rows)


if __name__ == "__main__":
    main()