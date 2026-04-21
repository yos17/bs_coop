from __future__ import annotations

import matplotlib.pyplot as plt

from config import DEFAULT_CONFIG, FIG_RESULTS_DIR, PAPER_FIG_DIR, RAW_RESULTS_DIR
from utils import ensure_dir, load_csv


plt.style.use("seaborn-v0_8-whitegrid")


def save_figure(fig, stem: str) -> None:
    ensure_dir(FIG_RESULTS_DIR)
    ensure_dir(PAPER_FIG_DIR)
    for directory in (FIG_RESULTS_DIR, PAPER_FIG_DIR):
        fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")


def load_required_csv(filename: str) -> list[dict]:
    path = RAW_RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}. Run python code/experiments.py first.")
    return load_csv(path)


def plot_sum_rate(summary_rows: list[dict]) -> None:
    labels = [row["rho_label"] for row in summary_rows]
    means = [float(row["mean_weighted_sum_rate"]) for row in summary_rows]
    stds = [float(row["std_weighted_sum_rate"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.errorbar(labels, means, yerr=stds, marker="o", linewidth=2.0, capsize=4.0, color="#0B7285")
    ax.set_xlabel("Cooperation setting")
    ax.set_ylabel("Weighted sum-rate")
    ax.set_title("Sum-rate versus cooperation radius")
    save_figure(fig, "sum_rate_vs_rho")
    plt.close(fig)


def plot_gap(summary_rows: list[dict]) -> None:
    filtered = [row for row in summary_rows if row["rho_label"] != "global"]
    labels = [row["rho_label"] for row in filtered]
    means = [float(row["mean_gap_to_global"]) for row in filtered]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(labels, means, marker="s", linewidth=2.0, color="#C92A2A")
    ax.set_xlabel("Cooperation setting")
    ax.set_ylabel("Gap to global benchmark")
    ax.set_title("Optimality gap versus radius")
    save_figure(fig, "gap_to_global_vs_rho")
    plt.close(fig)


def plot_runtime(summary_rows: list[dict]) -> None:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in summary_rows:
        grouped.setdefault(row["label"], []).append((int(row["num_bs"]), float(row["mean_runtime_sec"])))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    colors = {"localized-rho1": "#2B8A3E", "global": "#5F3DC4"}
    for label, pairs in grouped.items():
        pairs.sort(key=lambda item: item[0])
        xs = [item[0] for item in pairs]
        ys = [item[1] for item in pairs]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=label, color=colors.get(label))
    ax.set_xlabel("Number of base stations")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime scaling")
    ax.legend(frameon=True)
    save_figure(fig, "runtime_vs_bs")
    plt.close(fig)


def plot_signaling(summary_rows: list[dict]) -> None:
    labels = [row["rho_label"] for row in summary_rows]
    means = [float(row["mean_signaling_messages"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.bar(labels, means, color="#F08C00")
    ax.set_xlabel("Cooperation setting")
    ax.set_ylabel("Directed scalar messages")
    ax.set_title("Signaling cost versus radius")
    save_figure(fig, "signaling_vs_rho")
    plt.close(fig)


def plot_locality_scaling(summary_rows: list[dict]) -> None:
    rhos = [float(row["rho"]) for row in summary_rows]
    gaps = [float(row["mean_gap_to_global"]) for row in summary_rows]
    reference = [gaps[0] * (rho / rhos[0]) ** (2.0 - DEFAULT_CONFIG.path_loss_exponent) for rho in rhos]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.loglog(rhos, gaps, marker="o", linewidth=2.0, color="#C2255C", label="Measured true gap")
    ax.loglog(rhos, reference, linestyle="--", linewidth=2.0, color="#364FC7", label=r"Reference $\rho^{2-\alpha}$")
    ax.set_xlabel(r"Cooperation radius $\rho$")
    ax.set_ylabel("Gap to global benchmark")
    ax.set_title("Locality scaling on the 37-BS study")
    ax.legend(frameon=True)
    save_figure(fig, "locality_gap_scaling")
    plt.close(fig)


def plot_alpha_sweep(summary_rows: list[dict]) -> None:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in summary_rows:
        label = rf"$\alpha={float(row['alpha']):.1f}$"
        grouped.setdefault(label, []).append((float(row["rho"]), float(row["mean_gap_to_global"])))

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    colors = ["#1D4ED8", "#C2255C", "#2B8A3E"]
    for color, (label, pairs) in zip(colors, sorted(grouped.items())):
        pairs.sort(key=lambda item: item[0])
        xs = [item[0] for item in pairs]
        ys = [item[1] for item in pairs]
        ax.loglog(xs, ys, marker="o", linewidth=2.0, color=color, label=label)
    ax.set_xlabel(r"Cooperation radius $\rho$")
    ax.set_ylabel("Gap to global benchmark")
    ax.set_title("Effect of path-loss decay on locality")
    ax.legend(frameon=True)
    save_figure(fig, "alpha_sweep_gap_scaling")
    plt.close(fig)


def main() -> None:
    radius_summary = load_required_csv("radius_sweep_summary.csv")
    runtime_summary = load_required_csv("runtime_scaling_summary.csv")
    locality_summary = load_required_csv("locality_scaling_summary.csv")
    alpha_summary = load_required_csv("alpha_sweep_summary.csv")

    plot_sum_rate(radius_summary)
    plot_gap(radius_summary)
    plot_runtime(runtime_summary)
    plot_signaling(radius_summary)
    plot_locality_scaling(locality_summary)
    plot_alpha_sweep(alpha_summary)
    print("Saved figures to", FIG_RESULTS_DIR, "and", PAPER_FIG_DIR)


if __name__ == "__main__":
    main()
