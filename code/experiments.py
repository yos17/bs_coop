from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from math import log
from math import isinf

from channels import build_network
from config import DEFAULT_CONFIG, FIG_RESULTS_DIR, PAPER_FIG_DIR, RAW_RESULTS_DIR
from global_benchmark import run_global_benchmark
from localized_algorithm import run_ldbpa
from utils import ensure_dir, save_csv


def format_rho_label(rho: float, label: str) -> str:
    if label == "non-cooperative":
        return "non-coop"
    if isinf(rho):
        return "global"
    return f"rho={int(rho)}"


def summarize(rows: list[dict], group_keys: list[str], metric_keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        group_value = tuple(row[key] for key in group_keys)
        grouped[group_value].append(row)

    summary = []
    for group_value, items in grouped.items():
        exemplar = items[0]
        record = {}
        for key, value in zip(group_keys, group_value):
            record[key] = value
        for key in ("label", "rho_label", "rho_numeric", "num_bs"):
            if key in exemplar and key not in record:
                record[key] = exemplar[key]
        for metric in metric_keys:
            values = [float(item[metric]) for item in items]
            record[f"mean_{metric}"] = sum(values) / len(values)
            variance = sum((value - record[f"mean_{metric}"]) ** 2 for value in values) / len(values)
            record[f"std_{metric}"] = variance ** 0.5
        summary.append(record)
    return summary


def run_radius_sweep(config) -> tuple[list[dict], list[dict], list[dict]]:
    radius_rows: list[dict] = []
    convergence_rows: list[dict] = []
    methods = [("non-cooperative", 0.0)] + [(f"localized-rho{int(rho)}", rho) for rho in config.rho_values]

    for trial in range(config.monte_carlo_trials):
        network = build_network(config, ring_radius=config.ring_radius, users_per_bs=config.users_per_bs, seed=config.seed + trial)
        global_result = run_global_benchmark(network, config)

        method_results = [global_result]
        for label, rho in methods:
            method_results.append(
                run_ldbpa(
                    network,
                    rho=rho,
                    max_iters=config.max_iters,
                    tol=config.tol,
                    lambda_step=config.lambda_step,
                    price_step=config.price_step,
                    interference_budget_scale=config.interference_budget_scale,
                    label=label,
                )
            )

        global_objective = global_result["objective"]
        for result in method_results:
            rho = result["rho"]
            row = {
                "trial": trial,
                "label": result["label"],
                "rho_numeric": -1.0 if isinf(rho) else rho,
                "rho_label": format_rho_label(rho, result["label"]),
                "weighted_sum_rate": result["objective"],
                "runtime_sec": result["runtime_sec"],
                "signaling_messages": result["signaling_messages"],
                "iterations": result["iterations"],
                "gap_to_global": global_objective - result["objective"],
            }
            radius_rows.append(row)

        if trial == 0:
            for result in method_results:
                for point in result["history"]:
                    convergence_rows.append(
                        {
                            "label": result["label"],
                            "rho_label": format_rho_label(result["rho"], result["label"]),
                            "iteration": point["iteration"],
                            "objective": point["objective"],
                            "relative_change": point["relative_change"],
                        }
                    )

    summary = summarize(
        rows=radius_rows,
        group_keys=["rho_label"],
        metric_keys=["weighted_sum_rate", "runtime_sec", "signaling_messages", "iterations", "gap_to_global"],
    )
    order = {"non-coop": 0, "rho=1": 1, "rho=2": 2, "rho=3": 3, "global": 4}
    summary.sort(key=lambda item: order[item["rho_label"]])
    return radius_rows, summary, convergence_rows


def run_runtime_scaling(config) -> tuple[list[dict], list[dict]]:
    runtime_rows: list[dict] = []
    for ring in config.scaling_rings:
        for trial in range(config.runtime_trials):
            seed = 1000 + 17 * ring + trial
            network = build_network(config, ring_radius=ring, users_per_bs=config.runtime_users_per_bs, seed=seed)
            localized = run_ldbpa(
                network,
                rho=1.0,
                max_iters=max(20, config.max_iters // 2),
                tol=config.tol,
                lambda_step=config.lambda_step,
                price_step=config.price_step,
                interference_budget_scale=config.interference_budget_scale,
                label="localized-rho1",
            )
            global_result = run_global_benchmark(network, config)

            for result in (localized, global_result):
                runtime_rows.append(
                    {
                        "trial": trial,
                        "num_bs": network["num_bs"],
                        "label": result["label"],
                        "runtime_sec": result["runtime_sec"],
                        "iterations": result["iterations"],
                    }
                )

    summary = summarize(rows=runtime_rows, group_keys=["label", "num_bs"], metric_keys=["runtime_sec", "iterations"])
    summary.sort(key=lambda item: (item["label"], int(item["num_bs"])))
    return runtime_rows, summary


def fit_loglog_slope(summary_rows: list[dict], x_key: str, y_key: str) -> float:
    xs = [log(float(row[x_key])) for row in summary_rows if float(row[y_key]) > 0.0]
    ys = [log(float(row[y_key])) for row in summary_rows if float(row[y_key]) > 0.0]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x_val - mean_x) * (y_val - mean_y) for x_val, y_val in zip(xs, ys))
    denominator = sum((x_val - mean_x) ** 2 for x_val in xs)
    return numerator / denominator


def run_locality_scaling(config) -> tuple[list[dict], list[dict], list[dict]]:
    locality_rows: list[dict] = []
    for trial in range(config.locality_trials):
        seed = 3000 + trial
        network = build_network(
            config,
            ring_radius=config.locality_ring_radius,
            users_per_bs=config.locality_users_per_bs,
            seed=seed,
        )
        global_result = run_global_benchmark(network, config)
        global_objective = global_result["objective"]

        for rho in config.locality_rhos:
            result = run_ldbpa(
                network,
                rho=rho,
                max_iters=config.max_iters,
                tol=config.tol,
                lambda_step=config.lambda_step,
                price_step=config.price_step,
                interference_budget_scale=config.interference_budget_scale,
                label=f"localized-rho{int(rho)}",
            )
            locality_rows.append(
                {
                    "trial": trial,
                    "num_bs": network["num_bs"],
                    "rho": rho,
                    "weighted_sum_rate": result["objective"],
                    "gap_to_global": global_objective - result["objective"],
                    "runtime_sec": result["runtime_sec"],
                    "iterations": result["iterations"],
                }
            )

    summary = summarize(
        rows=locality_rows,
        group_keys=["rho"],
        metric_keys=["weighted_sum_rate", "gap_to_global", "runtime_sec", "iterations"],
    )
    summary.sort(key=lambda item: float(item["rho"]))

    empirical_slope = fit_loglog_slope(summary, "rho", "mean_gap_to_global")
    fit_rows = [
        {
            "empirical_loglog_slope": empirical_slope,
            "predicted_upper_bound_exponent": 2.0 - config.path_loss_exponent,
            "ring_radius": config.locality_ring_radius,
            "num_bs": (1 + 3 * config.locality_ring_radius * (config.locality_ring_radius + 1)),
            "users_per_bs": config.locality_users_per_bs,
            "trials": config.locality_trials,
        }
    ]
    return locality_rows, summary, fit_rows


def run_alpha_sweep(config) -> tuple[list[dict], list[dict], list[dict]]:
    alpha_rows: list[dict] = []
    for alpha in config.alpha_sweep_values:
        local_config = replace(config, path_loss_exponent=alpha)
        for trial in range(config.alpha_sweep_trials):
            seed = 5000 + 100 * int(10 * alpha) + trial
            network = build_network(
                local_config,
                ring_radius=config.locality_ring_radius,
                users_per_bs=config.locality_users_per_bs,
                seed=seed,
            )
            global_result = run_global_benchmark(network, local_config)
            global_objective = global_result["objective"]

            for rho in config.locality_rhos:
                result = run_ldbpa(
                    network,
                    rho=rho,
                    max_iters=local_config.max_iters,
                    tol=local_config.tol,
                    lambda_step=local_config.lambda_step,
                    price_step=local_config.price_step,
                    interference_budget_scale=local_config.interference_budget_scale,
                    label=f"alpha={alpha:.1f}",
                )
                alpha_rows.append(
                    {
                        "trial": trial,
                        "alpha": alpha,
                        "rho": rho,
                        "num_bs": network["num_bs"],
                        "weighted_sum_rate": result["objective"],
                        "gap_to_global": global_objective - result["objective"],
                    }
                )

    summary = summarize(
        rows=alpha_rows,
        group_keys=["alpha", "rho"],
        metric_keys=["weighted_sum_rate", "gap_to_global"],
    )
    summary.sort(key=lambda item: (float(item["alpha"]), float(item["rho"])))
    fit_rows = []
    for alpha in config.alpha_sweep_values:
        alpha_summary = [row for row in summary if float(row["alpha"]) == alpha]
        fit_rows.append(
            {
                "alpha": alpha,
                "empirical_loglog_slope": fit_loglog_slope(alpha_summary, "rho", "mean_gap_to_global"),
                "predicted_upper_bound_exponent": 2.0 - alpha,
                "ring_radius": config.locality_ring_radius,
                "num_bs": 1 + 3 * config.locality_ring_radius * (config.locality_ring_radius + 1),
                "users_per_bs": config.locality_users_per_bs,
                "trials": config.alpha_sweep_trials,
            }
        )
    return alpha_rows, summary, fit_rows


def main() -> None:
    config = DEFAULT_CONFIG
    ensure_dir(RAW_RESULTS_DIR)
    ensure_dir(FIG_RESULTS_DIR)
    ensure_dir(PAPER_FIG_DIR)

    radius_rows, radius_summary, convergence_rows = run_radius_sweep(config)
    runtime_rows, runtime_summary = run_runtime_scaling(config)
    locality_rows, locality_summary, locality_fit = run_locality_scaling(config)
    alpha_rows, alpha_summary, alpha_fit = run_alpha_sweep(config)

    save_csv(
        RAW_RESULTS_DIR / "radius_sweep_trials.csv",
        radius_rows,
        ["trial", "label", "rho_numeric", "rho_label", "weighted_sum_rate", "runtime_sec", "signaling_messages", "iterations", "gap_to_global"],
    )
    save_csv(
        RAW_RESULTS_DIR / "radius_sweep_summary.csv",
        radius_summary,
        [
            "rho_label",
            "label",
            "rho_numeric",
            "mean_weighted_sum_rate",
            "std_weighted_sum_rate",
            "mean_runtime_sec",
            "std_runtime_sec",
            "mean_signaling_messages",
            "std_signaling_messages",
            "mean_iterations",
            "std_iterations",
            "mean_gap_to_global",
            "std_gap_to_global",
        ],
    )
    save_csv(
        RAW_RESULTS_DIR / "runtime_scaling_trials.csv",
        runtime_rows,
        ["trial", "num_bs", "label", "runtime_sec", "iterations"],
    )
    save_csv(
        RAW_RESULTS_DIR / "runtime_scaling_summary.csv",
        runtime_summary,
        ["label", "num_bs", "mean_runtime_sec", "std_runtime_sec", "mean_iterations", "std_iterations"],
    )
    save_csv(
        RAW_RESULTS_DIR / "convergence_history.csv",
        convergence_rows,
        ["label", "rho_label", "iteration", "objective", "relative_change"],
    )
    save_csv(
        RAW_RESULTS_DIR / "locality_scaling_trials.csv",
        locality_rows,
        ["trial", "num_bs", "rho", "weighted_sum_rate", "gap_to_global", "runtime_sec", "iterations"],
    )
    save_csv(
        RAW_RESULTS_DIR / "locality_scaling_summary.csv",
        locality_summary,
        [
            "rho",
            "mean_weighted_sum_rate",
            "std_weighted_sum_rate",
            "mean_gap_to_global",
            "std_gap_to_global",
            "mean_runtime_sec",
            "std_runtime_sec",
            "mean_iterations",
            "std_iterations",
            "num_bs",
        ],
    )
    save_csv(
        RAW_RESULTS_DIR / "locality_scaling_fit.csv",
        locality_fit,
        [
            "empirical_loglog_slope",
            "predicted_upper_bound_exponent",
            "ring_radius",
            "num_bs",
            "users_per_bs",
            "trials",
        ],
    )
    save_csv(
        RAW_RESULTS_DIR / "alpha_sweep_trials.csv",
        alpha_rows,
        ["trial", "alpha", "rho", "num_bs", "weighted_sum_rate", "gap_to_global"],
    )
    save_csv(
        RAW_RESULTS_DIR / "alpha_sweep_summary.csv",
        alpha_summary,
        [
            "alpha",
            "rho",
            "num_bs",
            "mean_weighted_sum_rate",
            "std_weighted_sum_rate",
            "mean_gap_to_global",
            "std_gap_to_global",
        ],
    )
    save_csv(
        RAW_RESULTS_DIR / "alpha_sweep_fit.csv",
        alpha_fit,
        [
            "alpha",
            "empirical_loglog_slope",
            "predicted_upper_bound_exponent",
            "ring_radius",
            "num_bs",
            "users_per_bs",
            "trials",
        ],
    )

    print("Saved experiment outputs to", RAW_RESULTS_DIR)


if __name__ == "__main__":
    main()
