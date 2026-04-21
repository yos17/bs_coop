from __future__ import annotations

from localized_algorithm import run_ldbpa


def run_global_benchmark(network: dict, config) -> dict:
    return run_ldbpa(
        network,
        rho=float("inf"),
        max_iters=config.max_iters,
        tol=config.tol,
        lambda_step=config.lambda_step,
        price_step=0.0,
        interference_budget_scale=config.interference_budget_scale,
        label="global",
    )
