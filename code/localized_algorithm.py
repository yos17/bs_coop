from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.linalg import solve

from utils import (
    compute_user_metrics,
    estimate_signaling_messages,
    far_field_leakage,
    initialize_beamformers,
    neighborhoods,
    per_bs_power,
    served_user_lists,
)


def run_ldbpa(
    network: dict,
    rho: float,
    *,
    max_iters: int,
    tol: float,
    lambda_step: float,
    price_step: float,
    interference_budget_scale: float,
    label: str,
) -> dict:
    channels = network["channels"]
    user_bs = network["user_bs"]
    weights = network["weights"]
    power_limits = network["power_limits"]
    noise_power = network["noise_power"]
    bs_positions = network["bs_positions"]

    num_bs = network["num_bs"]
    num_users = network["num_users"]
    antenna_count = network["antenna_count"]

    served_users = served_user_lists(user_bs, num_bs)
    neighborhood_sets = neighborhoods(bs_positions, rho)
    beamformers = initialize_beamformers(channels, user_bs, power_limits)
    lambdas = np.zeros(num_bs, dtype=float)
    prices = np.zeros(num_users, dtype=float)

    history = []
    start_time = perf_counter()

    for iteration in range(1, max_iters + 1):
        metrics = compute_user_metrics(beamformers, channels, user_bs, noise_power, weights)
        desired_complex = metrics["desired_complex"]
        total_power = metrics["total_power"]

        receivers = desired_complex / np.maximum(total_power, 1e-12)
        mse = 1.0 - 2.0 * np.real(np.conjugate(receivers) * desired_complex) + (np.abs(receivers) ** 2) * total_power
        mse_weights = 1.0 / np.maximum(mse, 1e-9)

        updated_beamformers = np.zeros_like(beamformers)
        for bs_idx in range(num_bs):
            local_users = np.flatnonzero(np.isin(user_bs, neighborhood_sets[bs_idx]))
            local_matrix = np.zeros((antenna_count, antenna_count), dtype=np.complex128)
            for user in local_users:
                channel_vec = channels[user, bs_idx, :]
                outer = np.outer(channel_vec, np.conjugate(channel_vec))
                local_matrix += weights[user] * mse_weights[user] * (np.abs(receivers[user]) ** 2) * outer
                local_matrix += prices[user] * outer
            local_matrix += (lambdas[bs_idx] + 1e-6) * np.eye(antenna_count)

            for user in served_users[bs_idx]:
                drive = weights[user] * mse_weights[user] * np.conjugate(receivers[user]) * channels[user, bs_idx, :]
                updated_beamformers[bs_idx, user, :] = solve(local_matrix, drive, assume_a="her")

            used_power = per_bs_power(updated_beamformers)[bs_idx]
            if used_power > power_limits[bs_idx]:
                updated_beamformers[bs_idx, served_users[bs_idx], :] *= np.sqrt(power_limits[bs_idx] / used_power)

        used_power = per_bs_power(updated_beamformers)
        lambdas = np.maximum(0.0, lambdas + lambda_step * (used_power - power_limits))

        leakage = far_field_leakage(updated_beamformers, channels, user_bs, neighborhood_sets)
        leakage_budget = interference_budget_scale * noise_power * np.ones(num_users, dtype=float)
        prices = np.maximum(0.0, prices + price_step * (leakage - leakage_budget))

        new_metrics = compute_user_metrics(updated_beamformers, channels, user_bs, noise_power, weights)
        relative_change = np.linalg.norm(updated_beamformers - beamformers) / max(np.linalg.norm(beamformers), 1e-12)
        history.append(
            {
                "iteration": iteration,
                "objective": new_metrics["objective"],
                "relative_change": float(relative_change),
                "max_lambda": float(np.max(lambdas)) if lambdas.size else 0.0,
                "avg_price": float(prices.mean()),
            }
        )

        beamformers = updated_beamformers
        if relative_change < tol:
            break

    runtime_sec = perf_counter() - start_time
    final_metrics = compute_user_metrics(beamformers, channels, user_bs, noise_power, weights)
    iterations = len(history)
    signaling_messages = estimate_signaling_messages(neighborhood_sets, served_users, iterations)

    return {
        "label": label,
        "rho": rho,
        "beamformers": beamformers,
        "objective": final_metrics["objective"],
        "sinr": final_metrics["sinr"],
        "rates": final_metrics["rates"],
        "iterations": iterations,
        "runtime_sec": runtime_sec,
        "signaling_messages": signaling_messages,
        "history": history,
        "lambdas": lambdas,
        "prices": prices,
    }
