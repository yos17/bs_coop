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
    streams_per_user = network["streams_per_user"]

    served_users = served_user_lists(user_bs, num_bs)
    neighborhood_sets = neighborhoods(bs_positions, rho)
    beamformers = initialize_beamformers(channels, user_bs, power_limits, streams_per_user)
    lambdas = np.zeros(num_bs, dtype=float)
    prices = np.zeros(num_users, dtype=float)

    history = []
    start_time = perf_counter()

    for iteration in range(1, max_iters + 1):
        metrics = compute_user_metrics(beamformers, channels, user_bs, noise_power, weights)
        receivers = metrics["receivers"]
        mse_matrices = metrics["mse_matrices"]
        weight_matrices = np.zeros_like(mse_matrices)
        identity_d = np.eye(streams_per_user, dtype=np.complex128)
        for user in range(num_users):
            weight_matrices[user] = np.linalg.inv(mse_matrices[user] + 1e-9 * identity_d)

        updated_beamformers = np.zeros_like(beamformers)
        for bs_idx in range(num_bs):
            local_users = np.flatnonzero(np.isin(user_bs, neighborhood_sets[bs_idx]))
            local_matrix = np.zeros((antenna_count, antenna_count), dtype=np.complex128)
            for user in local_users:
                channel_matrix = channels[user, bs_idx, :, :]
                local_matrix += weights[user] * (
                    channel_matrix.conj().T
                    @ receivers[user]
                    @ weight_matrices[user]
                    @ receivers[user].conj().T
                    @ channel_matrix
                )
                local_matrix += prices[user] * (channel_matrix.conj().T @ channel_matrix)
            local_matrix += (lambdas[bs_idx] + 1e-6) * np.eye(antenna_count)

            for user in served_users[bs_idx]:
                channel_matrix = channels[user, bs_idx, :, :]
                drive = weights[user] * (channel_matrix.conj().T @ receivers[user] @ weight_matrices[user])
                updated_beamformers[bs_idx, user, :, :] = solve(local_matrix, drive, assume_a="her")

            used_power = per_bs_power(updated_beamformers)[bs_idx]
            if used_power > power_limits[bs_idx]:
                updated_beamformers[bs_idx, served_users[bs_idx], :, :] *= np.sqrt(power_limits[bs_idx] / used_power)

        used_power = per_bs_power(updated_beamformers)
        lambdas = np.maximum(0.0, lambdas + lambda_step * (used_power - power_limits))

        new_metrics = compute_user_metrics(updated_beamformers, channels, user_bs, noise_power, weights)
        leakage = far_field_leakage(new_metrics["responses"], user_bs, neighborhood_sets)
        leakage_budget = interference_budget_scale * np.maximum(new_metrics["noise_after_combining"], 1e-12)
        prices = np.maximum(0.0, prices + price_step * (leakage - leakage_budget))
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
