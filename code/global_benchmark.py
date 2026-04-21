from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, minimize

from localized_algorithm import run_ldbpa
from utils import compute_user_metrics, estimate_signaling_messages


def _global_user_channels(channels: np.ndarray) -> list[np.ndarray]:
    num_users, num_bs, _, _ = channels.shape
    return [np.concatenate([channels[user, bs_idx, :, :] for bs_idx in range(num_bs)], axis=1) for user in range(num_users)]


def _nullspace_basis(matrix: np.ndarray, *, tol: float = 1e-9) -> np.ndarray:
    if matrix.size == 0:
        return np.eye(matrix.shape[1], dtype=np.complex128)

    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    if singular_values.size == 0:
        rank = 0
    else:
        rank = int(np.sum(singular_values > tol * singular_values[0]))
    return vh.conj().T[:, rank:]


def _compute_bd_precoders(channels: np.ndarray, streams_per_user: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the paper's block-diagonalization construction for multiple
    streams per user: null the aggregate channels of all other users, then
    diagonalize the projected desired channel.
    """
    user_channels = _global_user_channels(channels)
    num_users = len(user_channels)
    total_antennas = user_channels[0].shape[1]
    beam_matrices = np.zeros((total_antennas, num_users, streams_per_user), dtype=np.complex128)
    stream_gains = np.zeros((num_users, streams_per_user), dtype=float)

    for user in range(num_users):
        interference_stack = [user_channels[idx] for idx in range(num_users) if idx != user]
        interference_matrix = (
            np.concatenate(interference_stack, axis=0)
            if interference_stack
            else np.zeros((0, total_antennas), dtype=np.complex128)
        )
        null_basis = _nullspace_basis(interference_matrix)
        if null_basis.shape[1] == 0:
            continue

        projected_channel = user_channels[user] @ null_basis
        _, singular_values, vh = np.linalg.svd(projected_channel, full_matrices=False)
        stream_count = min(streams_per_user, singular_values.size, null_basis.shape[1])
        if stream_count == 0:
            continue

        right_vectors = vh.conj().T[:, :stream_count]
        precoder = null_basis @ right_vectors
        beam_matrices[:, user, :stream_count] = precoder[:, :stream_count]
        stream_gains[user, :stream_count] = singular_values[:stream_count] ** 2

    return beam_matrices, stream_gains


def _power_allocation_objective(
    stream_weights: np.ndarray,
    stream_gains: np.ndarray,
    powers: np.ndarray,
    noise_power: float,
) -> float:
    snr = stream_gains * powers / max(noise_power, 1e-12)
    return float(np.dot(stream_weights, np.log2(1.0 + snr)))


def _power_from_prices(
    stream_prices: np.ndarray,
    stream_weights: np.ndarray,
    stream_gains: np.ndarray,
    noise_power: float,
) -> np.ndarray:
    safe_prices = np.maximum(stream_prices, 1e-12)
    safe_gains = np.maximum(stream_gains, 1e-12)
    return np.maximum(stream_weights / (np.log(2.0) * safe_prices) - noise_power / safe_gains, 0.0)


def _solve_dual_power_allocation(
    stream_weights: np.ndarray,
    stream_gains: np.ndarray,
    antenna_coeffs: np.ndarray,
    per_antenna_limits: np.ndarray,
    noise_power: float,
    *,
    max_iters: int,
    base_step: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    def dual_value_and_gradient(dual_prices: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        stream_prices = antenna_coeffs.T @ dual_prices
        local_powers = _power_from_prices(stream_prices, stream_weights, stream_gains, noise_power)
        utilities = stream_weights * np.log2(1.0 + (stream_gains * local_powers / max(noise_power, 1e-12)))
        dual_value = float(np.sum(utilities - stream_prices * local_powers) + np.dot(dual_prices, per_antenna_limits))
        gradient = per_antenna_limits - antenna_coeffs @ local_powers
        return dual_value, gradient, local_powers

    num_antennas = per_antenna_limits.size
    dual_prices = np.full(num_antennas, 1e-3, dtype=float)
    powers = np.zeros(stream_weights.size, dtype=float)
    best_feasible_powers = powers.copy()
    best_feasible_objective = -np.inf
    history: list[dict] = []

    warmup_iters = min(max_iters, 200)
    for iteration in range(1, warmup_iters + 1):
        _, _, updated_powers = dual_value_and_gradient(dual_prices)
        used_power = antenna_coeffs @ updated_powers
        violations = used_power - per_antenna_limits
        objective = _power_allocation_objective(stream_weights, stream_gains, updated_powers, noise_power)

        if np.all(violations <= 1e-8) and objective > best_feasible_objective:
            best_feasible_powers = updated_powers.copy()
            best_feasible_objective = objective

        relative_change = np.linalg.norm(updated_powers - powers) / max(np.linalg.norm(powers), 1e-12)
        history.append(
            {
                "iteration": iteration,
                "objective": objective,
                "relative_change": float(relative_change),
                "max_violation": float(np.max(violations)),
                "max_price": float(np.max(dual_prices)),
            }
        )

        step = base_step / np.sqrt(iteration)
        dual_prices = np.maximum(0.0, dual_prices + step * violations)
        powers = updated_powers

    def scipy_objective(dual_prices: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient, _ = dual_value_and_gradient(dual_prices)
        return value, gradient

    optimization = minimize(
        scipy_objective,
        np.maximum(dual_prices, 1e-9),
        method="L-BFGS-B",
        jac=True,
        bounds=Bounds(0.0, np.inf),
        options={"maxiter": max(100, max_iters), "ftol": 1e-12, "gtol": 1e-9},
    )

    if optimization.success:
        dual_prices = np.maximum(optimization.x, 0.0)
        _, _, powers = dual_value_and_gradient(dual_prices)
        used_power = antenna_coeffs @ powers
        violations = used_power - per_antenna_limits
        objective = _power_allocation_objective(stream_weights, stream_gains, powers, noise_power)
        history.append(
            {
                "iteration": warmup_iters + int(optimization.nit),
                "objective": objective,
                "relative_change": float(
                    np.linalg.norm(powers - best_feasible_powers) / max(np.linalg.norm(best_feasible_powers), 1e-12)
                ),
                "max_violation": float(np.max(violations)),
                "max_price": float(np.max(dual_prices)),
            }
        )
        if np.all(violations <= 1e-8) and objective > best_feasible_objective:
            best_feasible_powers = powers.copy()
            best_feasible_objective = objective

    final_used_power = antenna_coeffs @ powers
    if np.any(final_used_power > per_antenna_limits + 1e-9):
        scaling = float(np.min(per_antenna_limits / np.maximum(final_used_power, 1e-12)))
        powers = powers * min(scaling, 1.0)
    final_objective = _power_allocation_objective(stream_weights, stream_gains, powers, noise_power)
    if best_feasible_objective > final_objective:
        powers = best_feasible_powers

    return powers, dual_prices, history


def _assemble_beamformers(
    beam_matrices: np.ndarray,
    powers: np.ndarray,
    num_bs: int,
    antenna_count: int,
    num_users: int,
    streams_per_user: int,
) -> np.ndarray:
    weighted = beam_matrices.reshape(num_bs * antenna_count, num_users * streams_per_user) * np.sqrt(
        np.maximum(powers, 0.0)
    )[None, :]
    weighted = weighted.reshape(num_bs * antenna_count, num_users, streams_per_user)

    beamformers = np.zeros((num_bs, num_users, antenna_count, streams_per_user), dtype=np.complex128)
    for bs_idx in range(num_bs):
        start = bs_idx * antenna_count
        stop = start + antenna_count
        beamformers[bs_idx, :, :, :] = np.transpose(weighted[start:stop, :, :], (1, 0, 2))
    return beamformers


def run_global_benchmark(network: dict, config) -> dict:
    channels = network["channels"]
    weights = network["weights"]
    power_limits = network["power_limits"]
    noise_power = network["noise_power"]
    num_bs = network["num_bs"]
    num_users = network["num_users"]
    antenna_count = network["antenna_count"]
    streams_per_user = network["streams_per_user"]

    start_time = perf_counter()
    beam_matrices, stream_gains_matrix = _compute_bd_precoders(channels, streams_per_user)
    stream_weights = np.repeat(weights, streams_per_user)
    stream_gains = stream_gains_matrix.reshape(num_users * streams_per_user)
    antenna_coeffs = np.abs(beam_matrices.reshape(num_bs * antenna_count, num_users * streams_per_user)) ** 2
    per_antenna_limits = np.repeat(power_limits / antenna_count, antenna_count)

    dual_iters = max(250, 20 * config.max_iters)
    dual_step = max(0.02, config.lambda_step * np.mean(per_antenna_limits))
    powers, dual_prices, history = _solve_dual_power_allocation(
        stream_weights,
        stream_gains,
        antenna_coeffs,
        per_antenna_limits,
        noise_power,
        max_iters=dual_iters,
        base_step=dual_step,
    )

    beamformers = _assemble_beamformers(beam_matrices, powers, num_bs, antenna_count, num_users, streams_per_user)
    runtime_sec = perf_counter() - start_time
    final_metrics = compute_user_metrics(beamformers, channels, network["user_bs"], noise_power, weights)
    signaling_messages = estimate_signaling_messages(
        [np.arange(num_bs, dtype=int) for _ in range(num_bs)],
        [np.arange(num_users, dtype=int) for _ in range(num_bs)],
        len(history),
    )

    return {
        "label": "global-bd-dual",
        "rho": float("inf"),
        "beamformers": beamformers,
        "objective": final_metrics["objective"],
        "sinr": final_metrics["sinr"],
        "rates": final_metrics["rates"],
        "iterations": len(history),
        "runtime_sec": runtime_sec,
        "signaling_messages": signaling_messages,
        "history": history,
        "dual_prices": dual_prices,
        "stream_gains": stream_gains_matrix,
        "powers": powers.reshape(num_users, streams_per_user),
        "per_antenna_limits": per_antenna_limits,
    }


def run_full_neighborhood_reference(network: dict, config) -> dict:
    return run_ldbpa(
        network,
        rho=float("inf"),
        max_iters=config.max_iters,
        tol=config.tol,
        lambda_step=config.lambda_step,
        price_step=0.0,
        interference_budget_scale=config.interference_budget_scale,
        label="full-neighborhood",
    )
