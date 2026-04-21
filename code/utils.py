from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.linalg import solve


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def served_user_lists(user_bs: np.ndarray, num_bs: int) -> list[np.ndarray]:
    return [np.flatnonzero(user_bs == b) for b in range(num_bs)]


def neighborhoods(bs_positions: np.ndarray, rho: float) -> list[np.ndarray]:
    num_bs = bs_positions.shape[0]
    if np.isinf(rho):
        return [np.arange(num_bs, dtype=int) for _ in range(num_bs)]
    dist = np.linalg.norm(bs_positions[:, None, :] - bs_positions[None, :, :], axis=2)
    return [np.flatnonzero(dist[b] <= rho + 1e-12) for b in range(num_bs)]


def initialize_beamformers(channels: np.ndarray, user_bs: np.ndarray, power_limits: np.ndarray, streams_per_user: int) -> np.ndarray:
    num_bs = power_limits.size
    num_users = user_bs.size
    antenna_count = channels.shape[3]
    beamformers = np.zeros((num_bs, num_users, antenna_count, streams_per_user), dtype=np.complex128)

    for b in range(num_bs):
        served_users = np.flatnonzero(user_bs == b)
        if served_users.size == 0:
            continue
        for user in served_users:
            direct_channel = channels[user, b, :, :]
            _, _, vh = np.linalg.svd(direct_channel, full_matrices=False)
            right_vectors = vh.conj().T[:, :streams_per_user]
            beamformers[b, user, :, : right_vectors.shape[1]] = right_vectors

        power = per_bs_power(beamformers)[b]
        if power > 0.0:
            beamformers[b, served_users, :, :] *= np.sqrt(power_limits[b] / power)

    return beamformers


def compute_stream_responses(beamformers: np.ndarray, channels: np.ndarray) -> np.ndarray:
    return np.einsum("kbrt,bjtd->kjrd", channels, beamformers, optimize=True)


def compute_user_metrics(
    beamformers: np.ndarray,
    channels: np.ndarray,
    user_bs: np.ndarray,
    noise_power: float,
    weights: np.ndarray,
) -> dict:
    responses = compute_stream_responses(beamformers, channels)
    num_users = user_bs.size
    receive_antenna_count = channels.shape[2]
    streams_per_user = beamformers.shape[3]

    identity_r = np.eye(receive_antenna_count, dtype=np.complex128)
    identity_d = np.eye(streams_per_user, dtype=np.complex128)

    total_covariances = noise_power * identity_r[None, :, :] + np.einsum(
        "kjrd,kjsd->krs",
        responses,
        np.conjugate(responses),
        optimize=True,
    )
    desired_responses = responses[np.arange(num_users), np.arange(num_users), :, :]
    interference_covariances = total_covariances - np.einsum(
        "krd,ksd->krs",
        desired_responses,
        np.conjugate(desired_responses),
        optimize=True,
    )

    receivers = np.zeros((num_users, receive_antenna_count, streams_per_user), dtype=np.complex128)
    mse_matrices = np.zeros((num_users, streams_per_user, streams_per_user), dtype=np.complex128)
    rates = np.zeros(num_users, dtype=float)

    for user in range(num_users):
        regularized_total = total_covariances[user] + 1e-9 * identity_r
        receivers[user] = solve(regularized_total, desired_responses[user], assume_a="her")

        regularized_interference = interference_covariances[user] + 1e-9 * identity_r
        gain_matrix = desired_responses[user].conj().T @ solve(
            regularized_interference,
            desired_responses[user],
            assume_a="her",
        )
        sinr_matrix = identity_d + 0.5 * (gain_matrix + gain_matrix.conj().T)
        eigenvalues = np.maximum(np.linalg.eigvalsh(sinr_matrix).real, 1e-12)
        rates[user] = float(np.sum(np.log2(eigenvalues)))
        mse_matrices[user] = np.linalg.inv(sinr_matrix + 1e-9 * identity_d)

    effective = np.einsum("krp,kjrq->kjpq", np.conjugate(receivers), responses, optimize=True)
    desired_matrices = effective[np.arange(num_users), np.arange(num_users), :, :]
    diagonal_terms = np.diagonal(desired_matrices, axis1=1, axis2=2)
    desired_power = np.sum(np.abs(diagonal_terms) ** 2, axis=1)
    self_leakage = np.sum(np.abs(desired_matrices) ** 2, axis=(1, 2)) - desired_power
    inter_user_leakage = np.sum(np.abs(effective) ** 2, axis=(1, 2, 3)) - np.sum(np.abs(desired_matrices) ** 2, axis=(1, 2))
    interference_power = self_leakage + inter_user_leakage
    noise_after_combining = noise_power * np.sum(np.abs(receivers) ** 2, axis=(1, 2))
    total_power = desired_power + interference_power + noise_after_combining
    sinr = np.maximum(np.exp2(rates) - 1.0, 0.0)

    return {
        "responses": responses,
        "effective": effective,
        "desired_matrices": desired_matrices,
        "desired_power": desired_power,
        "interference_power": interference_power,
        "noise_after_combining": noise_after_combining,
        "total_power": total_power,
        "sinr": sinr,
        "rates": rates,
        "objective": float(np.dot(weights, rates)),
        "receivers": receivers,
        "mse_matrices": mse_matrices,
    }


def per_bs_power(beamformers: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(beamformers) ** 2, axis=(1, 2, 3))


def far_field_leakage(
    responses: np.ndarray,
    user_bs: np.ndarray,
    neighborhood_sets: list[np.ndarray],
) -> np.ndarray:
    num_users = user_bs.size
    leakage = np.zeros(num_users, dtype=float)
    response_power = np.sum(np.abs(responses) ** 2, axis=(2, 3))
    for user in range(num_users):
        allowed_bs = set(neighborhood_sets[user_bs[user]].tolist())
        mask = np.array([user_bs[other] not in allowed_bs for other in range(num_users)], dtype=bool)
        mask[user] = False
        leakage[user] = float(response_power[user, mask].sum())
    return leakage


def estimate_signaling_messages(
    neighborhood_sets: list[np.ndarray],
    served_users: list[np.ndarray],
    iterations: int,
) -> float:
    directed_edges = sum(max(len(neighbors) - 1, 0) for neighbors in neighborhood_sets)
    avg_users = float(np.mean([users.size for users in served_users])) if served_users else 0.0
    scalars_per_exchange = 3.0 + avg_users
    return float(iterations) * directed_edges * scalars_per_exchange


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
