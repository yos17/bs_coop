from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


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


def initialize_beamformers(channels: np.ndarray, user_bs: np.ndarray, power_limits: np.ndarray) -> np.ndarray:
    num_bs = power_limits.size
    num_users = user_bs.size
    antenna_count = channels.shape[2]
    beamformers = np.zeros((num_bs, num_users, antenna_count), dtype=np.complex128)
    for b in range(num_bs):
        served_users = np.flatnonzero(user_bs == b)
        if served_users.size == 0:
            continue
        for user in served_users:
            matched_filter = channels[user, b, :]
            norm_val = np.linalg.norm(matched_filter)
            if norm_val > 0.0:
                beamformers[b, user, :] = matched_filter / norm_val
        power = per_bs_power(beamformers)[b]
        if power > 0.0:
            beamformers[b, served_users, :] *= np.sqrt(power_limits[b] / power)
    return beamformers


def compute_effective_channels(beamformers: np.ndarray, channels: np.ndarray, user_bs: np.ndarray) -> np.ndarray:
    num_users = user_bs.size
    effective = np.zeros((num_users, num_users), dtype=np.complex128)
    for stream in range(num_users):
        bs_idx = user_bs[stream]
        effective[:, stream] = np.einsum(
            "km,m->k",
            np.conjugate(channels[:, bs_idx, :]),
            beamformers[bs_idx, stream, :],
        )
    return effective


def compute_user_metrics(
    beamformers: np.ndarray,
    channels: np.ndarray,
    user_bs: np.ndarray,
    noise_power: float,
    weights: np.ndarray,
) -> dict:
    effective = compute_effective_channels(beamformers, channels, user_bs)
    stream_power = np.abs(effective) ** 2
    desired_complex = np.diag(effective).copy()
    desired_power = np.abs(desired_complex) ** 2
    total_interference_power = stream_power.sum(axis=1) - desired_power
    total_power = desired_power + total_interference_power + noise_power
    sinr = desired_power / np.maximum(total_interference_power + noise_power, 1e-12)
    rates = np.log2(1.0 + sinr)
    return {
        "effective": effective,
        "desired_complex": desired_complex,
        "desired_power": desired_power,
        "interference_power": total_interference_power,
        "total_power": total_power,
        "sinr": sinr,
        "rates": rates,
        "objective": float(np.dot(weights, rates)),
    }


def per_bs_power(beamformers: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(beamformers) ** 2, axis=(1, 2))


def far_field_leakage(
    beamformers: np.ndarray,
    channels: np.ndarray,
    user_bs: np.ndarray,
    neighborhood_sets: list[np.ndarray],
) -> np.ndarray:
    effective = compute_effective_channels(beamformers, channels, user_bs)
    stream_power = np.abs(effective) ** 2
    num_users = user_bs.size
    leakage = np.zeros(num_users, dtype=float)
    for user in range(num_users):
        allowed_bs = set(neighborhood_sets[user_bs[user]].tolist())
        mask = np.array([user_bs[stream] not in allowed_bs for stream in range(num_users)], dtype=bool)
        mask[user] = False
        leakage[user] = float(stream_power[user, mask].sum())
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
