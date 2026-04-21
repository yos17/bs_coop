from __future__ import annotations

import math

import numpy as np


def hex_axial_coordinates(ring_radius: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for q in range(-ring_radius, ring_radius + 1):
        r_min = max(-ring_radius, -q - ring_radius)
        r_max = min(ring_radius, -q + ring_radius)
        for r in range(r_min, r_max + 1):
            coords.append((q, r))
    coords.sort(key=lambda item: (abs(item[0]) + abs(item[1]) + abs(item[0] + item[1]), item[0], item[1]))
    return coords


def axial_to_cartesian(coords: list[tuple[int, int]], inter_site_distance: float) -> np.ndarray:
    positions = np.zeros((len(coords), 2), dtype=float)
    for idx, (q, r) in enumerate(coords):
        x = inter_site_distance * (q + 0.5 * r)
        y = inter_site_distance * (math.sqrt(3.0) / 2.0) * r
        positions[idx] = np.array([x, y], dtype=float)
    return positions


def generate_hex_layout(ring_radius: int, inter_site_distance: float) -> np.ndarray:
    coords = hex_axial_coordinates(ring_radius)
    return axial_to_cartesian(coords, inter_site_distance)


def sample_user_positions(
    bs_positions: np.ndarray,
    users_per_bs: int,
    user_radius: float,
    min_radius: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    user_positions = []
    user_bs = []
    min_radius_sq = min_radius * min_radius
    max_radius_sq = user_radius * user_radius
    for bs_idx, bs_pos in enumerate(bs_positions):
        for _ in range(users_per_bs):
            theta = rng.uniform(0.0, 2.0 * math.pi)
            radius = math.sqrt(rng.uniform(min_radius_sq, max_radius_sq))
            offset = radius * np.array([math.cos(theta), math.sin(theta)], dtype=float)
            user_positions.append(bs_pos + offset)
            user_bs.append(bs_idx)
    return np.asarray(user_positions, dtype=float), np.asarray(user_bs, dtype=int)


def generate_channels(
    user_positions: np.ndarray,
    bs_positions: np.ndarray,
    antenna_count: int,
    path_loss_exponent: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(user_positions[:, None, :] - bs_positions[None, :, :], axis=2)
    path_loss = np.power(1.0 + distances, -path_loss_exponent)
    fading = (
        rng.standard_normal((user_positions.shape[0], bs_positions.shape[0], antenna_count))
        + 1j * rng.standard_normal((user_positions.shape[0], bs_positions.shape[0], antenna_count))
    ) / math.sqrt(2.0)
    channels = np.sqrt(path_loss)[..., None] * fading
    return channels.astype(np.complex128), path_loss


def build_network(config, ring_radius: int | None = None, users_per_bs: int | None = None, seed: int | None = None) -> dict:
    ring = config.ring_radius if ring_radius is None else ring_radius
    users = config.users_per_bs if users_per_bs is None else users_per_bs
    rng = np.random.default_rng(config.seed if seed is None else seed)

    bs_positions = generate_hex_layout(ring, config.inter_site_distance)
    user_positions, user_bs = sample_user_positions(
        bs_positions=bs_positions,
        users_per_bs=users,
        user_radius=config.user_radius,
        min_radius=config.min_user_radius,
        rng=rng,
    )
    channels, path_loss = generate_channels(
        user_positions=user_positions,
        bs_positions=bs_positions,
        antenna_count=config.antenna_count,
        path_loss_exponent=config.path_loss_exponent,
        rng=rng,
    )

    num_bs = bs_positions.shape[0]
    num_users = user_positions.shape[0]
    return {
        "bs_positions": bs_positions,
        "user_positions": user_positions,
        "user_bs": user_bs,
        "channels": channels,
        "path_loss": path_loss,
        "weights": np.ones(num_users, dtype=float),
        "power_limits": np.full(num_bs, config.bs_power, dtype=float),
        "noise_power": float(config.noise_power),
        "antenna_count": int(config.antenna_count),
        "num_bs": int(num_bs),
        "num_users": int(num_users),
        "ring_radius": int(ring),
        "users_per_bs": int(users),
    }
