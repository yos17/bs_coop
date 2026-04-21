from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT_DIR / "paper"
RESULTS_DIR = ROOT_DIR / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
FIG_RESULTS_DIR = RESULTS_DIR / "figs"
PAPER_FIG_DIR = PAPER_DIR / "figs"


@dataclass(frozen=True)
class SimulationConfig:
    antenna_count: int = 4
    users_per_bs: int = 2
    runtime_users_per_bs: int = 1
    ring_radius: int = 1
    monte_carlo_trials: int = 8
    runtime_trials: int = 3
    locality_trials: int = 6
    alpha_sweep_trials: int = 4
    max_iters: int = 40
    tol: float = 1e-4
    noise_power: float = 1e-2
    bs_power: float = 1.0
    inter_site_distance: float = 1.0
    user_radius: float = 0.45
    min_user_radius: float = 0.10
    path_loss_exponent: float = 3.6
    lambda_step: float = 0.20
    price_step: float = 0.02
    interference_budget_scale: float = 0.10
    seed: int = 7
    rho_values: tuple[float, ...] = (1.0, 2.0, 3.0)
    scaling_rings: tuple[int, ...] = (1, 2, 3)
    locality_rhos: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
    locality_ring_radius: int = 3
    locality_users_per_bs: int = 2
    alpha_sweep_values: tuple[float, ...] = (3.0, 3.6, 4.2)


DEFAULT_CONFIG = SimulationConfig()
