"""Filesystem discovery and label utilities for experiment results."""

import contextlib
from pathlib import Path
from typing import Any

import numpy as np
from utility.plot_loaders import load_exploitabilities, load_runtime
import yaml

ALGORITHMS = [
    "PSO",
    "OMD",
    "PI_smooth_policy_iteration",
    "PI_boltzmann_policy_iteration",
    "DampedFP_damped",
]


def version_to_algorithm_dir(version_withhyper: str) -> str:
    """Map version_withhyper string to its algorithm directory name.

    Example: "pso_sweep_temp0p20" → "PSO"
    """
    v = version_withhyper.lower()
    if "pso_sweep" in v:
        return "PSO"
    elif "omd_sweep" in v:
        return "OMD"
    elif "smooth_pi_sweep" in v or "smooth_policy_iteration" in v:
        return "PI_smooth_policy_iteration"
    elif "boltzmann_pi_sweep" in v or "boltzmann_policy_iteration" in v:
        return "PI_boltzmann_policy_iteration"
    elif "policy_iteration_sweep" in v:
        return "PI_policy_iteration"
    elif "fplay_sweep" in v or "fictitious" in v:
        return "DampedFP_fictitious_play"
    elif "pure_fp_sweep" in v or v.startswith("pure"):
        return "DampedFP_pure"
    elif "damped_sweep" in v:
        return "DampedFP_damped"
    else:
        parts = version_withhyper.split("_")
        return parts[0].capitalize() if parts else version_withhyper


def version_to_algorithm_name(version_withhyper: str) -> str:
    """Convert version_withhyper to a short human-readable algorithm name.

    Example: "damped_sweep_damped0p10" → "Damped FP"
    """
    v = version_withhyper.lower()
    if "pure_fp" in v or v.startswith("pure"):
        return "Fixed Point (FP)"
    elif "fplay" in v or "fictitious" in v:
        return "Fictitious Play"
    elif "smooth_policy_iteration" in v or "smooth_pi_sweep" in v:
        return "Smooth PI"
    elif "boltzmann_policy_iteration" in v or "boltzmann_pi_sweep" in v:
        return "Boltzmann PI"
    elif "omd_sweep" in v:
        return "OMD"
    elif "pso_sweep" in v:
        return "PSO"
    elif "damped_sweep_damped" in v:
        return "Damped FP"
    elif "policy_iteration_sweep" in v:
        return "PI"
    else:
        return version_withhyper


def extract_hyperparameters(version_withhyper: str) -> str:
    """Extract hyperparameters from a version name as a compact LaTeX-ready string.

    Example: "pso_sweep_temp0p20_w0p30_c10p30_c21p20" → "$\\tau$, w, c1, c2 = 0.2, 0.30, 0.30, 1.20"
    """
    parts = version_withhyper.split("_")
    temp_value = damped_value = lr_value = w_value = c1_value = c2_value = None

    for part in parts:
        if part.startswith("damped") and "damped" in part:
            with contextlib.suppress(ValueError):
                damped_value = float(part.replace("damped", "").replace("p", "."))
        elif part.startswith("temp"):
            with contextlib.suppress(ValueError):
                temp_value = float(part.replace("temp", "").replace("p", "."))
        elif part.startswith("lr"):
            with contextlib.suppress(ValueError):
                lr_value = float(part.replace("lr", "").replace("p", "."))
        elif part.startswith("w") and len(part) > 1:
            with contextlib.suppress(ValueError):
                w_value = float(part.replace("w", "").replace("p", "."))
        elif part.startswith("c1"):
            with contextlib.suppress(ValueError):
                c1_value = float(part.replace("c1", "").replace("p", "."))
        elif part.startswith("c2"):
            with contextlib.suppress(ValueError):
                c2_value = float(part.replace("c2", "").replace("p", "."))

    param_names: list[str] = []
    param_values: list[str] = []

    if temp_value is not None:
        param_names.append("$\\tau$")
        param_values.append(f"{temp_value:.1f}")
    if damped_value is not None:
        param_names.append("$\\lambda$")
        param_values.append(f"{damped_value:.1f}")
    if lr_value is not None:
        param_names.append("lr")
        param_values.append(f"{lr_value:.4f}")
    if w_value is not None:
        param_names.append("w")
        param_values.append(f"{w_value:.2f}")
    if c1_value is not None:
        param_names.append("c1")
        param_values.append(f"{c1_value:.2f}")
    if c2_value is not None:
        param_names.append("c2")
        param_values.append(f"{c2_value:.2f}")

    if param_names and param_values:
        return f"{', '.join(param_names)} = {', '.join(param_values)}"
    return ""


def group_exploitabilities_by_seed(
    environment: str,
    outputs_dir: str | Path = "outputs",
) -> dict[str, dict[str, Any]]:
    """Group exploitabilities by version and seed.

    Scans outputs/{environment}/**/exploitabilities.npz and organises results as:
        {version_withhyper: {"groups": [[exp_seed0_run0, ...], [exp_seed1_run0, ...]], "seed_names": [...]}}
    """
    outputs_dir = Path(outputs_dir)
    env_dir = outputs_dir / environment

    if not env_dir.exists():
        raise FileNotFoundError(f"Environment directory not found: {env_dir}")

    version_groups: dict[str, dict[str, list[np.ndarray]]] = {}

    for algorithm_dir in env_dir.iterdir():
        if not algorithm_dir.is_dir():
            continue
        for seed_dir in algorithm_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed_name = seed_dir.name
            for version_dir in seed_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                version_name = version_dir.name
                version_groups.setdefault(version_name, {})
                version_groups[version_name].setdefault(seed_name, [])
                for timestamp_dir in version_dir.iterdir():
                    if not timestamp_dir.is_dir():
                        continue
                    exp_path = timestamp_dir / "exploitabilities.npz"
                    if exp_path.exists():
                        try:
                            version_groups[version_name][seed_name].append(
                                load_exploitabilities(exp_path)
                            )
                        except Exception as e:
                            print(f"Warning: Failed to load {exp_path}: {e}")

    result: dict[str, dict[str, Any]] = {}
    for version_name, seed_dict in version_groups.items():
        sorted_seeds = sorted(seed_dict.keys())
        result[version_name] = {
            "groups": [seed_dict[s] for s in sorted_seeds],
            "seed_names": sorted_seeds,
        }
    return result


def group_runtimes_by_seed(
    environment: str,
    outputs_dir: str | Path = "outputs",
) -> dict[str, dict[str, Any]]:
    """Group wall-clock runtimes by version and seed.

    Returns:
        {version_withhyper: {"runtimes": [float, ...], "seed_names": [str, ...]}}
    """
    outputs_dir = Path(outputs_dir)
    env_dir = outputs_dir / environment

    if not env_dir.exists():
        raise FileNotFoundError(f"Environment directory not found: {env_dir}")

    version_data: dict[str, dict[str, list]] = {}

    for algorithm_dir in env_dir.iterdir():
        if not algorithm_dir.is_dir():
            continue
        for seed_dir in algorithm_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed_name = seed_dir.name
            for version_dir in seed_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                version_name = version_dir.name
                version_data.setdefault(version_name, {"runtimes": [], "seeds": []})
                for timestamp_dir in version_dir.iterdir():
                    if not timestamp_dir.is_dir():
                        continue
                    metrics_path = timestamp_dir / "metrics.npz"
                    if metrics_path.exists():
                        try:
                            rt = load_runtime(metrics_path)
                            version_data[version_name]["runtimes"].append(rt)
                            version_data[version_name]["seeds"].append(seed_name)
                        except Exception as e:
                            print(f"Warning: Failed to load {metrics_path}: {e}")

    return {
        v: {"runtimes": d["runtimes"], "seed_names": sorted(set(d["seeds"]))}
        for v, d in version_data.items()
    }


def get_versions_for_algorithm(
    environment: str,
    algorithm: str,
    outputs_dir: str | Path = "outputs",
) -> list[str]:
    """Return all version_withhyper names that have exploitabilities data for an algorithm."""
    outputs_dir = Path(outputs_dir)
    algorithm_dir = outputs_dir / environment / algorithm

    if not algorithm_dir.exists():
        raise FileNotFoundError(f"Algorithm directory not found: {algorithm_dir}")

    versions_set: set[str] = set()
    for seed_dir in algorithm_dir.iterdir():
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        for version_dir in seed_dir.iterdir():
            if not version_dir.is_dir():
                continue
            has_data = any(
                (ts_dir / "exploitabilities.npz").exists()
                for ts_dir in version_dir.iterdir()
                if ts_dir.is_dir()
            )
            if has_data:
                versions_set.add(version_dir.name)

    return sorted(versions_set)


def get_versions_for_comparison(
    environment: str,
    fixed_versions: list[str] | None = None,
    results_dir: str | Path | None = None,
) -> list[str]:
    """Combine fixed versions with best-rank-1 versions from *_best_models.yaml files.

    Run plot_exploitability_by_algorithm (in plot_sweep.py) for each algorithm first
    to generate the YAML files this function reads.
    """
    if fixed_versions is None:
        fixed_versions = [
            "pure_fp_sweep",
            "fplay_sweep",
            "policy_iteration_sweep_temp0p20",
        ]

    if results_dir is None:
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results" / environment
    else:
        results_dir = Path(results_dir) / environment

    best_from_yaml: list[str] = []
    if results_dir.exists():
        for yaml_file in results_dir.glob("*_best_models.yaml"):
            try:
                with open(yaml_file) as f:
                    yaml_data = yaml.safe_load(f)
                if (
                    yaml_data
                    and "best_models" in yaml_data
                    and yaml_data["best_models"]
                ):
                    rank1 = yaml_data["best_models"][0]
                    if rank1.get("rank") == 1:
                        best_from_yaml.append(rank1["version"])
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")

    return list(dict.fromkeys(fixed_versions + best_from_yaml))


def get_four_rooms_walls(grid_dim: tuple) -> np.ndarray:
    """Generate a walls mask for FourRoomsAversion2D (0 = wall, 1 = free)."""
    n_rows, n_cols = grid_dim
    N_flat = n_rows * n_cols
    walls = np.ones(N_flat, dtype=int)
    mid_row, mid_col = n_rows // 2, n_cols // 2
    doors = {(2, 5), (8, 5), (5, 8), (5, 2)}

    for row in range(n_rows):
        if (row, mid_col) not in doors:
            idx = row * n_cols + mid_col
            if 0 <= idx < N_flat:
                walls[idx] = 0

    for col in range(n_cols):
        if (mid_row, col) not in doors:
            idx = mid_row * n_cols + col
            if 0 <= idx < N_flat:
                walls[idx] = 0

    return walls
