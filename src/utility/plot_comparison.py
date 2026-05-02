"""Compare best hyperparameter version of each algorithm for one environment.

Usage:
    PYTHONPATH=src python -m utility.plot_comparison <environment> [options]
    PYTHONPATH=src python src/utility/plot_comparison.py <environment> [options]

Prerequisite: run plot_sweep.py for each algorithm first — it writes
results/{environment}/{algorithm}_best_models.yaml which this script reads to
select the best hyperparameter version per algorithm.

Produces:
  - Exploitability comparison plot (mean ± std per algorithm)
  - Runtime box plot

Example:
    # Step 1: generate best_models.yaml for each algorithm
    for algo in PSO OMD DampedFP_damped PI_smooth_policy_iteration PI_boltzmann_policy_iteration; do
        PYTHONPATH=src python -m utility.plot_sweep MultipleEquilibriaGame $algo --log-scale
    done

    # Step 2: compare
    PYTHONPATH=src python -m utility.plot_comparison MultipleEquilibriaGame --log-scale
"""

import argparse
from pathlib import Path
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from conf.visualization.visualization_schema import ColorsConfig  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from utility.plot_discovery import get_versions_for_comparison  # noqa: E402
from utility.plot_primitives import (  # noqa: E402
    plot_exploitability_multiple_versions,
    plot_runtime_multiple_versions,
)


def plot_runtime_for_env(
    environment: str,
    outputs_dir: str | Path = "outputs",
    results_dir: str | Path | None = None,
    ylabel: str = "Wall-clock runtime (s)",
    return_fig: bool = False,
    fn: str | Path | None = None,
    colors: ColorsConfig | None = None,
    color_list: list[str] | None = None,
    label_format: str = "algorithm",
    legend_loc: str | None = None,
    show_legend: bool = False,
) -> Figure | None:
    """Plot runtime box chart for an environment using best hyperparameters per algorithm.

    Reads best versions from *_best_models.yaml (written by plot_sweep.py) and
    fixed non-sweep versions to build the version list, then delegates to
    plot_runtime_multiple_versions.
    """
    versions_withhyper = get_versions_for_comparison(
        environment=environment,
        results_dir=results_dir,
    )
    if len(versions_withhyper) == 0:
        raise ValueError(
            f"No versions found for environment '{environment}'. "
            "Run plot_sweep.py for each algorithm first to generate best_models.yaml files."
        )

    if fn is None:
        project_root = Path(__file__).parent.parent
        env_results_dir = project_root / "results" / environment
        env_results_dir.mkdir(parents=True, exist_ok=True)
        fn = env_results_dir / "runtime_best_versions.pdf"

    return plot_runtime_multiple_versions(
        environment=environment,
        versions_withhyper=versions_withhyper,
        outputs_dir=outputs_dir,
        ylabel=ylabel,
        return_fig=return_fig,
        fn=fn,
        colors=colors,
        color_list=color_list,
        label_format=label_format,
        legend_loc=legend_loc,
        show_legend=show_legend,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare best version of each algorithm for one environment."
    )
    parser.add_argument(
        "environment", type=str, help="Environment name (e.g. MultipleEquilibriaGame)"
    )
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument(
        "--plot-every-n",
        type=int,
        default=10,
        help="Subsample: plot every N-th iteration (default: 10)",
    )
    parser.add_argument(
        "--legend-loc",
        type=str,
        default="upper right",
        help="Legend location (default: upper right)",
    )
    parser.add_argument(
        "--color-list",
        type=str,
        nargs="+",
        default=[
            "#F86262",
            "#F0816A",
            "#7F11F5",
            "#0936C8",
            "#63B0F8",
            "#FA8FBF",
            "#703F62",
            "#97B9C3",
        ],
    )
    parser.add_argument(
        "--marker-list",
        type=str,
        nargs="+",
        default=["o", "s", "D", "^", "v", "p", "h", "*"],
    )
    parser.add_argument(
        "--skip-runtime", action="store_true", help="Skip the runtime box plot"
    )
    args = parser.parse_args()

    legend_location = args.legend_loc.replace("_", " ") if args.legend_loc else None

    versions = get_versions_for_comparison(environment=args.environment)
    if not versions:
        print(
            f"No versions found for '{args.environment}'. "
            "Run plot_sweep.py for each algorithm first."
        )
        sys.exit(1)

    print(f"Comparing {len(versions)} versions: {versions}")

    plot_exploitability_multiple_versions(
        environment=args.environment,
        versions_withhyper=versions,
        outputs_dir=args.outputs_dir,
        log_scale=args.log_scale,
        color_list=args.color_list,
        legend_loc=legend_location,
        plot_every_n=args.plot_every_n,
        marker_list=args.marker_list,
        label_format="algorithm",
    )

    if not args.skip_runtime:
        plot_runtime_for_env(
            environment=args.environment,
            outputs_dir=args.outputs_dir,
            label_format="algorithm",
        )
