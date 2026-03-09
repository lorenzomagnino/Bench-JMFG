<div align="center">

<img src="favicon.svg" width="90" alt="Bench-MFG icon"/>

# Bench-MFG

**A benchmark suite for Mean Field Game algorithms**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-purple.svg)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/badge/ruff-⚡-gold.svg)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://github.com/pre-commit/pre-commit)
[![arXiv](https://img.shields.io/badge/arXiv-2602.12517-b31b1b.svg)](https://arxiv.org/pdf/2602.12517)

</div>

---

## Overview

Bench-MFG provides a unified framework for benchmarking algorithms on Mean Field Game (MFG) environments. It supports multiple solvers configured through [Hydra](https://hydra.cc) and accelerated with [JAX](https://github.com/google/jax).

### Environments

<table>
<thead>
<tr style="background-color: #4A90E2;">
<th style="padding: 8px; text-align: left; color: white;"><strong>Environment</strong></th>
<th style="padding: 8px; text-align: left; color: white;"><strong>Variants</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="padding: 8px;"><strong>No Interaction</strong></td>
<td style="padding: 8px;">• Move Forward</td>
</tr>
<tr>
<td style="padding: 8px;"><strong>Contractive Game</strong></td>
<td style="padding: 8px;">• Coordination Game</td>
</tr>
<tr>
<td style="padding: 8px;"><strong>Lasry-Lions Game</strong></td>
<td style="padding: 8px;">• Beach Bar Problem<br>• <em>(anti)</em> Two Beach Bars</td>
</tr>
<tr>
<td style="padding: 8px;"><strong>Potential Game</strong></td>
<td style="padding: 8px;">• Four Room Exploration<br>• <em>(anti)</em> RockPaperScissor</td>
</tr>
<tr>
<td style="padding: 8px;"><strong>Dynamics-Coupled Game</strong></td>
<td style="padding: 8px;">• SIS Epidemic<br>• Kinetic Congestion</td>
</tr>
<tr>
<td style="padding: 8px;"><strong>MF Garnet</strong> <em>(novel!)</em></td>
<td style="padding: 8px;">• Random Instances</td>
</tr>
</tbody>
</table>

### Algorithms

<table>
<thead>
<tr style="background-color: #FF8C42;">
<th style="padding: 8px; text-align: left; color: white;"><strong>Category</strong></th>
<th style="padding: 8px; text-align: left; color: white;"><strong>Algorithms</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="padding: 8px;"><strong>BR-based Fixed Point</strong></td>
<td style="padding: 8px;">• Fixed Point<br>• Damped Fixed Point<br>• Fictitious Play</td>
</tr>
<tr>
<td style="padding: 8px;"><strong>Policy-Eval. Based</strong></td>
<td style="padding: 8px;">• Policy Iteration (PI)<br>• Smoothed PI<br>• Boltzmann PI<br>• Online Mirror Descent</td>
</tr>
<tr>
<td style="padding: 8px;"><strong>Exploitability Min.</strong></td>
<td style="padding: 8px;">• MF-PSO <em>(novel!)</em></td>
</tr>
</tbody>
</table>

**Framework Features:** ✓ Hydra · ✓ JAX & Python · ✓ Log, Save and Plot

---

## Quick Start

```bash
# Create and activate environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### Run an experiment

Select your algorithm and environment in `conf/defaults.yaml`, then:

```bash
python main.py
```

For detailed instructions on batch runs see [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Configuration

All configuration lives under `conf/` and is managed by Hydra:

| File / Folder | Purpose |
|---|---|
| `conf/defaults.yaml` | Top-level defaults |
| `conf/algorithm/` | Per-algorithm settings (pso, omd, pi, …) |
| `conf/environment/` | Environment configurations |
| `conf/experiment/` | Experiment overrides |
| `conf/logging/` | WandB logging settings |
| `conf/visualization/` | Plot settings |

---

## Repository Structure

```
Bench-MFG/
├── conf/                    # Hydra configuration files
│   ├── defaults.yaml
│   ├── algorithm/
│   ├── environment/
│   ├── experiment/
│   ├── logging/
│   └── visualization/
├── envs/                    # MFG environments
│   ├── mf_garnet/           # MF Garnet (novel)
│   ├── four_rooms_obstacles/
│   ├── lasry_lions_chain/
│   ├── contraction_game/
│   ├── kinetic_congestion/
│   ├── sis_epidemic/
│   └── ...
├── learner/                 # Solver implementations
│   ├── jax/                 # JAX-accelerated solvers
│   └── python/              # Pure-Python solvers
├── utility/                 # Shared utilities
│   ├── create_environment.py
│   ├── create_solver.py
│   ├── run_training.py
│   ├── save_results.py
│   ├── wandb_logger.py
│   └── MFGPlots.py
├── outputs/                 # Experiment results
├── scripts/                 # Helper shell scripts
├── main.py                  # Entry point
└── pyproject.toml           # Project dependencies
```

---

## Outputs

Results are written to `outputs/YYYY-MM-DD/<Env>/<Algorithm>/<Experiment>/`:

| File | Contents |
|---|---|
| `*_results.npz` | Policy, mean field, exploitabilities |
| `mfg_experiment.log` | Full execution log |
