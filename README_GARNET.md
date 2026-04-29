# MF-Garnet Experiment Scripts

This directory contains scripts for running and aggregating MF-Garnet experiments.

## MF-Garnet Game Generation

A **MF-Garnet** (Mean-Field GARNET) game is a randomly generated Mean Field Game (MFG) instance. Each run uses a different `seed` to produce a distinct game. Here's the construction pipeline in `src/envs/mf_garnet/mf_garnet.py`:

### 1. Randomize coupling coefficients
From the seed, four coupling scalars are drawn from `Uniform[0,1]`:
- `cp`, `rho_p` — control how strongly the mean field influences **transitions**
- `cr`, `rho_r` — control how strongly the mean field influences **rewards**

### 2. Build the base transition kernel `P0` (sparse)
For each `(state, action)` pair, `branching_factor` (default: 5) next-states are sampled at random, and a probability vector over them is sampled from a **Dirichlet distribution**. This creates a sparse but randomized transition kernel.

### 3. Build the mean-field coupling tensor `C`
A tensor `C[s, a, :, :]` of shape `(N_states, N_actions, N_states, N_states)` is drawn from a **standard normal**. It modulates how the mean-field distribution `mu` bends transitions.

At runtime, the actual transition is:
- **Additive**: `intensity = cp * P0(s,a) + rho_p * (C[s,a] @ mu)`
- **Multiplicative**: `intensity = P0(s,a) * (cp + rho_p * (C[s,a] @ mu))`

then normalized to a valid probability.

### 4. Build the base reward `R0` and interaction matrix `M`
- `R0[s,a]` is drawn from `Normal(0, reward_scale)`
- `M[s,y]` is drawn from a normal matrix, then symmetrized (`potential` game) or anti-symmetrized (`cyclic` game)

The actual reward is:
- **Additive**: `cr * R0[s,a] + rho_r * (M[s] @ mu)`
- **Multiplicative**: `R0[s,a] * (cr + rho_r * (M[s] @ mu))`

## Overview

The MF-Garnet experiments follow this protocol:

1. **Fix class**: Choose `dynamics_structure` (additive/multiplicative) and `reward_structure` (additive/multiplicative)
2. **Fix model parameters**: Set `num_states`, `num_actions` (e.g., 5, 5)
3. **Fix algorithm hyperparameters**: Use fixed hyperparameters for each algorithm
4. **Repeat X times**: Each run pairs (garnet_seed_i, algo_seed_i) where:
   - `garnet_seed_i` varies from 0 to X-1 (different MFG instances)
   - `algo_seed_i` is a different algorithm seed for each instance
5. **Aggregate**:
   - For each algorithm: compute mean and std of final exploitability over X runs

This gives X runs per algorithm (instead of X×Y runs), making it much more lightweight.

## Script
**Usage:**
```bash
cd Bench-MFG
source .venv/bin/activate
bash scripts/run_garnet_omd.sh
```
Or we cna also use the general script commenting out the algorithm that we are not runnign
```bash
cd Bench-MFG
source .venv/bin/activate
bash scripts/run_garnet_experiments.sh
```

**Configuration:**
Edit the variables at the top of the script:
- `NUM_INSTANCES`: Number of different MFG instances/runs (X, default: 10)
- `ALGORITHM_SEEDS`: Array of algorithm seeds, one per instance (default: [42, 10, 111, 1032, 999, 1234, 5678, 9012, 3456, 7890])
- `NUM_STATES`, `NUM_ACTIONS`: Model parameters (default: 5, 5)
- `DYNAMICS_STRUCTURE`, `REWARD_STRUCTURE`: Class to fix (default: "additive", "multiplicative")
- Algorithm hyperparameters (PSO_TEMP, DAMPEDFP_CONSTANT, etc.)

**Note:** Make sure `ALGORITHM_SEEDS` has at least `NUM_INSTANCES` elements.

**Output:**
Results are saved in:
```
outputs/MFGarnet/<algorithm>/seed_<algo_seed_i>/garnet_<algorithm>_<dynamics>_<reward>/<timestamp>/
```

Each run produces one result file with the paired (garnet_seed, algo_seed).

### `garnet_results_table`

Script to aggregate results and compute statistics.

**Usage:**
```bash
cd Bench-MFG
source .venv/bin/activate
python utility/garnet_results_table.py
```

**Output:**
Prints a summary table with:
- Mean final exploitability for each algorithm (averaged over X instances)
- Standard deviation of final exploitability for each algorithm
- Number of instances processed

## Customization

To run experiments with different configurations:

1. **Change the class**: Edit `DYNAMICS_STRUCTURE` and `REWARD_STRUCTURE` in `run_garnet_experiments.sh`
2. **Change model size**: Edit `NUM_STATES` and `NUM_ACTIONS`
3. **Change number of instances**: Edit `NUM_INSTANCES` (and ensure `ALGORITHM_SEEDS` has enough elements)
4. **Change algorithm seeds**: Edit `ALGORITHM_SEEDS` array (one seed per instance)
5. **Change algorithm hyperparameters**: Edit the corresponding variables in the script
6. **Run only specific algorithms**: Comment out unwanted `run_algorithm` calls in the script
