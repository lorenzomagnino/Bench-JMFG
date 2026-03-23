# Device & Parallelism Report — 2026-03-23

## What Was Done

- **`src/conf/config_schema.py`** — Added `device: str = "cpu"` field to `MFGConfig`.
  Valid values are `"cpu"` and `"cuda"` (maps to the JAX GPU backend).

- **`src/envs/mfg_model_class_jit.py`** — Added:
  - `get_jax_device(device_str)`: resolves `"cuda"` → JAX GPU backend, any other value → JAX CPU backend, with a graceful fallback to CPU when the requested backend is unavailable.
  - `exploitability_batch_pmap(...)`: multi-device variant of `exploitability_batch_jax` that shards the particle batch across all available JAX devices via `jax.pmap`; pads the batch to a multiple of `n_devices` for even sharding; falls back to `vmap` on single-device setups.

- **`src/learner/jax/fp_jax.py`** — Added `jax_device` parameter to `DampedFP_jax.__init__`; added `_put(arr)` helper (`jax.device_put(arr, self.jax_device)`); replaced all `jnp.asarray()` boundary conversions with `self._put()`.

- **`src/learner/jax/omd_jax.py`** — Same pattern as above for `OMD_jax`.

- **`src/learner/jax/pi_jax.py`** — Same pattern for `PI_jax`.

- **`src/learner/jax/pso_jax.py`** — Same pattern for `PSO_jax`; additionally switched `_batch_exploitability` to call `exploitability_batch_pmap` so multi-device sharding activates automatically when multiple JAX devices are present.

- **`src/envs/mfg_model_class.py`** — Parallelised the two CPU-bound matrix-building loops in the pure-Python backend:
  - `_build_transition_matrix`: per-state rows dispatched to `ProcessPoolExecutor(max_workers=min(os.cpu_count(), S))`.
  - `_build_reward_matrix`: same pattern.
  Module-level helper functions `_compute_transition_row` and `_compute_reward_row` are defined at the top of the file so they are picklable by worker processes.

- **`src/utility/create_solver.py`** — Imported `get_jax_device`; resolved `jax_device = get_jax_device(cfg.device)` once in `create_solver()` and forwarded it to all four JAX solver constructors (`PSO_jax`, `DampedFP_jax`, `OMD_jax`, `PI_jax`).

## Problems Encountered

- **`jax.pmap` with callable-field dataclasses**: `EnvSpec` contains `Callable` fields (transition, reward) that cannot be passed as sharded pmap arguments. Resolved by capturing `spec` and `initial_mean_field` from the enclosing scope inside the pmap body, making them static broadcast constants rather than sharded inputs.

- **`ProcessPoolExecutor` pickling**: Worker functions must be defined at module level (not as lambdas or nested functions) for Python's `pickle` to serialise them. Module-level `_compute_transition_row` and `_compute_reward_row` helpers satisfy this requirement.

- **Single-device pmap**: `jax.pmap` requires at least one device replica per shard. The implementation guards with `if n_devices == 1: return exploitability_batch_jax(...)` so it degrades gracefully on the typical single-CPU setup.

## Declarations

- All hardcoded device references removed — no `.cuda()`, `.cpu()`, or hard-coded `jax.devices("cpu")[0]` outside the `get_jax_device` resolver.
- Device selection is driven entirely by `cfg.device` (`"cpu"` | `"cuda"`), set in `MFGConfig` and resolved once in `create_solver`.
- JAX arrays at numpy→JAX boundaries are placed via `jax.device_put(arr, self.jax_device)` in all four JAX learners.
- CPU-bound Python matrix-building loops (`_build_transition_matrix`, `_build_reward_matrix`) are now parallelised with `ProcessPoolExecutor`.
- PSO particle-batch exploitability evaluation uses `jax.pmap` for multi-device sharding when multiple JAX devices are available, falling back to `jax.vmap` otherwise.
