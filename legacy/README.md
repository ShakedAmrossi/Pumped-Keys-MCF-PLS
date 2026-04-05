# Legacy Code

This directory contains exploratory figure-generation functions developed
during the v1 and v2 iterations of the simulation.

## What is here

`figure_sweeps.py` — ~3800 lines of sweep and plotting functions (Figs 2–25
from the earlier exploratory phase) plus `main_full_legacy()` which writes
to `mcf_pls_figs/` (not `mcf_pls_figs3/`).

## Why it is kept

- Reproducibility of earlier results reported in lab notes.
- Reference for the exploratory analyses that informed the publication design.
- Contains some functions (e.g., `sweep_secrecy_capacity`, `ml_adversary_benchmark`)
  that may be useful for future experiments.

## What is NOT here

Any function called by `run.py` / `main()` — those live in `mcf_pls/benchmarks.py`.

## How to run legacy figures

```python
from legacy.figure_sweeps import main_full_legacy
main_full_legacy()
```

Output goes to `mcf_pls_figs/` (older format, not the publication suite output).
