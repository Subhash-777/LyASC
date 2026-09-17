# LyASC Model — Lyapunov-Adaptive Skip Connections

Experimental workspace for the LyASC architecture, implementing and evaluating
stability-signal-driven skip-connection gating as described in
[LyASC_ARCHITECTURE.md](../LyASC_ARCHITECTURE.md).

## Plan

The implementation blueprint is at [../LyASC_plan.md](../LyASC_plan.md).
That file is authoritative for the implementation order, acceptance criteria,
and workspace conventions.

## Environment

```bash
# The lyasc-experiments env is cloned from lyasc-base with mlflow added.
# See environment_delta.yml for the delta.
conda activate lyasc-experiments
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Workspace Structure

```
lyasc_model/
├── configs/              # YAML experiment configs
├── results/
│   ├── checkpoints/      # model checkpoints
│   ├── figures/          # stability maps, comparison plots
│   ├── logs/             # decisions.log, per-run logs, env_report
│   ├── review/           # HTML review reports
│   └── tables/           # result CSVs
├── src/
│   ├── evaluation/       # compare_to_baseline.py
│   ├── models/           # lyasc_skip.py, stability_estimator.py, etc.
│   ├── tracking/         # mlflow_tracking.py
│   ├── training/         # train.py (wrapper around base_model), config.py
│   └── utils/            # logging_utils.py (decisions.log shim)
├── tests/                # unit tests
└── environment_delta.yml # deps added on top of base_model/environment.yml
```

## Comparison Baselines

All baselines come from `../base_model/results/tables/reproduced_table*_FINAL.csv`
(verified, read-only). Key reference numbers:

| Variant    | 3-seed mean error | Source |
|-----------|-------------------|--------|
| 1xSkip    | 6.27%             | Table 1 |
| 2rSkip+LN | 6.15%             | Table 1 |
| wSkip+LN  | 8.16%             | Table 1 |
