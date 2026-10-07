# LyASC / base_model

Faithful reproduction harness for the base paper used in the LyASC project:

> Liu, F., Ren, X., Zhang, Z., Sun, X., & Zou, Y. (2021). *Rethinking Skip Connection with
> Layer Normalization in Transformers and ResNets.* arXiv:2105.07205.

Scope: **image-classification track only** (CIFAR-10 / CIFAR-100, PreAct-ResNet).
See `plan.md` for the full blueprint — dataset setup, exact architecture specification,
training protocol, experiment matrix, and acceptance criteria. Read `plan.md` before
touching any code in `src/`.

## Quick start

```bash
conda env create -f environment.yml
conda activate lyasc-base
bash scripts/download_cifar10.sh
bash scripts/download_cifar100.sh
pytest tests/
bash scripts/run_table1_experiments.sh
```

## Status

This workspace currently contains the **planning scaffold only** (directory structure,
configs with paper-matched hyperparameters, download scripts, and empty module
stubs with defined interfaces). Model, training-loop, and evaluation code under
`src/` is intentionally left as TODO stubs — see `plan.md` Section 7 for the
implementation runbook this scaffold is designed for.
