# plan.md — Base Paper Reproduction Blueprint

**Workspace:** `LyASC/base_model/`
**Base paper:** Liu, F., Ren, X., Zhang, Z., Sun, X., & Zou, Y. (2021). *Rethinking Skip Connection with Layer Normalization in Transformers and ResNets.* arXiv:2105.07205 (COLING 2020 track). Source PDF: `2105_07205v1.pdf`.
**Goal:** Reproduce the paper's **image-classification track** (CIFAR-10 / CIFAR-100, PreAct-ResNet) exactly enough to trust it as the base-model comparison for the LyASC project, before any LyASC-specific code is written.

---

## 0. Scope decision (read this first)

The base paper reports results on **two modalities**: image classification (CIFAR-10/100 + PreAct-ResNet) and machine translation (IWSLT/WMT + Transformer). **This reproduction covers the image-classification track only.** LyASC's Phase 1 scope is CIFAR-10/100 on ResNet backbones (per the revised blueprint), so the Transformer/NLP track (Tables 2, 5, 7) is explicitly out of scope here. If a future LyASC phase extends to sequence models, a `base_model_nlp/` workspace should be created separately — do not fold it into this one.

**What "reproduced with confidence" means for this task:** for every row of Table 1, Table 3, Table 4, and Table 6 in the paper, our re-run mean error rate (averaged over 5 seeds, matching the paper's protocol) should land within **±0.4 absolute percentage points** of the paper's reported number. The paper's own reimplementation margin (their 1xSkip beat He et al.'s original report by 0.06%) tells us tenths-of-a-point differences are normal; anything off by more than ~0.5 points on a row means something in the recreated architecture or training protocol is wrong and must be debugged before trusting this as a baseline for LyASC.

---

## 1. Environment setup (conda)

Create a **dedicated conda environment** for this reproduction — do not reuse or pollute any other environment, since exact package versions matter for numerical reproducibility.

```bash
# from anywhere
conda create -n lyasc-base python=3.10 -y
conda activate lyasc-base

# PyTorch (CUDA 11.8 build shown; swap cu118 -> cpu or another CUDA tag to match your GPU/driver)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118

# supporting packages
pip install numpy==1.26.4 pandas==2.2.2 matplotlib==3.9.0 pyyaml==6.0.1 tqdm==4.66.4 \
            tensorboard==2.17.0 scipy==1.13.1 pytest==8.2.2
```

Alternatively, use the pinned `environment.yml` at the workspace root:

```bash
conda env create -f environment.yml
conda activate lyasc-base
```

Record `python -m torch.utils.collect_env` output into `results/logs/env_report.txt` once the environment is built — this is needed later if reproduced numbers don't match and hardware/driver differences need to be ruled out.

---

## 2. Workspace tree structure

```
LyASC/
└── base_model/
    ├── plan.md                          <- this file
    ├── README.md
    ├── environment.yml
    │
    ├── data/
    │   ├── cifar10/                     <- raw CIFAR-10 goes here (Section 3)
    │   └── cifar100/                    <- raw CIFAR-100 goes here (Section 3)
    │
    ├── src/
    │   ├── __init__.py
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── preact_resnet.py         <- PreAct-ResNet-{20,32,44,56,110} backbone
    │   │   └── skip_variants.py         <- xSkip / xSkip+LN / xSkip+BN / rSkip+LN / rSkip+BN / wSkip+LN gate logic
    │   ├── data/
    │   │   ├── __init__.py
    │   │   └── cifar_datamodule.py      <- dataset loading, augmentation, normalization
    │   ├── training/
    │   │   ├── __init__.py
    │   │   ├── train.py                 <- main training loop (single run, single config)
    │   │   └── config.py                <- hyperparameter dataclasses / config parsing
    │   ├── evaluation/
    │   │   ├── __init__.py
    │   │   ├── evaluate.py              <- computes error rate, aggregates 5-seed mean ± std
    │   │   ├── reproduce_tables.py      <- runs full experiment matrix, emits CSVs matching paper tables
    │   │   └── gradient_norm_probe.py   <- reproduces Figure 3 (per-block gradient norm bar charts)
    │   └── utils/
    │       ├── __init__.py
    │       ├── seed.py                  <- deterministic seeding helper
    │       └── logging_utils.py
    │
    ├── configs/
    │   ├── cifar10_table1.yaml          <- all 10 rows of Table 1 (CIFAR-10, ResNet-110)
    │   ├── cifar10_table3.yaml          <- contracted skip variants (5 rows)
    │   ├── cifar10_table4.yaml          <- batch-norm variants (7 rows)
    │   └── cifar100_table6.yaml         <- ResNet-{20,32,44,56,110} baseline vs 2rSkip+LN (2 rows x 5 depths)
    │
    ├── scripts/
    │   ├── download_cifar10.sh          <- manual download + checksum + extraction
    │   ├── download_cifar100.sh
    │   ├── run_table1_experiments.sh
    │   ├── run_table3_experiments.sh
    │   ├── run_table4_experiments.sh
    │   ├── run_table6_experiments.sh
    │   └── run_gradient_norm_analysis.sh
    │
    ├── results/
    │   ├── logs/                        <- per-run training logs + env_report.txt
    │   ├── checkpoints/                 <- saved model weights per run
    │   ├── tables/                      <- reproduced_table1.csv, reproduced_table3.csv, etc.
    │   └── figures/                     <- reproduced_figure3.png, reproduced_figure4.png
    │
    └── tests/
        ├── test_preact_resnet.py        <- shape/param-count sanity checks
        └── test_skip_variants.py        <- unit tests for each block equation vs. paper's Eq. 1-6
```

---

## 3. Dataset acquisition (manual download)

Do **not** rely solely on `torchvision.datasets.CIFAR10(download=True)` as the first attempt — corporate/campus networks often block the University of Toronto host, and the paper's exact preprocessing needs the raw pickled batches anyway for reproducible per-channel mean/std computation. Download manually as follows.

### CIFAR-10
- **Official source:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Direct file:** https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- **Size:** ~163 MB
- **MD5 checksum:** `c58f30108f718f92721af3b95e74349`
- **Place at:** `LyASC/base_model/data/cifar10/cifar-10-python.tar.gz`, then extract in place:
  ```bash
  cd LyASC/base_model/data/cifar10
  tar -xzf cifar-10-python.tar.gz
  # produces data/cifar10/cifar-10-batches-py/{data_batch_1..5, test_batch, batches.meta}
  ```

### CIFAR-100
- **Official source:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Direct file:** https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
- **Size:** ~161 MB
- **MD5 checksum:** `eb9058c3a382ffc7106e4002c42a8d85`
- **Place at:** `LyASC/base_model/data/cifar100/cifar-100-python.tar.gz`, then extract in place:
  ```bash
  cd LyASC/base_model/data/cifar100
  tar -xzf cifar-100-python.tar.gz
  # produces data/cifar100/cifar-100-python/{train, test, meta}
  ```

**Verify integrity before doing anything else:**
```bash
md5sum data/cifar10/cifar-10-python.tar.gz     # must match c58f30108f718f92721af3b95e74349
md5sum data/cifar100/cifar-100-python.tar.gz   # must match eb9058c3a382ffc7106e4002c42a8d85
```

`scripts/download_cifar10.sh` and `scripts/download_cifar100.sh` automate the above (download via `curl`/`wget`, checksum, extract) for environments where network access to `cs.toronto.edu` *is* available; use them as a starting point and fall back to manual browser download + `scp`/drag-and-drop into the `data/` folders if the host is blocked from the training machine.

**Preprocessing to match the paper exactly:**
- Per-channel mean/std computed from the **CIFAR-10 training set only**: mean ≈ `[0.4914, 0.4822, 0.4465]`, std ≈ `[0.2470, 0.2435, 0.2616]` (standard published values — recompute from the actual downloaded batches and use the recomputed values, don't hardcode, to be safe).
- CIFAR-100 mean ≈ `[0.5071, 0.4865, 0.4409]`, std ≈ `[0.2673, 0.2564, 0.2762]` (same caveat — recompute).
- Train-time augmentation: zero-pad 4 pixels on each side → random 32×32 crop → random horizontal flip → normalize.
- Test-time: normalize only, no augmentation.
- No separate validation split is used by the paper (it evaluates directly on the official 10k test set) — do the same, don't carve out a dev set.

---

## 4. Exact architecture to recreate

### 4.1 Backbone
**PreAct-ResNet** as defined in He et al. (2016b, *Identity Mappings in Deep Residual Networks*), which the base paper builds directly on top of. For CIFAR, depth = 6n+2:
- **ResNet-20** → n=3, **ResNet-32** → n=5, **ResNet-44** → n=7, **ResNet-56** → n=9, **ResNet-110** → n=18.
- 3 stages of `n` basic blocks each, channel widths {16, 32, 64}, stride-2 at the first block of stages 2 and 3.
- Each basic block: `BN → ReLU → Conv3x3 → BN → ReLU → Conv3x3`, pre-activation ordering (this is what "PreAct" means — normalization/activation *before* the conv, not after).
- Stem: single 3×3 conv (16 channels) before stage 1. Head: BN → ReLU → global average pool → FC(num_classes).
- **ResNet-110 = 54 two-layer residual blocks**, matching the paper's own description — use this as a hard sanity check on the implementation (54 blocks × 2 conv layers = 108, + stem conv + FC = 110).

**Implementation risk to flag and verify against the paper/He et al. 2016b before trusting results:** at the two stage-transition blocks (where channel width doubles and stride=2), the identity shortcut cannot be a literal identity — it needs either a zero-padded identity (option A) or a 1×1 projection conv (option B). Confirm which the base paper uses (it does not state this explicitly; He et al.'s CIFAR experiments used option A to keep the network parameter-light) and confirm whether the modulating factor λ is applied to that projected/padded shortcut the same way as to the plain identity shortcuts elsewhere. Log this decision explicitly in `results/logs/` once made — it materially affects reproducibility and is exactly the kind of detail that silently breaks a "faithful" reproduction.

### 4.2 Skip-connection variants (implement all of these — they are the actual object of study)

All variants follow the general form from the paper: `y = G(λx + F(x, W))`, where `G` is an optional normalization function and `λ` is a scalar (or, for wSkip, a per-channel vector).

| Variant | Equation | G | λ |
|---|---|---|---|
| `Baseline` / `1xSkip` | `x + F(x, W)` | none | 1 |
| `2xSkip` | `2x + F(x, W)` | none | 2 |
| `3xSkip` | `3x + F(x, W)` | none | 3 |
| `1xSkip+LN` | `LN(x + F(x, W))` | LayerNorm | 1 |
| `2xSkip+LN` | `LN(2x + F(x, W))` | LayerNorm | 2 |
| `3xSkip+LN` | `LN(3x + F(x, W))` | LayerNorm | 3 |
| `1rSkip+LN` | `LN(x + F(x, W))` (= 1xSkip+LN) | LayerNorm | 1 |
| `2rSkip+LN` | `LN(x + LN(x + F(x, W)))` | LayerNorm, recursive | 2 |
| `3rSkip+LN` | `LN(x + LN(x + LN(x + F(x, W))))` | LayerNorm, recursive | 3 |
| `0.5xSkip` (Table 3) | `0.5x + F(x, W)` | none | 0.5 |
| `LN(0.5x+F)` (Table 3) | `LN(0.5x + F(x, W))` | LayerNorm | 0.5 |
| `LN(x+2F)` (Table 3) | `LN(x + 2·F(x, W))` | LayerNorm | — (scales F, not x) |
| `LN(x+3F)` (Table 3) | `LN(x + 3·F(x, W))` | LayerNorm | — (scales F, not x) |
| `1xSkip+BN` / `2xSkip+BN` / `3xSkip+BN` (Table 4) | same as xSkip+LN but `G`=BatchNorm | BN | 1,2,3 |
| `1rSkip+BN` / `2rSkip+BN` / `3rSkip+BN` (Table 4) | same as rSkip+LN but `G`=BatchNorm | BN | 1,2,3 |
| `wSkip+LN` | `LN(w · x + F(x))`, `w ∈ ℝ^C` learnable, init `w=1` | LayerNorm | learned vector |

**Note for the LyASC project specifically:** `wSkip+LN` is the single most important row to get exactly right — it is the paper's "trivial extension: learn λ" ablation, and it is the direct ancestor of LyASC's own mandatory "learned gate, no stability signal" control baseline. The paper reports it *underperforms* the fixed-scalar recursive version (8.66% vs ~6.02% error) and that the learned `w` collapses to a small norm (~0.24 average per-dimension scale), which the paper interprets as overfitting risk. Reproducing this specific failure mode is a strong signal the reimplementation is faithful — if `wSkip+LN` reproduces *well* (matches the paper's poor result and small-norm collapse) it validates the whole harness, since this is the row most likely to silently diverge from the paper if some detail is wrong.

### 4.3 Recursive definition — implement literally as the recurrence, not via the closed form
`y_λ = LN(x + y_{λ-1})`, `y_1 = LN(x + F(x, W))`. Implement it as an actual recursive/iterative computation for `λ` steps, not by algebraically pre-expanding it — this keeps the implementation checkable line-by-line against the paper's Equation 4, and the recursive computation is cheap regardless (λ ∈ {1,2,3} only).

---

## 5. Training protocol (must match to reproduce numbers)

The paper states it **"mostly adopts hyperparameters from He et al. (2016b) to keep the number of variables controlled and comparable, except that we use a weight decay rate of 0.0002."** He et al. (2016b)'s CIFAR protocol (standard and widely replicated) is:

- **Optimizer:** SGD, momentum = 0.9, Nesterov = False (unless verified otherwise).
- **Weight decay:** **0.0002** (explicit override stated in the base paper — do not use He et al.'s original 0.0001).
- **Batch size:** 128.
- **LR schedule:** initial LR = 0.1, divided by 10 at iteration 32,000 and again at 48,000, training terminates at 64,000 iterations. With 50,000 training images and batch size 128, that is ≈128 epochs total, step-downs at ≈82 and ≈123 epochs. **Verify this iteration/epoch mapping explicitly in code** (log both iteration count and epoch count per checkpoint) since off-by-one schedule errors are a classic silent reproduction failure.
- **Weight decay scope:** apply to conv/linear weights; verify whether BN/LN affine parameters and biases are excluded from weight decay (common practice, not stated explicitly by either paper) — implement it excluded by default and note this as a deliberate assumption in `results/logs/`.
- **Initialization:** standard He (Kaiming) initialization for conv layers, matching He et al.'s own papers.
- **Seeds:** run **5 random seeds** per configuration (paper reports "Average of 5 runs" for Table 1) and report mean error rate; for Table 6 the paper doesn't restate the run count explicitly — use 5 there too for consistency and note the assumption.
- **Metric:** top-1 **classification error rate (%)** on the official 10,000-image test set (lower is better), not accuracy — match the paper's convention exactly so numbers are directly comparable without conversion mistakes.

---

## 6. Experiment matrix (what to actually run, and what to check it against)

| Config file | Paper table | Rows to reproduce | Model | Dataset |
|---|---|---|---|---|
| `configs/cifar10_table1.yaml` | Table 1 | Baseline/1xSkip 6.31–6.37, 2xSkip 8.41, 3xSkip 9.23, 1xSkip+LN 7.72, 2xSkip+LN 6.29, 3xSkip+LN 6.07, 1rSkip+LN 7.72, 2rSkip+LN **6.02 (best)**, 3rSkip+LN 6.55 | PreAct-ResNet-110 | CIFAR-10 |
| `configs/cifar10_table3.yaml` | Table 3 | baseline 6.31, 0.5x+F 18.61, LN(0.5x+F) 29.88, LN(x+2F) 24.98, LN(x+3F) 61.02 | PreAct-ResNet-110 | CIFAR-10 |
| `configs/cifar10_table4.yaml` | Table 4 | 1xSkip 6.31, 1xSkip+BN 8.23, 2xSkip+BN 6.43, 3xSkip+BN 6.50, 1rSkip+BN 8.23, 2rSkip+BN 6.77, 3rSkip+BN 6.68 | PreAct-ResNet-110 | CIFAR-10 |
| `configs/cifar100_table6.yaml` | Table 6 | ResNet-20 baseline 33.38 / +2rSkip+LN 32.99; ResNet-32 32.00 / 31.39; ResNet-44 30.50 / 29.85; ResNet-56 29.81 / 28.98; ResNet-110 28.71 / 27.69 | PreAct-ResNet-{20,32,44,56,110} | CIFAR-100 |
| (also run) `wSkip+LN` | Table 1 text (not tabulated) | 8.66 avg error, final ‖w‖ ≈1.34 at middle layers → ~0.24 avg per-dim scale | PreAct-ResNet-110 | CIFAR-10 |

Also reproduce, as secondary verification (not strictly required to trust the baseline, but strongly recommended since it validates the *mechanism*, not just the final number):
- **Figure 3** — per-block gradient norm bar charts at middle-depth blocks, for 1xSkip, 2xSkip, 2xSkip+BN, 0.5xSkip, 1xSkip+LN, 2xSkip+LN, 2rSkip+LN, 0.5xSkip+LN, averaged over 10,000 random training examples on a pretrained model.
- **Figure 4** — training/validation loss and error curves for `2rSkip+LN` vs `wSkip` (both without LN suffix shown, per the paper's own note) over 50,000 steps, showing `wSkip`'s slow start.

Acceptance rule (restated from Section 0): every reproduced number within **±0.4 absolute points** of the paper's value. If `wSkip+LN`, Figure 3, and Figure 4 all reproduce the paper's *qualitative* story (small-w collapse, gradient malformation on unnormalized scaling, wSkip's slow start) even where the exact numbers drift slightly, that is strong corroborating evidence the harness is faithful.

---

## 7. Execution runbook (step-by-step order)

1. `conda env create -f environment.yml && conda activate lyasc-base`
2. `bash scripts/download_cifar10.sh` and `bash scripts/download_cifar100.sh` (or manual download per Section 3); verify checksums.
3. `pytest tests/` — confirm `preact_resnet.py` produces the correct parameter count and block count for ResNet-110 (54 two-layer blocks) and that each skip-variant module in `skip_variants.py` matches its equation from Section 4.2 via a hand-computed unit test on a tiny random tensor.
4. `bash scripts/run_table1_experiments.sh` — trains all 10 Table 1 variants × 5 seeds on CIFAR-10.
5. `bash scripts/run_table3_experiments.sh` and `run_table4_experiments.sh` — same pattern, smaller variant sets.
6. `bash scripts/run_table6_experiments.sh` — trains baseline vs 2rSkip+LN across 5 depths × 5 seeds on CIFAR-100 (10 configs total).
7. `python src/evaluation/reproduce_tables.py` — aggregates all logged runs into `results/tables/reproduced_table{1,3,4,6}.csv`, computes mean/std per row, and diffs against the paper's hardcoded reference values (Section 6) with the ±0.4-point rule, printing a PASS/FAIL per row.
8. `bash scripts/run_gradient_norm_analysis.sh` — loads pretrained checkpoints from step 4/5, runs `gradient_norm_probe.py` to reproduce Figure 3.
9. Manually inspect `results/figures/reproduced_figure4.png` against the paper's Figure 4 for the `wSkip` slow-start qualitative match.
10. Only once steps 4–9 pass the acceptance rule: sign off on this base model as validated, and hand off to the LyASC-specific implementation phase (outside this workspace).

---

## 8. Known risks / things likely to silently break reproduction

- **LR schedule as iterations vs. epochs** — implement and log both; a wrong conversion is the most common source of a systematic ~1-2 point gap across *every* row.
- **Weight decay scope** (BN/LN params excluded or not) — try both if the first attempt doesn't match, and record which was used.
- **Downsampling-block shortcut handling under λ-scaling** (Section 4.1) — undocumented by the paper, must be decided and logged.
- **Data augmentation order and padding mode** (zero-pad vs reflect-pad) — use zero-pad 4px, the standard for this literature, unless results are systematically off.
- **PreAct-ResNet head details** — confirm whether a final BN-ReLU is applied before global average pooling (yes, per He et al. 2016b) — omitting it is a common bug that mainly hurts deeper variants.
- **Random seed coverage** — "5 runs" must use 5 *different* seeds, not the same seed with nondeterministic cuDNN; set and log actual seed values per run for full traceability.
