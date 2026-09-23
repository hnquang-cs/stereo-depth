# stereo-depth — label-free self-supervised stereo

A clean re-implementation of the stereo network from **"A Learned Stereo Depth System
for Robotic Manipulation in Homes"** (Shankar, Tjersland, Ma, Stone, Bajracharya,
[arXiv:2109.11644](https://arxiv.org/abs/2109.11644)), trained **without any
ground-truth disparity or depth**.

The architecture is the paper's. The training signal is Monodepth's
([Godard et al. 2017](https://arxiv.org/abs/1609.03677)): where the original regresses
onto ground-truth disparity, this trains on stereo photometric reconstruction,
left–right consistency and edge-aware smoothness. Ground truth appears in exactly one
place — measuring the finished model.

The paper's NSCE term is **absent, not replaced**: it is anchored on ground-truth
disparity, so it has no label-free form, and substituting an invented loss for it would
no longer be a reimplementation of the paper.

The question this repository is built to answer is: **how far can the same stereo
architecture get without disparity labels, measured honestly against real ground truth?**

---

# Training Requires No Ground-Truth Labels

The full pipeline runs on a dataset containing nothing but

```
dataset/
├── left/
└── right/
```

There is no supervised disparity loss, no supervised depth loss and no
ground-truth-derived target of any kind. The complete objective is Monodepth's,
equation 2:

```
L = 1.0 · L_photometric      (SSIM + L1, both directions)   a_ap
  + 1.0 · L_left_right                                      a_lr
  + 0.1 · L_smoothness       (edge-aware, mean-normalised)  a_ds
```

`a_lr = 1` is meaningful only because Monodepth's disparity is a *fraction of image
width*; the disparity-space terms here are normalised by the search range for exactly
that reason.

Two optional terms are implemented but default to `0.0`, and are not part of Monodepth:
`low_resolution` (the same terms repeated on the soft-argmin output, the closest
analogue of Monodepth's four-scale sum) and `confidence` (a label-free matchability
target).

This is enforced, not just asserted:

* A dataset constructed in `DatasetMode.TRAIN` or `DatasetMode.VALIDATION` never opens
  a ground-truth file. Only `DatasetMode.BENCHMARK` does.
* `assert_label_free()` runs on **every training batch** and raises if any
  ground-truth key is present.
* `tests/test_label_isolation.py` corrupts the ground-truth files on disk and asserts
  that the training loss, the gradient norm, the predicted disparity and every logged
  quantity are all bit-identical — while separately asserting the benchmark path *does*
  see the change, so the test cannot pass vacuously.
* `scripts/audit_label_leakage.py` statically audits the repository (see
  [Label Leakage Audit](#label-leakage-audit)).

# Ground Truth Is Used for Evaluation Only

`evaluate.py` is the only entry point that reads ground truth. It loads a checkpoint,
calls `model.eval()`, runs under `torch.no_grad()`, and constructs no optimiser.
Checkpoint selection during training uses a **label-free** criterion (validation
photometric reconstruction loss), so ground truth never even influences *which* weights
are kept.

```
UNLABELED IMAGES → training → frozen checkpoint → benchmark inference → GT comparison → metrics
```

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Get data (see docs/DATASETS.md)
python -m stereo.data.download middlebury kitti2015
python -m stereo.data.download --verify

# 2. Train, label-free, from random initialisation
python train.py --config configs/train_unlabeled.yaml

# 3. Benchmark the frozen checkpoint against ground truth
python evaluate.py --checkpoint outputs/train_unlabeled/best.pt \
                   --protocol middlebury2014 \
                   --dataset-root datasets/middlebury/MiddEval3/trainingH \
                   --both            # raw AND post-processed, side by side

# 4. Run on a pair of images
python inference.py --checkpoint outputs/train_unlabeled/best.pt \
                    --left a.png --right b.png --focal-length 1075 --baseline 0.12
```

## Repository layout

```
stereo-depth/
├── train.py  evaluate.py  inference.py       entry points
├── configs/                                  train_unlabeled / adapt_unlabeled / evaluate
├── stereo/
│   ├── geometry.py             disparity conventions, warping, rescaling, depth, padding
│   ├── config.py               dataclass config + YAML loader
│   ├── postprocess.py          matchability post-processing (kept out of the network)
│   ├── model/                  blocks, feature_extractor, cost_volume, aggregation,
│   │                           refinement, stereo_net
│   ├── losses/                 photometric, smoothness, consistency, confidence
│   ├── data/                   base (mode + guard), stereo_folder, sceneflow, kitti,
│   │                           middlebury, eth3d, augmentation, io, registry, download
│   ├── training/               objective, loop
│   ├── evaluation/             disparity_metrics, depth_metrics, confidence_metrics,
│   │                           protocols, benchmark          ← the only GT consumers
│   └── utils/                  checkpoint, seed, calibration, visualization
├── scripts/audit_label_leakage.py
├── notebooks/kaggle_training.ipynb
├── tests/
└── docs/DATASETS.md  docs/REPORT.md
```

Everything lives under one `stereo/` package so imports are unambiguous (a top-level
`datasets/` module would collide with the HuggingFace package of that name).

## Architecture

Preserved from the paper, verified against the reference implementation:

```
left  ──► dilated ResNet ──┐
                           ├──► cross-correlation cost volume (16-d features)
right ──► dilated ResNet ──┘              │
                                   3D convs → flatten → dilated 2D convs
                                          │
                             ┌────────────┴────────────┐
                             ▼                         ▼
                        soft argmin               matchability
                             │                         │
                             └──► second dilated ResNet ◄──── reference image
                                          │
                              full-resolution disparity
```

| Paper component | Here |
|---|---|
| Dilated ResNet feature extractor, 16-d, /4 or /8 | `model/feature_extractor.py` |
| Cross-correlation cost volume | `model/cost_volume.py` |
| 3D then 2D cost aggregation | `model/aggregation.py` |
| Soft argmin | `cost_volume.soft_argmin` |
| Matchability (negative entropy) | `cost_volume.matchability` |
| Second dilated ResNet refinement | `model/refinement.py` |

Hybrid-dilated-convolution rates `(1, 2, 5, 9)`, the `[3, 4, 8]` block schedule, the
pre-activation residual blocks and the `4D → 4D → 2D → D → D` aggregation channel
schedule all follow the reference.

### Bidirectional prediction without a second cost-volume implementation

The reverse direction comes from a derived identity rather than a second indexing scheme.
With `flip(a)[x] = a[W-1-x]` and `x' = W-1-x`:

```
corr(flip(src), flip(ref))[d, x] = src[W-1-x] · ref[W-1-x+d] = src[x'] · ref[x' + d]
```

which is the right-referenced cost volume, mirrored. Feeding the mirrored feature maps
through the *same* weights therefore produces `d_R`, mirrored, with identical border
statistics — no extra feature-extraction pass and no duplicated code.
`tests/test_model.py::test_mirror_trick_gives_the_right_referenced_volume` checks the
identity numerically.

### Disparity convention

Derived, not guessed (`stereo/geometry.py`):

```
x_R = x_L − f·B/Z,   d = x_L − x_R = f·B/Z ≥ 0

d_L:  left pixel x matches right pixel x − d    →  Î_L(x) = I_R(x − d_L(x))
d_R:  right pixel x matches left pixel x + d    →  Î_R(x) = I_L(x + d_R(x))
```

Warping uses `align_corners=True` with hand-computed normalisation (exact for integer
shifts). Resizing uses `align_corners=False`, whose coordinate map scales x-differences
by exactly `W_new/W_old` — under `align_corners=True` the factor would be
`(W_new−1)/(W_old−1)`, a different number and a classic source of sub-pixel drift.

## Arbitrary resolution

Inputs are padded on the **right and bottom** to a multiple of 16 (the deepest stride)
and the outputs are cropped back. Padding right/bottom leaves every pixel's x coordinate
unchanged, so disparity values are untouched; left padding would shift x and silently
corrupt disparity, so it is never done. Tested at 224×224, 320×240, 384×384, 640×192,
1242×375, 960×540 and 150×100.

## Dynamic disparity search range

```python
compute_num_disparities(width, downsample, cap=384)   # min(width // 2, 384), floored to a multiple of downsample
```

```
224 → 112     512 → 256     640 → 320     1024 → 384     1920 → 384
```

This width-driven rule is a **new implementation choice, not the paper's policy** — the
paper picks the range per dataset by hand (256 for Scene Flow/KITTI, 384 for the authors'
camera, 512 for Middlebury). The rule automates that choice; the 384 cap matches the
paper's headline configuration.

**Important limitation.** The aggregation stack flattens the disparity axis into
channels, so the search range is part of the weights: a trained checkpoint has one fixed
`num_disparities`. Spatial resolution stays completely free, but running a model on a
much wider image does not widen its search range. The original architecture has the same
property — which is why the paper rebuilds per dataset. Checkpoints record their config
and `build_model_from_checkpoint()` reconstructs the exact architecture.

## Training stages

| Stage | What | Config |
|---|---|---|
| 1 | Self-supervised training from random init (no pretrained weights of any kind) | `train_unlabeled.yaml` |
| 2 | Adaptation to your unlabeled camera, smaller learning rate | `adapt_unlabeled.yaml` |
| 3 | Frozen ground-truth benchmark | `evaluate.py` |

Stages 1–2 use no ground truth. Stage 3 may, because no optimisation happens.

The paper's own Stage 2 — EMA teacher/student pseudo-label self-training — is **not
implemented**. It is not part of the Monodepth objective this uses, and it was removed
rather than left as dead configuration.

One design point that matters:

* **The photometric loss is not masked by the occlusion mask.** It is a masked mean, so
  a model that made its two disparity maps disagree everywhere could drive that mask —
  and the loss — to zero. Following Monodepth, occlusions are handled by the left–right
  consistency term, and the photometric term is masked only by things the network cannot
  manipulate: the valid-warp region and the image border.
### Matchability without ground truth

Matchability is parameter-free — the negative entropy of `softmin(cost)` — so it is
shaped entirely by whatever shapes the cost volume. In the paper that shaping comes from
the Noise-Sampling Cross-Entropy loss, which peaks the cost curve **at the ground-truth
disparity**. That is unavailable here.

`losses/confidence.py` substitutes a label-free target: a binary reliability mask
(left–right agreement ∧ valid warp ∧ low photometric residual), regressed onto
`exp(matchability)` with BCE. It is weighted low (0.05) by default and can be set to 0.

Stated plainly: this is a new loss, it is self-referential (the target is a function of
the model's own outputs), and **its effect on disparity accuracy has not been measured
here** because no benchmark run was possible in this environment. A collapse monitor logs
mean confidence and warns above 0.99.

## Augmentation

* **Random scale, aspect ratio and resize** are applied **per batch**, not per sample.
  Combining a random scale with a fixed tensor shape otherwise requires random cropping,
  which this project does not use. One random output size per batch keeps the batch
  rectangular without cropping anything. Intrinsics are updated so `depth = f·B/d` survives.
* **No random cropping anywhere.** `tests/test_data.py::test_no_random_crop_transform_exists`
  asserts the module does not even contain one. The reference implementation's
  `RandomCrop` is training augmentation, not required preprocessing, so it was dropped
  rather than silently retained.
* **Colour jitter is applied with identical parameters to both views** — independent
  jitter breaks the brightness constancy the photometric loss rests on. The un-jittered
  pair is kept as `left_clean`/`right_clean`: the network sees the jittered pair while
  the photometric loss always reconstructs the clean images, so the jitter cannot be
  "solved" by the reconstruction. Both share the same geometry, so no disparity
  rescaling is needed.

## Evaluation protocols

`stereo/evaluation/protocols.py` encodes each protocol completely — split, valid-pixel
mask and its justification, resolution, post-processing, aggregation — and every result
file records it.

| Protocol | Primary metrics | Source |
|---|---|---|
| `sceneflow` | EPE, %Bad(1.0), global per-pixel | paper Table IV; mask `1e-3 < d_gt < 251` |
| `sceneflow_literature_mask` | same, mask `d_gt < 192` | the convention of the comparison methods |
| `middlebury2014` | bad2, bad4, avgerr, rms, A90, A95 (nonocc/all), per image | paper Table V metric set, **training split** |
| `eth3d`, `kitti2015`, `kitti2012`, `custom_folder` | secondary | not paper metrics |

Both aggregations are always computed and always labelled: `global_*` is the reference
implementation's per-pixel accumulation, `image_*` is the per-image mean used by
Middlebury and KITTI. Post-processed runs additionally report
`ground_truth_pixel_coverage`, because a metric computed on 40% of the pixels is not
comparable to one computed on all of them.

**No scale alignment is applied anywhere.** A calibrated stereo network predicts metric
disparity; median scaling would measure something else. `median_scale_factor()` exists
for protocols that mandate it and the applied value is always recorded.

See [docs/REPORT.md](docs/REPORT.md) for the full protocol derivation, the published
reference numbers, and what could and could not be reproduced here.

## Kaggle

`notebooks/kaggle_training.ipynb` runs with `MODE = "train_unlabeled"`,
`"adapt_unlabeled"` or `"evaluate"`. Ground truth is loaded **only** when
`MODE == "evaluate"`.

Every setting lives in a single documented **Control Panel** cell; the rest of the notebook
derives from it, so adding a dataset or switching to evaluation is a one-line edit. A
preflight cell then prints what your settings actually mean (disparity reach in pixels, cost
volume memory, which data will be downloaded) and warns about inconsistencies before anything
runs.

Large datasets are **attached, not downloaded** — Scene Flow is 132 GB against Kaggle's 20 GB
quota:

```python
DATASETS = {"sceneflow": 1.0}
ATTACHED = {"sceneflow": "/kaggle/input/sceneflow"}
```

The mirror's internal folder layout is discovered automatically (official or
`FlyingThings3D_subset`, at any nesting depth); see [docs/DATASETS.md](docs/DATASETS.md).

## Tests

```bash
python -m pytest tests/ -q          # 109 tests
```

Covers: disparity sign conventions (including an assertion that the *wrong* warp
direction fails), sub-pixel warping, left–right consistency on a slanted surface,
disparity rescaling under resize, the dynamic-disparity rule and its alignment, padding
round-trips, forward passes at seven resolutions, the mirror-trick identity, soft-argmin
and matchability limits, every loss, gradient isolation, metrics against
hand-computed values, weighted multi-dataset sampling, post-processing gates, an
end-to-end Trainer epoch, a full train → freeze → evaluate run on synthetic data with
analytically known ground truth, the Middlebury nonocc/all protocol on synthetic scenes,
and an architecture-fidelity comparison against the reference implementation (skipped
automatically if `mmstereo` is not present next to this repository).

That last one is worth highlighting: at the reference's own Scene Flow model settings,
this implementation has **exactly** the reference's parameter count — 5,661,646 total,
and identical per component (feature extractor 708,000 / aggregation 1,857,848 /
refinement 3,095,798) — and its cost volume is numerically equal to the reference's, in
both directions.

## Label Leakage Audit

```bash
python scripts/audit_label_leakage.py --strict
```

Tokenises every Python file (AST for docstrings, `tokenize` for comments, column-precise)
so a mention in prose is counted separately from one in executable code — and a dict key
like `"disparity_gt"` counts as code, not prose. Current result:

```
FORBIDDEN         0 in code      stereo/{model,losses,training}/, geometry.py, config.py, train.py
GUARD             5 in code      stereo/data/base.py — names the keys in order to reject them
LOADERS          15 in code      dataset modules — _load_ground_truth, BENCHMARK mode only
ALLOWED         133 in code      stereo/evaluation/, postprocess, evaluate.py, tests, docs
Evaluation imports on the training path: 0
RESULT: PASS
```

## Limitations

Read [docs/REPORT.md](docs/REPORT.md#limitations) for the full list. The headline ones:

* **No benchmark numbers were produced.** This environment has no GPU and no datasets;
  every component is tested, but no accuracy result was measured. No number in this
  repository is fabricated — unavailable results are marked *Not measured*.
* **The reference implementation ships no pretrained checkpoint**, so the "original
  mmstereo checkpoint" comparison row cannot be filled without training it from scratch
  on Scene Flow with ground truth.
* **The paper's Middlebury row is the hidden TEST split.** Only the training split can be
  scored locally, and it is labelled as such.
* **The label-free matchability loss is unvalidated** (see above).
* A label-free model is expected to be **less accurate** than the original supervised one.
  That gap is the measurement this repository exists to make, not something to hide.

## Relationship to the reference implementations

Neither repository was copied. `mmstereo` supplied the architecture, the metric
definitions and the post-processing constants; its PyTorch Lightning harness, Hydra-style
config tree, ONNX/TorchScript export, duplicated residual-block classes and crop-based
augmentation were dropped. `monodepth` supplied the self-supervised concepts; its MSE
reprojection loss was replaced with the published SSIM+L1 form, its unsigned smoothness
term was corrected to the published absolute-value formulation, and its warp — which adds
a width-fraction disparity to a `[-1,1]` grid, a factor-of-two error — was re-derived in
pixels.

## Citation

```
@article{shankar2021learned,
  title={A Learned Stereo Depth System for Robotic Manipulation in Homes},
  author={Shankar, Krishna and Tjersland, Mark and Ma, Jeremy and Stone, Kevin and Bajracharya, Max},
  journal={arXiv preprint arXiv:2109.11644},
  year={2021}
}
```
