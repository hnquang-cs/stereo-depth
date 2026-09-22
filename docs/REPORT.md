# Final report

Re-implementation of the stereo network from *A Learned Stereo Depth System for Robotic
Manipulation in Homes* (Shankar, Tjersland, Ma, Stone, Bajracharya; arXiv:2109.11644),
trained **label-free** and evaluated against real ground truth.

Status vocabulary used throughout, per the project's reporting rule:

| Term | Meaning |
|---|---|
| **Implemented** | code exists |
| **Tested** | verified by an executed test in this session |
| **Not tested** | implemented, never exercised |
| **Blocked** | cannot be done in this environment |
| **Published** | transcribed from the paper |
| **Not measured** | would require data/compute unavailable here |

---

## 1. Architecture

The paper's five components are preserved. Mapping:

| Paper | Module | Detail |
|---|---|---|
| Dilated ResNet feature extraction, 16-d, /4 or /8 | `stereo/model/feature_extractor.py` | `[3, 4, 8]` blocks, widths `(w, 2w, 4w)`, HDC rates `(1, 2, 5, 9)` in the deepest group, top-down 1×1 score head |
| Cross-correlation cost volume | `stereo/model/cost_volume.py` | element-wise (vector-valued) product, `(B, C, D, H, W)` |
| 3D then 2D cost aggregation | `stereo/model/aggregation.py` | Conv3d `C → C/2 → 4`, flatten, dilated 2D residual stack `4D → 4D → 2D → D → D`, 1×1 output |
| Soft argmin | `cost_volume.soft_argmin` | `Σ k · softmin(cost)_k` |
| Matchability | `cost_volume.matchability` | `Σ p log p`, `p = softmin(cost)`; `confidence = exp(m) ∈ [1/D, 1]` |
| Second dilated ResNet refinement | `stereo/model/refinement.py` | image + low-res disparity + matchability → residual on the upsampled base disparity, `ReLU` output |

### Fidelity, measured

**Tested.** `tests/test_architecture_fidelity.py` instantiates the reference `mmstereo`
model and compares it directly. At the reference's own `config_sceneflow.yaml` model
settings (`num_disparities=256`, `downsample=4`, `fe_features=16`, `fe_internal_features=16`):

| Component | Reference | This implementation |
|---|---:|---:|
| Feature extractor (backbone + score head) | 708,000 | 708,000 |
| Cost aggregation | 1,857,848 | 1,857,848 |
| Refinement | 3,095,798 | 3,095,798 |
| **Total** | **5,661,646** | **5,661,646** |

Also verified numerically: the cost volume is bit-comparable to the reference's
`is_right=False` branch, the mirror trick reproduces its `is_right=True` branch exactly,
and `max_disparity` / `max_disparity_small` / `scale` agree.

These tests skip automatically if the reference repository is absent.

## 2. Differences from `mmstereo`

Removed: PyTorch Lightning, OmegaConf/Hydra config tree, Kornia, TurboJPEG, ONNX and
TorchScript export, the `Sample`/`SampleElement`/`ElementKeys` namedtuple-and-enum data
layer, `CameraEffect` batch transforms, the `null_loss`/`dummy_loss`/`valid_loss` NaN
plumbing, and the near-duplicate `PreactBasicResidualBlock` / `LeakyPreactBasicResidualBlock`
pair (one class with a `leaky` flag).

Corrected or simplified:

* The reference's `RefineInBlock` accepts a `cost_volume` argument and never uses it.
  Dropped — which also matches the paper's stated refinement inputs (image, low-resolution
  disparity, matchability).
* Two cost-volume indexing branches (`is_right`) replaced by one function plus the mirror
  identity (§4 below).
* Hard-coded resolutions and dataset paths replaced by explicit padding and configuration.
* `cv2.imread` BGR replaced with RGB, so a checkpoint trained here expects RGB.
* PFM reading no longer negates: the reference decodes PFM through OpenCV, which applies
  the header's signed scale, then negates to compensate. `read_pfm` uses the sign only for
  byte order, as the format specifies.
* Removed all three supervised losses (`DisparityLoss`, `NsceLoss`, and the supervised part
  of the smoothness path) — this is the point of the project.
* Removed `RandomCrop`. Inspected first: it is training augmentation (random offsets with a
  valid-disparity retry loop plus a vertical shift of the right image), not required
  preprocessing, so dropping it loses no dataset correctness.

## 3. MonoDepth concepts reused

From Godard et al. 2017, reimplemented rather than copied:

* **SSIM + L1 appearance matching**, `α = 0.85`, 3×3 SSIM.
* **Edge-aware first-order smoothness**, `|∂d| · exp(−|∂I|)`.
* **Left–right consistency** as both a loss and an occlusion signal.
* From Monodepth2: **mean-normalised disparity** in the smoothness term.

Three defects in the local `monodepth` were *not* reproduced, and the corrections are
documented at the point of deviation:

1. Its reprojection loss is plain MSE — dominated by the few high-residual pixels
   (occlusions, speculars) that stereo photometric training must survive. Replaced with
   the published SSIM+L1.
2. Its smoothness term omits the absolute values on both the disparity and the image
   gradients, making the penalty *signed*: it can be driven arbitrarily negative by a
   disparity field that decreases monotonically to the right.
   `tests/test_losses.py::test_smoothness_penalises_gradients_and_respects_edges` pins the
   corrected behaviour.
3. Its warp adds a width-fraction disparity to a `[-1, 1]` normalised grid, which is off by
   a factor of two. Warping here is derived in pixels in `stereo/geometry.py`.

## 4. Geometry, derived

```
x_L = fX/Z + c,   x_R = f(X−B)/Z + c   ⇒   x_R = x_L − fB/Z
d = x_L − x_R = fB/Z ≥ 0

d_L(x): left pixel x ↔ right pixel x − d   ⇒   Î_L(x) = I_R(x − d_L(x))
d_R(x): right pixel x ↔ left pixel x + d   ⇒   Î_R(x) = I_L(x + d_R(x))
```

**Reverse direction without a second cost volume.** With `flip(a)[x] = a[W−1−x]` and
`x' = W−1−x`:

```
corr(flip(src), flip(ref))[d, x] = src[W−1−x] · ref[W−1−x+d] = src[x'] · ref[x' + d]
```

which is the right-referenced volume, mirrored — with the *left*-referenced border
pattern, so the same aggregation weights see the same statistics. Cost: zero extra
feature-extractor passes. **Tested** numerically against both reference branches.

**Resize convention.** Warping uses `align_corners=True` with hand-computed normalisation
(exact for integer shifts). Resizing uses `align_corners=False`, whose coordinate map
scales x-differences by exactly `W_new/W_old`; under `align_corners=True` the factor is
`(W_new−1)/(W_old−1)`. Mixing the two silently is a sub-pixel disparity bug, so
`resize_disparity()` bundles the interpolation with the matching scale factor and
`resize_scale_x()` exposes both conventions. This was found by a failing test, not by
inspection.

## 5. Label-free training formulation

```
L = w_photo  · L_photometric      SSIM+L1, both directions, masked by valid warp + border
  + w_lr     · L_left_right       |d_L(x) − d_R(x − d_L(x))|, both directions
  + w_smooth · L_smoothness       |∂d̃| · exp(−|∂I|), d̃ = d / (mean d + ε)
  + w_low    · (photometric + smoothness on the soft-argmin output)
  + w_pseudo · ramp · L_pseudo    EMA teacher, filtered, detached
  + w_conf   · L_confidence       label-free matchability target
```

No `L_GT`, no `L_supervised_disparity`, no `L_supervised_depth`.

Two choices that resist degenerate solutions, and the reasoning:

* **The photometric term is not masked by the occlusion mask.** It is a masked mean, so a
  model that made its two disparity maps disagree everywhere would drive that mask — and
  the loss — to zero. Occlusions are handled by the left–right consistency term instead
  (as in Monodepth), and the photometric mask contains only things the network cannot
  manipulate: valid warp and image border.
* **The low-resolution terms exist because the cost volume is where matchability lives.**
  Without a direct gradient at the soft-argmin output, the refinement network learns to
  ignore a bad coarse input rather than fix it.

## 6. Teacher–student self-training

`θ_t ← m·θ_t + (1−m)·θ_s`, including BatchNorm buffers. The teacher is built with
`requires_grad_(False)`, updated under `no_grad`, and run under `no_grad`.

A teacher pixel becomes a pseudo-label only if **all** hold:
`exp(matchability) ≥ 0.5`, left–right agreement `< 1 px`, photometric residual `< 0.15`,
valid warp, and disparity inside the search range. The mask is computed **entirely from
teacher outputs**, which is what stops the student from shrinking the mask to shrink the
loss — it can only influence it through the EMA, with a `1/(1−m)`-step lag.

Anti-collapse measures, all implemented: warm-up before the teacher starts, linear ramp of
the pseudo-label weight, EMA rather than a hard copy, five-way reliability filter,
`pseudo_valid_ratio` logged every iteration, and warnings when coverage leaves
`[0.05, 0.98]` or mean confidence exceeds 0.99.

**Tested:** teacher starts as an exact copy; EMA arithmetic on weights and buffers; teacher
receives no gradient while the student does; targets and masks are detached; masked pixels
receive zero gradient; the ramp schedule; each filter criterion rejects independently; the
collapse warning fires (observed in a real CLI run).

## 7. Matchability without ground truth

Matchability is parameter-free, so it is shaped only by whatever shapes the cost volume.
The paper shapes it with the Noise-Sampling Cross-Entropy loss, which peaks the cost curve
**at the ground-truth disparity** — unavailable by construction here.

Substitute: a binary label-free reliability target `r` (left–right agreement ∧ valid warp ∧
low photometric residual), regressed onto `exp(matchability)` with BCE, weight 0.05,
enabled only after warm-up, detached target, `area` interpolation to cost-volume resolution
so the target is the *fraction* of reliable pixels in each cell.

**Honest limitation.** This is a new loss, not the paper's. It is self-referential — the
target is a function of the model's own outputs. Two things argue against the trivial
"confident everywhere" solution: the photometric term in `r` is not satisfiable by an
arbitrary self-consistent disparity field in textured regions, and the loss is off during
warm-up. Neither is a proof. **Its effect on disparity accuracy is Not measured** (§10). It
is weighted low and `loss.confidence: 0.0` disables it while keeping the architectural
component intact.

## 8. Dataset strategy

| Mode | Returns | Reads ground truth |
|---|---|---|
| `TRAIN` | `left`, `right`, `left_clean`, `right_clean`, `metadata` | never |
| `VALIDATION` | same, unaugmented | never |
| `BENCHMARK` | adds `disparity_gt`, `valid_gt_mask`, `nonocc_mask` | yes |

Scene Flow, KITTI 2012/2015, Middlebury and ETH3D all contribute **images only** to
training and ground truth only to evaluation. Benchmark datasets additionally refuse a
`transform`, because the protocol dictates the resolution.

Download and preparation for all five is automated in `stereo/data/download.py` with
verified URLs; see [DATASETS.md](DATASETS.md).

**Recommended sampling weights**, reasoned from size and diversity rather than copied:
Scene Flow 0.45 (~22k frames, the bulk of the signal), Middlebury 0.20 (15 scenes but real
indoor high-resolution imagery — the target domain), KITTI 2015 0.15, ETH3D 0.10, KITTI
2012 0.10. Weights are per-draw probabilities, so a 15-scene set is not drowned by a 22k
one. **Tested** that the realised draw fractions match the requested weights.

## 9. Original evaluation protocol

Determined by reading the paper together with the reference implementation's
`metrics/` and `utils.get_disparity_valid_mask`.

| | Scene Flow | Middlebury 2014 |
|---|---|---|
| Split | FlyingThings3D **TEST** | **TEST** (official leaderboard) |
| Metrics | EPE (px), % Bad(1.0) | bad 2.0, bad 4.0, avgerr, rms, A90, A95 — each nonocc/all |
| Computed on | disparity | disparity |
| Resolution | native | native |
| Model config | ndisp 256, CV downsample 4 | ndisp 512, CV downsample 8 |
| Valid mask | `1e-3 < d_gt < 251` | every finite ground-truth pixel; nonocc from `mask0nocc.png` |
| Post-processing | **off** — "only the raw output of our learned model" | **off**, same sentence |
| Scale alignment | none | none |
| Aggregation | reference code accumulates globally per pixel | Middlebury SDK scores per image |

Metric definitions implemented exactly: `bad_t` is `|err| > t` (so the reference's
`correct_t`, which is `|err| ≤ t`, is its complement — the paper's %Bad(1.0) is
`100 − correct_1.0`); `avgerr` is the mean absolute error; `rms` is the root mean square
error; `A90`/`A95` are error percentiles; KITTI `D1` requires `|err| > 3` **and**
`|err|/d_gt > 0.05`. **Tested** against hand-computed values.

Both aggregations are always computed and labelled (`global_*` vs `image_*`), because they
differ — **tested** with a deliberately constructed case where they disagree.

Post-processing, when requested, is the paper's: `exp(matchability) ≥ 0.25` and a connected
depth region `≥ 2000 px`. The region-similarity tolerance is **not specified by the paper**;
it is an explicit, configurable assumption (default 1 px). Post-processed runs report
`ground_truth_pixel_coverage`, since a metric on 40% of the pixels is not comparable to one
on all of them.

The paper reports no confidence benchmark, so confidence metrics (sparsification AUC, ROC
AUC) are labelled secondary and reproduce no original protocol.

### Ground-truth depth evaluation

Predicted disparity → depth via `Z = fB/(d + doffs)` (the `doffs` term is Middlebury's own
formula), then compared only where trusted ground truth exists. Sparse LiDAR/laser ground
truth is **never interpolated** to manufacture evaluation points. Degenerate disparity
(zero, negative, NaN, Inf) is handled explicitly and reported through a mask rather than
producing non-finite depth — **tested**.

**No scale alignment is applied.** `median_scale_factor()` exists for protocols that mandate
it, is never called by default, and the applied value is always recorded.
**Tested** that a uniform factor-2 depth error scores `AbsRel = 0.5` unaligned and `0.0`
aligned — i.e. that alignment would indeed hide the error.

## 10. Benchmark results

**No accuracy numbers were measured.** This environment has no GPU and none of the
datasets; training the model to convergence on Scene Flow is a multi-GPU-day job.
Per the project's rules, nothing is fabricated.

### Scene Flow FlyingThings3D, test split — protocol `sceneflow`

| Model | Training | EPE (px) | % Bad(1.0) | Status |
|---|---|---:|---:|---|
| Original paper (Table IV) | supervised, GT disparity | 0.936 | 10.0 | **Published** |
| Reference `mmstereo` checkpoint | supervised | — | — | **Not available** (the repository ships no weights) |
| This implementation | label-free self-supervised | — | — | **Not measured** (no GPU/data here) |
| This implementation + post-processing | label-free self-supervised | — | — | **Not measured** |

### Middlebury 2014 — protocol `middlebury2014`

| Model | Split | bad2.0 nocc/all | avgerr nocc/all | Status |
|---|---|---|---|---|
| Original paper (Table V) | **test** | 12.7 / 17.4 | 3.6 / 5.77 | **Published** |
| This implementation | **training** | — | — | **Not measured** |

The paper's full published row, for reference: bad2.0 12.7/17.4, bad4.0 7.26/11.0,
avgerr 3.6/5.77, rms 14.2/19.7, A90 3.82/14.5, A95 21.0/35.5.

**These two rows would not be comparable even once measured.** The paper's is the hidden
TEST split scored by the Middlebury server; only the 15-scene TRAINING split has public
ground truth. The protocol records `split="training"` and the summary attaches the caveat.
The leaderboard also aggregates with per-image weights published on the website and absent
from the SDK, so neither of the two aggregations computed here is the leaderboard's number.

### To produce the missing rows

```bash
python -m stereo.data.download sceneflow            # 45 GB images + 87 GB disparity
python train.py --config configs/train_unlabeled.yaml
python evaluate.py --checkpoint outputs/train_unlabeled/best.pt \
                   --protocol sceneflow --dataset-root datasets/sceneflow --both
```

The `mmstereo` row needs the reference repository trained from scratch on Scene Flow
**with** ground truth (`python train.py --config config_sceneflow.yaml`, 20 epochs,
batch 16 at 896×480 — the README targets a 24 GB Titan RTX), then evaluated through
`evaluate.py` here so both rows share one metric implementation.

### Known differences to state in any eventual comparison

Training data (the paper mixes custom synthetic, commercial synthetic, real captures and
Middlebury repeated 100×/epoch; none of that is available), supervision (GT regression +
NSCE vs photometric), training length (the paper: 1000 pretraining + 200 fine-tuning
epochs), crop-based augmentation (the paper crops 1440×896; this does not crop at all),
`num_disparities` (see §12), channel order, and — for Middlebury — the split.

### Ablations

**Not run.** Configurations A–F (photometric only → + smoothness → + LR consistency →
+ teacher → + adaptation → + post-processing) are each reachable by setting the
corresponding weight to zero in `configs/train_unlabeled.yaml`, but running them requires
the compute this environment lacks. Had they been run, the GT results would be development
measurements and would have to be declared as such, with a held-out split kept for the
final number.

## 11. Arbitrary resolution

Inputs are padded **right and bottom** to a multiple of 16 and cropped back. Right/bottom
padding preserves every pixel's x coordinate, so disparity is untouched; left padding would
shift x and corrupt disparity silently.

**Tested** at 224×224, 320×240, 384×384, 640×192, 1242×375, 960×540 and 150×100 (the last
three requiring padding). A Scene Flow-native 960×540 forward pass runs in 4.3 s on this
CPU and returns finite disparity at the exact input shape.

## 12. Dynamic disparity range — and where it diverges from the paper

`compute_num_disparities(width, downsample, cap=384) = min(width // 2, 384)`, floored to a
multiple of `downsample`. **Tested**: 224→112, 512→256, 640→320, 1024→384, 1920→384, and
the alignment property over 280 width/downsample combinations.

This is explicitly a **new implementation choice**. The paper selects per dataset by hand:
256 (Scene Flow, KITTI), 384 (the authors' camera), 512 (Middlebury).

**The rule does not reproduce those choices, and one divergence matters.** Measured:

| Dataset | Paper | Rule gives | Consequence |
|---|---:|---:|---|
| Scene Flow (960 wide) | 256 | 384 | wider search than needed; more compute, no accuracy loss |
| Authors' camera (2560) | 384 | 384 | agrees |
| Middlebury (2872) | **512** | **384** | **cannot represent disparity above 375 px** |

Middlebury disparities exceed that at full and half resolution, so the rule as specified
would cap the prediction and inflate the error. The cap is a config field precisely so the
paper's setting can be restored: `StereoNetConfig.for_width(2872, downsample=8,
max_disparities_cap=512)` gives 512. **Tested**, and stated in the config comments.

**Architectural limitation.** The aggregation stack flattens the disparity axis into
channels, so the search range is part of the weights — a trained checkpoint has one fixed
`num_disparities`, and only the *spatial* resolution is free. The original has the same
property (which is why the paper rebuilds per dataset). **Tested** by asserting a
112-disparity state dict fails to load into a 384-disparity model. Checkpoints record their
config; `build_model_from_checkpoint()` reconstructs the exact architecture.

## 13. Augmentation

**Geometric, per batch:** random scale `[0.8, 1.2]`, aspect `[0.9, 1.1]`, rounded to a
multiple of 16. Per *batch* because combining a random scale with a fixed tensor shape
otherwise needs random cropping. Intrinsics are rescaled so `depth = fB/d` survives.
**Tested** that both views get an identical resize and that `focal_length` follows.

**Photometric, per sample:** brightness/contrast/saturation/hue/gamma with **identical
parameters for both views** — independent jitter breaks brightness constancy.
`asymmetric_probability` exists for modelling real left/right gain mismatch, defaults to 0.
**Tested** that identical inputs remain identical after jitter.

**Teacher/student appearance split:** the student sees the jittered pair, the teacher the
clean pair (weak augmentation), and the photometric loss always reconstructs the clean
images — disparity is geometry and is appearance-invariant, so this is consistent. Teacher
and student share the *same* geometric transform, so no pseudo-label rescaling is needed;
`resize_disparity()` is there for the case where that changes.

**No random cropping anywhere.** Asserted by a test that the augmentation module contains
no crop at all.

## 14. Tests executed

`python -m pytest tests/ -q` → **109 passed**, in this session.

| File | Tests | Covers |
|---|---:|---|
| `test_geometry.py` | 24 | disparity sign (including that the *wrong* warp fails), sub-pixel warp, LR consistency on constant and slanted fields, resize rescaling, both `align_corners` factors, depth round-trip and degenerate disparity, dynamic rule + alignment, padding |
| `test_model.py` | 15 | forward at 7 resolutions incl. padding, bidirectional shapes, confidence bounds, cost-volume indexing, mirror-trick identity, soft-argmin peak, matchability limits, checkpoint round-trip, baked-in disparity range |
| `test_losses.py` | 11 | SSIM, photometric minimal at true disparity, warp masking, smoothness edge-awareness and normalisation escape, LR consistency, occlusion mask, pseudo-label masking/detachment, each filter criterion, confidence direction |
| `test_metrics.py` | 11 | EPE/RMS/bad-t/A90/A95/D1 against hand calculations, reference valid mask, `ignore_edge`, global vs per-image divergence, bad/correct complementarity, depth metrics, median scaling, sparse masks |
| `test_data.py` | 13 | folder dataset, calibration units, collation, weighted sampling proportions, stereo-consistent jitter, batch resize + intrinsics, no-crop assertion, PFM round-trip, config loading and rejection, shipped configs, post-processing gates |
| `test_training.py` | 9 | EMA copy/arithmetic/buffers, no teacher gradient, ramp, objective keys, objective signature, loss decreases on unlabeled data, full `Trainer` epoch with validation and checkpointing |
| `test_label_isolation.py` | 9 | mode-dependent keys, guard function, benchmark augmentation refusal, **GT corruption changes nothing**, training without any GT files, no evaluation imports, no supervised loss, static audit |
| `test_evaluation_end_to_end.py` | 6 | fixture self-consistency, perfect and biased predictors, full train→freeze→evaluate, post-processed run reporting, published-reference labelling |
| `test_middlebury_protocol.py` | 5 | PFM `inf` handling, `mask0nocc`, calibration units, nonocc/all separation, perfect predictor |
| `test_architecture_fidelity.py` | 6 | parameter counts vs the reference (total and per component), output shapes and bounds, cost-volume equivalence for both branches, documented cap divergence |

Also executed end to end in this session: `train.py` (2 epochs on synthetic data, EMA
teacher started, collapse warning fired correctly), `evaluate.py --both` (raw and
post-processed, `summary.json` + `per_image.csv` + visualisations written),
`inference.py` (disparity/confidence/depth `.npy` + visualisation),
`scripts/audit_label_leakage.py --strict` (PASS), and both branches of the notebook's
comparison-table cell. The notebook's 35 cells were parsed for syntax; the notebook has
**not** been executed on Kaggle.

## 15. Label Leakage Audit

Verified three independent ways.

**Structural.** A dataset in `TRAIN`/`VALIDATION` mode never calls `_load_ground_truth`.
`assert_label_free()` runs on every training batch and raises on any of
`disparity_gt`, `depth_gt`, `valid_gt_mask`, `disparity_gt_right`, `nonocc_mask`.

**Behavioural.** `test_corrupting_ground_truth_cannot_change_training` computes the
training loss, gradient norm, teacher disparity and pseudo-label mask for a fixed batch,
overwrites every ground-truth file on disk with `-999`, and asserts all four are unchanged
— while separately asserting the *benchmark* path does see the change, so the test cannot
pass merely because nothing reads those files. A companion test deletes the ground-truth
directory entirely and confirms training still runs.

**Static.** `scripts/audit_label_leakage.py` tokenises every Python file — AST for
docstrings, `tokenize` for comments, column-precise — so prose is counted separately from
executable code, and a dict key like `"disparity_gt"` counts as code. Result:

```
FORBIDDEN         0 in code    stereo/{model,losses,training}/, geometry.py, config.py, train.py
GUARD             5 in code    stereo/data/base.py — names the keys in order to reject them
LOADERS          15 in code    dataset modules — _load_ground_truth, BENCHMARK mode only
ALLOWED         133 in code    stereo/evaluation/, postprocess, evaluate.py, tests, docs
Evaluation imports on the training path: 0
RESULT: PASS
```

The audit runs as part of the test suite, so it cannot rot.

**Selection.** Checkpoint selection uses `val/photometric` — a label-free quantity — and the
saved checkpoint records `selection_is_label_free: True`. Ground truth influences neither
the weights nor which weights are kept.

## 16. Final scientific validation

| Question | Answer |
|---|---|
| Does it preserve the paper's architecture? | **Yes, tested** — exact parameter match with the reference, component by component |
| Can it train from random init on left/right only? | **Yes, tested** — loss decreases; a full `Trainer` epoch runs on a labels-free folder |
| Are the warp and disparity signs correct? | **Yes, tested** — including that the wrong direction demonstrably fails |
| Does the EMA teacher work? Are pseudo-labels detached and filtered? | **Yes, tested** — arithmetic, buffers, gradient isolation, each filter criterion |
| Does arbitrary resolution work? | **Yes, tested** — 7 resolutions, padded and unpadded |
| Does the width rule work? | **Yes, tested** — and its divergence from the paper's Middlebury choice is measured and documented (§12) |
| Can GT disparity and real depth be evaluated correctly? | **Yes, tested** against hand-computed metrics and synthetic scenes with known answers |
| Were the paper's exact metrics identified and reproduced? | **Yes, implemented and tested**; the published values are recorded |
| Was the new model compared under the same protocol? | **No — Not measured.** No GPU or datasets here |
| Can ground truth influence training in any code path? | **No** — verified structurally, behaviourally and statically |

## 16b. Measured: the left-right consistency weight

Reported after a first real training run produced a flat disparity map
(min 0, max 37.7, mean 33.9 px).

Two direct measurements explain it. With that run's logged values
(photo 0.1953, lr_cons 5.5470 px), the left-right term at weight 0.5 was
**93.4%** of the objective against photometric's 6.6% -- because it is a
*pixel*-scale quantity weighed against an image residual in [0, 1]. And a
constant disparity field is **exactly** left-right consistent (error 0.0000)
while a realistically structured field scores 3.08, so an oversized weight does
not merely dominate: it rewards flatness.

Normalising the term by the disparity search range fixes the scale mismatch and
makes the weight resolution-independent. It did **not** fix the collapse. A
4-weight, 3-seed sweep on a synthetic pair with known disparity (true 4 -> 16 px,
400 steps each) measured the term as monotonically harmful:

| `left_right` | correlation with truth | photometric |
|---:|---:|---:|
| **0.0** | **+0.606 ± 0.054** | **0.0438** |
| 0.25 | +0.360 ± 0.074 | 0.0448 |
| 1.0 | +0.273 ± 0.083 | 0.0480 |
| 4.0 | +0.144 ± 0.137 | 0.0507 |

The default is therefore **0.0**, set from this measurement rather than from
theory. A photometric-only control reached +0.748 correlation with the lowest
photometric loss, confirming the core self-supervised machinery works.

**Limits of this evidence.** One image pair, a deliberately small model, 400
steps. No occlusions and no generalisation pressure -- exactly the conditions
under which this regulariser would be expected to help. It may well help on real
multi-image training; there is no evidence here either way, only evidence that it
hurts on what could be measured. Left-right consistency remains implemented and
is still used as a *signal* for occlusion detection and pseudo-label filtering,
neither of which depends on the loss weight.

## 16c. Measured: the matching cost needs a spatial window

Found by downloading a real Middlebury pair and matching it directly, after
every synthetic test had passed. It is the single largest error in the project
so far, and synthetic data hid it: the shifted-noise fixtures used throughout
the test suite are *blurred* noise, which has strong local structure, so a
per-pixel comparison happens to work on them. Real photographs have flat,
repetitive regions where it does not.

`CostVolumeLoss` built its target from a **per-pixel** absolute difference. On
the real `Motorcycle` pair (true disparity 2.2–18.1 px) that target scored:

| matching window | MAE vs. ground truth |
|---|---:|
| 1x1 (what the code did) | **13.28 px** |
| *constant-prediction baseline* | *4.52 px* |
| 9x9 | **1.78 px** |

A single pixel's intensity matches equally well at dozens of disparities, so a
1x1 cost carries almost no information — it was **worse than predicting a
constant**, i.e. the signal the cost volume was being trained toward was worse
than no signal. Every classical stereo matcher aggregates over a window;
`MATCH_WINDOW = 9` is that. See `docs/figures/matching-diagnosis.png`.

### Search range: fewer disparities is better

With the window fixed, a sweep over resolution and search range on the same
pair, all errors converted to native-resolution pixels so they are comparable
(constant-prediction baseline: **14.95**):

| train size | `num_disparities` | bins spanned by true disparity | MAE (native px) |
|---|---:|---:|---:|
| 224x224 | 96 | 4.0 | 14.31 |
| 224x224 | 48 | 4.0 | 10.95 |
| 448x448 | 96 | 7.9 | 8.71 |
| 448x448 | 192 | 7.9 | 11.86 |
| 640x384 | 320 | 11.4 | 10.83 |
| 640x384 | 96 | 11.4 | 6.97 |
| **640x384** | **64** | 11.4 | **6.23** |

Two things follow, both contrary to what was assumed before measuring:

1. **Small inputs are worse, not better.** A cost-volume bin is always
   `downsample` (4) full-resolution pixels wide, so at 224 px width the entire
   true disparity range spans only 4 bins — barely distinguishable from a
   constant. Training at 224x224 to "simplify" makes the matching problem
   harder, not easier.
2. **An oversized search range costs accuracy.** At a fixed 640x384, cutting
   `num_disparities` from 320 to 64 nearly halves the error (10.83 -> 6.23),
   because every extra candidate is another chance at a spurious match. The
   `min(width // 2, 384)` policy is a safe *upper bound*, not a good default:
   it should be set from the disparity actually present in the data.

## 16d. Measured: the refinement head must start as an identity

`DisparityRefinement.out` takes `base_disparity` as one of its input channels
and its result is added back to `base_disparity`. Under the generic Kaiming init
the head therefore computes `(1 + w) * base_disparity` for a random `w` — a
global rescaling of the disparity, present before any training. Measured over
five seeds on a constant input, the refined output came out at **0.73x to 1.34x**
the coarse disparity at step 0.

The reference implementation carries the identical structure
(`hdrn_alpha_stereo.py:283-285`) and is unharmed by it because it trains the
refined output against ground-truth disparity, which pins the scale down at once.
Label-free, the only full-resolution signal is the photometric residual, which is
too weak and too non-convex to undo a global rescaling. On the real pair it did
not undo it:

| | step 0 | step 200 |
|---|---:|---:|
| coarse MAE | 37.33 | 17.36 |
| refined MAE *(before fix)* | **159.92** | **71.50** |

The refinement was multiplying the error by ~4x and the coarse improvement was
not reaching the output. `zero_init_residual()` zeroes that one layer after the
generic init, so the initial residual is exactly zero and training starts from
`refined == coarse`. Architecture and parameter count are unchanged
(5,661,646, still an exact match with `mmstereo`). Verified by
`test_refinement_starts_as_an_exact_identity` (bitwise equality across three
seeds) and `test_refinement_can_still_learn_a_nonzero_residual` (the zeroed
layer still receives gradient).

## 16e. Resolution robustness: why the search range must be a constant

`num_disparities` is a *construction-time* parameter -- the aggregation stack
flattens the disparity axis into channels -- so the `min(width // 2, 384)`
policy makes the **architecture itself** depend on the training width:

| train width | `num_disparities` | parameters |
|---:|---:|---:|
| 224 | 112 | 4,163,434 |
| 448 | 224 | 5,227,462 |
| 640 | 320 | 6,703,582 |
| 960+ | 384 | 7,976,942 |

A checkpoint trained at 640 **refuses to load** at 224:
`size mismatch for aggregation.conv2d.0.conv1.weight: [320,320,3,3] vs
[112,112,3,3]`. So "dynamic disparity" in practice means *a different,
incompatible model per resolution*, which is the opposite of one model that
runs on any input. Pinned by
`test_width_derived_range_produces_incompatible_checkpoints`.

### The invariant is `d / width`

Disparity is purely horizontal and scales **linearly** with horizontal resize,
so `d / width` is invariant under resizing while `d` is not. The architecture is
therefore specified as a search *fraction* at a declared `canonical_width`, and
any other resolution is handled by resizing to that width and scaling the answer
back (`stereo.model.predict_disparity`). Height never enters the disparity, so
it is left free; the rest of the network is fully convolutional.

Verified end to end: one model with `num_disparities=96, canonical_width=640`
runs at 640x384, 1242x375, 224x224, 960x540, 993x437 and 1920x1080 -- aspect
ratios 1.00 to 3.31 -- each returning output at the input's own resolution. The
value rescaling is exact: a stub predicting a constant 40 px at width 640
returns 20.00 / 40.00 / 80.00 / 77.62 px at widths 320 / 640 / 1280 / 1242.

**Aspect distortion is geometrically harmless.** Resizing 1242x375 to 640x384
scales disparity by 640/1242 only; the vertical factor does not enter. The
existing `BatchGeometricAugment` (scale 0.8-1.2, aspect 0.9-1.1) is what buys
robustness to the appearance change.

### Evaluation runs at the canonical width too

`benchmark.py` previously called `model.forward_left` on whatever resolution the
dataset produced. With a canonical-width model that is wrong: Middlebury's
native images are 1500+ px wide, so a range declared at 640 would cover a
different fraction of the image than the model ever saw, and the predicted
values would be in the wrong units. Evaluation now goes through
`predict_left_disparity`, which runs at the canonical width and scales back into
the benchmark image's own pixels, so metrics stay in native pixels while the
model sees what it was trained on. `test_benchmark_uses_the_canonical_width_path`
guards against reverting it.

`compute_num_disparities` and `StereoNetConfig.for_width` are retained -- the
original specification requires the `min(width // 2, 384)` rule to exist in one
place, and it is still the right *upper bound* -- but they are no longer the
default path, because the measurement in 16c shows the rule picks close to the
worst usable value.

## 16f. Measured: the soft-argmin must be restricted to the cost peak

A real cost curve is multi-modal: repeated texture and textureless regions
produce several near-equal minima. An expectation taken over the whole curve
lands *between* the modes, on a disparity no mode supports. On the real
Middlebury pair, same cost volume, 24 bins, errors in native pixels:

| estimator | MAE |
|---|---:|
| hard argmin (not differentiable) | 6.97 |
| full expectation, T=0.02 | 9.14 |
| full expectation, T=0.10 | 14.45 |
| full expectation, T=0.50 | 17.67 |
| **peak-restricted +/-2 bins, T=0.10** | **6.84** |

The full expectation is worse than a hard argmin at **every** temperature tried,
and degrades monotonically as the distribution is smoothed. Restricting the
expectation to a window around the peak is still differentiable, and still
recovers sub-bin precision, so it beats hard argmin rather than merely matching
it. `soft_argmin_window` defaults to 2; `None` restores the reference
implementation's behaviour exactly, which
`test_full_expectation_is_still_available` pins.

### It did not replicate end to end, so the default is `None`

The caveat above turned out to be the whole story. Read out of a *trained*
network's own cost volume, on the same pair, every estimator scores the same:

| read-out of the LEARNED volume | MAE |
|---|---:|
| hard argmin | 20.35 |
| window = 1 | 20.26 |
| window = 2 | 20.33 |
| window = 4 | 20.42 |
| **window = None (full expectation)** | **19.18** |

A same-seed A/B through the training loop agreed: at step 800 the window gave
coarse 20.10 / refined 19.37 against 20.19 / 16.16 for the full expectation --
better on every label-free quantity (photo 0.1421 vs 0.1533, cost-volume loss
0.726 vs 0.747) but worse on the metric that matters.

The reason is in 16h: the learned volume correlates only **+0.349** with the
photometric target it is trained toward, and its argmin agrees just **58.5%** of
the time. The estimator cannot recover a disparity the volume does not encode,
so sharpening the read-out changes nothing. `soft_argmin_window` therefore
defaults to `None`. The option and its measurements are kept because the effect
on a *correct* volume is real and large; it will matter once the volume is
fixed, and not before.

This is recorded as a change that was made, measured, and reverted.

## 16h. Where the remaining error actually is

The network converges to ~20 px coarse MAE on a single real pair after 1000
steps, while block matching on the same photometric cost gets **4.33**. Two
measurements localise the failure.

**The learned volume does not match its target.** Correlation with the
photometric cost volume is +0.349 per pixel and argmin agreement is 58.5%, so
the cost-volume loss is not succeeding at the one thing it exists to do.

**The other loss terms make it worse, not better.** Same seed, same 800 steps,
only the weights change:

| objective | coarse MAE | vs floor |
|---|---:|---:|
| cost-volume loss **only** | **14.10** | 3.3x |
| cost-volume loss x10 | 15.10 | 3.4x |
| full objective (current default) | ~20.1 | 4.6x |

Training on the cost-volume term alone is substantially *better* than the full
objective. The photometric, smoothness, low-resolution, confidence and range
terms are, in aggregate, pulling the prediction away from the truth -- and
raising the cost-volume weight does not buy the difference back.

Neither of these is fixed. They are the next thing to work on, and no accuracy
claim should be made until they are.

## 16g. Label-free calibration of the search range

Because the range must be fixed before training and both errors are costly
(16c), it is measured from the training images rather than guessed.
`calibrate_disparity_range` block-matches the two views against each other,
discards ambiguous pixels with a ratio test (best cost must beat the best cost
outside the peak by a factor of 0.8), and takes a high percentile of what
survives. It reads `left` and `right` only.

Validated against ground truth it never saw, on the real Middlebury pair:

| | estimate | truth |
|---|---:|---:|
| p99 disparity at 640px width | **48.0 px** | 50.0 px |
| recommended `num_disparities` | **56** | true max 51.7 |
| ground-truth pixels covered | **100.00%** | -- |

For comparison, the `min(width // 2, 384)` rule recommends 320 for the same
data. `test_calibration_reads_only_the_two_views` pins the label-free property
by checking that a sample carrying a ground-truth key produces an identical
answer.

## 16i. The chosen training geometry and search range

**Width is the only fixed dimension.** A horizontal resize scales disparity by
the same factor; a vertical one does not change it at all. So fixing the width
fixes what the search range means, and the height is free to follow each image's
own aspect ratio. Samples therefore leave the transform at different heights and
`collate_samples` pads them to the batch maximum **at the bottom** -- which
leaves every pixel's x coordinate, and so its disparity, untouched -- emitting a
`valid_mask` that the photometric and cost-volume terms honour. Replicated
padding matches itself perfectly at every disparity, so without the mask it
would contribute a confident, meaningless target.

This also removes a train/test mismatch that was present: inference already
preserved aspect ratio (`canonical_size`), while training squashed 1242x375
KITTI to 640x384. The network was being shown two different geometries for the
same scene.

Resulting training heights at width 640: KITTI 192, FlyingThings3D 352,
Middlebury 432, a square image 640.

**The search range is 128 at a canonical width of 640** -- 20% of image width.
Clipping is a hard failure (the model cannot represent the disparity at all)
while an oversized range is a soft cost, so the rule is: the smallest range that
covers the data.

| dataset | native width | max disparity | at width 640 |
|---|---:|---:|---:|
| Middlebury (measured, `Motorcycle`) | 741 | 51.7 | **52** |
| KITTI (published) | 1242 | ~192 | **~99** |
| FlyingThings3D (standard protocol cap) | 960 | 192 | **128** |

128 is the smallest value covering all three. What it costs, measured at
640x384 on CPU:

| `num_disparities` | params | cost volume | forward | Middlebury BM MAE |
|---:|---:|---:|---:|---:|
| 64 | 3,924,382 | 15.7 MB | 470 ms | 6.23 |
| **128 (chosen)** | **4,272,046** | **31.5 MB** | **745 ms** | **7.44** |
| 320 (what `min(width//2, 384)` gives) | 6,703,582 | 78.6 MB | 1607 ms | 10.83 |

Against the old 320 that is **36% fewer parameters, 60% less cost-volume memory,
2.2x faster and 31% more accurate** -- the lightweight/fast requirement and the
accuracy requirement point the same way, with no trade-off between them.

**Limits of this evidence.** Only the Middlebury number is measured, and on a
single pair: `vision.middlebury.edu` was unreachable from this environment, so
the multi-scene measurement was not made, and the KITTI and FlyingThings3D rows
are published figures rather than something checked here. `DISPARITY_RANGE =
"auto"` runs `calibrate_disparity_range` on the actual training mixture and
should be preferred over this default when the data is to hand.

## 17. Limitations

1. **No benchmark numbers.** Nothing about accuracy is claimed. Every component is tested;
   no accuracy result was measured.
2. **No reference checkpoint.** `mmstereo` ships no weights, so the most informative
   comparison row requires training the original from scratch with ground truth.
3. **Middlebury split mismatch is unavoidable** — the paper's row is the hidden test split.
4. **The Middlebury leaderboard's weighted average is not reproducible** from the SDK;
   two clearly-labelled aggregations are reported instead.
5. **The label-free matchability loss is unvalidated** (§7). Weighted low; disable with
   `loss.confidence: 0.0`.
6. **The post-processing region tolerance is an assumption** — the paper gives the
   confidence threshold (0.25) and the region size (2000 px) but not the similarity
   criterion.
7. **The width rule caps below Middlebury's needs** (§12). Implemented as specified, with
   the consequence measured and an escape hatch.
8. **The Kaggle notebook has not been executed on Kaggle** — cells are syntax-checked and
   the library calls they make are covered by tests.
9. **Refinement output is not clamped to the search range.** The residual is unbounded
   above, so an untrained model can emit disparities beyond `max_disparity` (observed: 472
   px from a model whose range is 59). The reference behaves the same way; training is what
   constrains it. Worth knowing when reading early-training logs.
10. **Loss weights are untuned starting points**, not results of a search.
11. **Remaining non-determinism** is documented in `stereo/utils/seed.py`: cuDNN algorithm
    selection, DataLoader worker interleaving, and AMP accumulation order.
