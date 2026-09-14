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
