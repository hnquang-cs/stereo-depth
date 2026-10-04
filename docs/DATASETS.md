# Datasets: download and preparation

Every URL below was checked to resolve and serve the stated content. Sizes are the
`Content-Length` reported by the servers.

## Attaching on Kaggle (what the notebook does)

Training is supervised, so the notebook trains only on labelled data, all of it **attached**
as Kaggle inputs. Nothing is downloaded in the GPU session, where a download would burn GPU
quota for as long as it took:

| Dataset | Source | Share | Role |
|---|---|---|---|
| FlyingThings3D | the prepared dataset (official), else `kiraarsene/flying-things-3d` | 55% | **TRAIN split only**; TEST is the paper's Table IV benchmark, held out |
| Monkaa, Driving | the prepared dataset (official) | 5% + 5% | Scene Flow's other two subsets |
| Middlebury | the prepared dataset | 25% | every release with public ground truth |
| KITTI 2015, 2012 | the prepared dataset | 5% + 5% | the 200 + 194 labelled training pairs |

```python
DATASETS = {"sceneflow": 0.55, "monkaa": 0.05, "driving": 0.05,
            "middlebury": 0.25, "kitti2015": 0.05, "kitti2012": 0.05}
SCENEFLOW_SPLIT = "TRAIN"     # holds out the paper's evaluation split
```

### The official Scene Flow, streamed

The `kiraarsene/flying-things-3d` mirror turned out to hold 1,500 usable FlyingThings3D frames
(its other 48,900 pairs are augmented copies whose labels fit no scale), squashed to
224 x 224. The preparation can instead stream the official release from Freiburg
(`SCENEFLOW = ("flyingthings3d", "monkaa", "driving")`, the default):

| Subset | Frames | Images (finalpass WebP) | Disparity |
|---|---|---|---|
| FlyingThings3D | 22,390 train + 4,370 test | 6.1 GB | 93.2 GB |
| Monkaa | 8,664 | 3.0 GB | 29.9 GB |
| Driving | 4,392 | 1.0 GB | 9.6 GB |

The disparity is more than a Kaggle session's disk, so the archives are read in byte ranges
a few hundred MB ahead of the reader, unpacked as they arrive, and every range is deleted
once read (`stereo/data/prepare_sceneflow.py`). Training frames are halved to 480 x 270 -- an
exact 2x, so each label is the median of the 2 x 2 block its pixel averages -- and stored as
WebP and 16-bit PNG disparity; FlyingThings3D's TEST frames stay at 960 x 540. About 13 GB,
3-4 hours. With it attached, the training notebook uses it in place of the mirror and ignores
the mirror's label corrections, and the `sceneflow` protocol scores the official TEST split.

### The prepared dataset (made once)

`notebooks/prepare_data.ipynb` runs `python -m stereo.data.prepare` on a **CPU** session
(no GPU quota): it downloads Middlebury and KITTI from their official servers, takes only
what has ground truth, shrinks the 3000 px 2014 and 1920 px 2021 scenes to at most 960 px,
checks every release by warping, and writes about 1 GB. With `kiraarsene/flying-things-3d`
attached as well, it also lists that container's 189,000 arrays -- ten minutes on Kaggle's
network mount -- and ships the listing in `index_cache/`, so a training session reads it
instead of spending GPU time rebuilding it. Saved as a private Kaggle dataset and attached,
it is found by its `stereo_data_manifest.json`. Without it, Middlebury falls
back to the `minhanhtruong/middleburystereodataset` mirror and KITTI is skipped.

KITTI's 394 pairs are all its stereo benchmarks label: one frame (`_10`) per scene, with
LiDAR ground truth (and, for 2015, fitted car models); the test sets' ground truth is
private. Its other frames carry no labels, so training does not draw them.

**Layout does not matter.** Every loader searches its attached directory for the pair of
views at any nesting depth (`stereo/data/discovery.py`) and prints the real tree if it cannot:

| Dataset | Layouts handled |
|---|---|
| FlyingThings3D | `frames_finalpass\|frames_cleanpass/TRAIN/<A\|B\|C>/<scene>/left`, the flat `FlyingThings3D_subset` release, or bare `left`/`right` trees with no pass directory |
| KITTI | raw / Eigen (`<date>/<date>_drive_NNNN_sync/image_02/data`), 2015 (`image_2`), 2012 (`colored_0`) |
| FlyingThings3D, HDF5 | a single container holding every array; chosen automatically when the attached directory holds one |
| Middlebury | every release, in any mix — see below |

### Middlebury: one loader, seven layouts

Each release names its files differently and stores disparity differently. Every rule
below was checked on downloaded data by warping the right view with the converted label
(`stereo.data.check_label_scale`); the notebook repeats that check per release on the
attached mirror and drops a release that fails it.

| Release | Views | Ground truth: left, right | To pixels |
|---|---|---|---|
| 2001 (6 scenes) | `im2.ppm`, `im6.ppm` | `disp2.pgm`, `disp6.pgm` | ÷ 8 |
| 2003 (Cones, Teddy) | `im2`, `im6` (`.png` or `.ppm`) | `disp2`, `disp6` (`.png` / `.pgm`) | × width / 1800 (quarter ÷ 4, half ÷ 2, full ÷ 1; measured, only quarter is documented) |
| 2005 (6 of 9 have GT), 2006 (21) | `view1.png`, `view5.png`, in the scene directory or `Illum1/Exp1` (2005), `Illum1/Exp2` (2006) | `disp1.png`, `disp5.png` | full ÷ 1, half ÷ 2, third ÷ 3 |
| 2014 (23 with GT), 2021 (24) | `im0.png`, `im1.png` | `disp0.pfm`, `disp1.pfm` | as is |
| MiddEval3 training | `im0.png`, `im1.png` | `disp0GT.pfm` (+ `mask0nocc.png`), `disp1GT.pfm` (archive `MiddEval3-GT1-*.zip`) | as is |

- Value 0 (PNG/PGM) or `inf` (PFM) marks unknown disparity.
- **Not disparity**, though named like it: 2014's `disp0y.pfm` (the *vertical* disparity of
  the imperfect rectification), `disp0-n.pgm` (sample count) and `disp0-sd.pfm` (standard
  deviation); 2021's `orig/disp0.pfm` (superseded).
- 2005/2006 exposures differ by up to 9×; the release default is used, never the first found.
- Skipped, with a count in the printed summary: scenes without public ground truth (the
  MiddEval3 test set; 2005 Computer, Drumsticks, Dwarves), and extra copies of a scene
  (Q/H/F sizes, 2014 perfect + imperfect, MiddEval3 and its source release). Of the copies,
  the smallest at least `TRAIN_WIDTH` wide is used.
- Not handled: the 2001 page's Tsukuba and Map, which use other file names and encodings.

### Horizontal flip and the right view's disparity

The paper flips stereo pairs horizontally. A flip must also swap the views, so the
flipped pair's left label is the old **right** view's disparity, mirrored, and its valid
mask is the right view's (the two views' unknown regions differ by up to 13% of pixels).
A labelled pair without right-view disparity is therefore never flipped. Training loads
the right view's disparity where the data has it: every Middlebury release and the
FlyingThings3D image tree (`disparity/.../right/*.pfm`). The `kiraarsene/flying-things-3d`
HDF5 container has only `disp` (the left view), so its pairs are not flipped; the
notebook prints, per dataset, whether the flip applies.

### When a mirror's labels are wrong

Every training run checks each dataset's labels by warping before it trains (section 4 of
the notebook). A dataset that fails is left out of training, and the check measures the
factor to 1% and prints the correction to set:

```python
DISPARITY_SCALE = {"sceneflow": 0.5}   # multiplies that dataset's labels
EXCLUDE         = {"sceneflow": "aug_"}  # drops pairs whose name matches
```

Pairs whose views are not vertically aligned are reported as `NOT RECTIFIED`: no label
correction can fix them, so they are dropped instead. Training and evaluation apply the same
corrections. The `kiraarsene/flying-things-3d` container's labels measured about 2x too
large for its 224 x 224 images.

### Validation

Validation scores a held-out part of each training dataset, never trained on, so the
checkpoint is chosen by generalisation rather than fit (`VAL_FRACTION` in the notebook):

- **Middlebury**: whole scene groups, about 10% of them. Copies of a scene, MiddEval3's
  variants of it (`Piano`/`PianoL`, `Motorcycle`/`MotorcycleE`, ...) and numbered siblings
  (`Cloth1-4`, 2021's `artroom1`/`artroom2`) stay on one side. A stable hash of the group
  decides, so the held-out scenes do not change between runs. The notebook lists them.
- **FlyingThings3D**: the last 2% of the TRAIN split, as one contiguous block, so a
  sequence's nearly identical neighbouring frames stay together. TEST stays untouched
  for the paper's Table IV comparison.

Each dataset is scored separately in a fixed order (`val/<dataset>/epe`), and `val/epe`
-- the checkpoint criterion -- is their mean weighted as in training.

### What the paper evaluates on

| Paper | Evaluation set | Reproducible? |
|---|---|---|
| Table IV | Scene Flow FlyingThings3D **TEST** (EPE 0.936, %Bad 10.0) | **Yes**, if the mirror ships TEST + disparity |
| Table V | Middlebury 2014 **TEST** (bad2.0 12.7/17.4) | **No, for anyone** — GT is held by the Middlebury server; results exist only via online submission |

KITTI raw carries **no disparity ground truth** (the Eigen protocol scores *depth* against
projected LiDAR, a different benchmark), so `KittiStereoDataset` refuses `BENCHMARK` mode on
it and explains why.

Because the paper *trains* on the Middlebury training set, training on it here follows the
paper — but that is Middlebury's only publicly-labelled split, so it then cannot double as a
local benchmark. Set `DATASETS["middlebury"] = 0` to hold it out instead.

## Downloading (outside Kaggle)

```bash
python -m stereo.data.download --list             # describe everything
python -m stereo.data.download middlebury         # fetch one
python -m stereo.data.download --all --dry-run    # print commands without running
python -m stereo.data.download --verify           # check what is already prepared
```

Downloads are resumable (`curl -C -`), archives are cached under `datasets/_archives/`,
and `--delete-archives` removes them after extraction.

| Dataset | Auto | Size | Login | Extra tool |
|---|---|---:|---|---|
| Middlebury 2014 (MiddEval3) | yes | 163 MB | no | – |
| ETH3D two-view | yes | 1.1 GB | no | `7z` |
| KITTI 2015 | yes | 1.7 GB | no | – |
| KITTI 2012 | yes | 2.0 GB | no | – |
| Scene Flow FlyingThings3D | yes, but 138 GB | 45 GB + 93 GB | no | – |

On Kaggle, **attach Scene Flow rather than downloading it** — see *Scene Flow on Kaggle* below.

## What each download gives you

### Middlebury 2014 — `python -m stereo.data.download middlebury`
```
https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip   (105 MB)
https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-H.zip    ( 51 MB)
```
Extracts to `datasets/middlebury/MiddEval3/{trainingH,testH}/<scene>/` with
`im0.png`, `im1.png`, `disp0GT.pfm`, `mask0nocc.png`, `calib.txt`.

Swap `-H` for `-F` (full) or `-Q` (quarter) in both URLs for other resolutions.

**The paper's Table V is the TEST split**, whose ground truth is held by the Middlebury
server. Only `trainingH` (15 scenes) can be scored locally, and the evaluation output
labels it `split="training"` so it is never confused with the published row.

### ETH3D — `python -m stereo.data.download eth3d`
```
https://www.eth3d.net/data/two_view_training.7z      (~1.1 GB)
https://www.eth3d.net/data/two_view_training_gt.7z   (  14 MB)
```
Needs 7z: `brew install p7zip` (macOS) or `apt-get install p7zip-full` (Debian/Ubuntu).
Both archives extract into the same `two_view_training/` tree, which uses Middlebury
file names. Ground truth is sparse laser scan data, so most pixels are invalid and the
`valid_gt_mask` matters.

### KITTI 2015 — `python -m stereo.data.download kitti2015`
```
https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip        (1.6 GB)
https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow_calib.zip  (1.6 MB)
```
Gives `training/{image_2,image_3,disp_occ_0,disp_noc_0,calib_cam_to_cam}` and a
`testing/` split with images only. The calib archive is what makes metric depth possible.

### KITTI 2012 — `python -m stereo.data.download kitti2012`
```
https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip  (1.9 GB)
```
Gives `training/{colored_0,colored_1,disp_occ,disp_noc,calib}`.

### Scene Flow on Kaggle — attach, do not download

Scene Flow is 132 GB, well past Kaggle's 20 GB working-directory quota. Attach a community
mirror as an input dataset instead and tell the notebook where it is:

```python
DATASETS = {"sceneflow": 1.0}                        # Control Panel
ATTACHED = {"sceneflow": "/kaggle/input/sceneflow"}  # path from Add Data
```

**The mirror's internal layout does not matter.** `stereo.data.sceneflow.discover_sceneflow()`
searches the attached directory up to 5 levels deep for either recognised structure:

| Layout | Looks like |
|---|---|
| `official` | `frames_finalpass/TRAIN/<A\|B\|C>/<scene>/left/*.png` + `disparity/TRAIN/...` |
| `subset` | `train/image_clean/left/*.png` + `train/disparity/left/*.pfm` |

Discovery is **structural**, not path-name matching: it locates an image-pass directory
(`frames_*`, `image_clean`, `image_final`), works out how — or whether — that tree is split,
then indexes every directory beneath it holding a `left`/`right` pair, at any depth. One code
path therefore covers official FlyingThings3D (`TRAIN/<A|B|C>/<scene>/left`), Monkaa
(`<scene>/left`), Driving (`<focallength>/<direction>/<speed>/left`), the flat subset release,
and mirrors that nest or flatten any of them.

It prefers FlyingThings3D (the paper's benchmark) and `finalpass`, falls back to
`frames_cleanpass`, and maps `TEST` onto the subset release's `val`. Override with
`SCENEFLOW_SUBSET` / `SCENEFLOW_PASS` in the notebook, or `subset=` / `pass_name=` on
`SceneFlowDataset`.

#### Worked example: `arthurthom/sceneflow`

That mirror is double-nested, cleanpass-only, has no `TRAIN` level, and bundles four datasets:

```
sceneflow/
├── FlyingThings3D/FlyingThings3D/{frames_cleanpass,disparity}/<A|B|C>/<scene>/left|right/
├── Driving/Driving/{frames_cleanpass,disparity}/<focallength>/<direction>/<speed>/left|right/
├── Monkaa/Monkaa/{frames_cleanpass,disparity}/<scene>/left|right/
└── kitti2015/{training,testing}/
```

```python
DATASETS = {"sceneflow": 1.0}
ATTACHED = {
    "sceneflow": "/kaggle/input/datasets/arthurthom/sceneflow",
    "kitti2015": "/kaggle/input/datasets/arthurthom/sceneflow/kitti2015/training",
}
```

**This mirror keeps no TRAIN/TEST division.** Training on it is fine — training is label-free
— but its "TEST" split is the same frames as TRAIN, so benchmarking it would score the model
on its own training images. `SceneFlowDataset` **refuses** `BENCHMARK` mode on an unsplit tree
rather than emit a contaminated number; benchmark on `middlebury2014`, `eth3d` or `kitti2015`
instead, which have genuine held-out splits. The escape hatch
(`allow_unsplit_benchmark=True`) exists only for a copy the model provably never saw.

An **images-only mirror is fine for training** — training here is label-free. Disparity is
only needed to benchmark, and `SceneFlowDataset` raises a clear error naming the problem if
you try to benchmark against a copy that has none.

If a mirror is not recognised, the error prints that mirror's actual directory tree so it can
be reported rather than guessed at. Outside Kaggle, the same auto-discovery works on a
manually staged copy, so `root` can point anywhere sensible.

### Scene Flow by direct download (not on Kaggle) — `python -m stereo.data.download sceneflow`
```
.../FlyingThings3D/raw_data/flyingthings3d__frames_finalpass.tar       ( 42 GB)
.../FlyingThings3D/derived_data/flyingthings3d__disparity.tar.bz2      ( 87 GB)
```
(full host: `https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/`)

**Training here needs only the images.** The 87 GB disparity archive is required *only*
to run the Table IV benchmark. If you are just training, download `frames_finalpass`
alone and skip the rest.

On Kaggle, attach an existing Scene Flow mirror as a dataset instead of downloading —
see the notebook.

## Layout the loaders expect

```
datasets/
├── sceneflow/
│   ├── frames_finalpass/TRAIN|TEST/<A|B|C>/<scene>/{left,right}/*.png
│   └── disparity/       TRAIN|TEST/<A|B|C>/<scene>/{left,right}/*.pfm
├── middlebury/MiddEval3/trainingH/<scene>/{im0.png,im1.png,disp0GT.pfm,mask0nocc.png,calib.txt}
├── eth3d/two_view_training/<scene>/{im0.png,im1.png,disp0GT.pfm,mask0nocc.png,calib.txt}
├── kitti2015/training/{image_2,image_3,disp_occ_0,disp_noc_0,calib_cam_to_cam}/
└── kitti2012/training/{colored_0,colored_1,disp_occ,disp_noc,calib}/
```

`python -m stereo.data.download --verify` checks all of this and prints the exact
`--dataset-root` / `--protocol` arguments for each dataset it finds.

## Your own camera

No preparation and no ground truth:

```
my_camera/
├── left/   000001.png ...
├── right/  000001.png ...
└── calib.txt      # optional, Middlebury syntax, only for metric depth
```

Matching filenames pair the views; unpaired files are skipped with a warning.

## A note on disparity file formats

* **PFM** (Scene Flow, Middlebury, ETH3D) stores rows bottom-to-top and uses the *sign*
  of the scale line for endianness. `stereo/data/io.py:read_pfm` flips the rows and uses
  the sign only for byte order, which yields positive left-referenced disparity.
  The reference implementation instead decodes PFM through OpenCV, which multiplies by
  the signed scale, and compensates with a negation — do not copy that negation here.
* **KITTI** stores `disparity * 256` as uint16 with 0 meaning "no ground truth".
* **Middlebury/ETH3D** mark invalid pixels as `inf`; `mask0nocc.png` is 255 for
  non-occluded valid pixels.
