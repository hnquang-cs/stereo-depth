# Datasets: download and preparation

Every URL below was checked to resolve and serve the stated content. Sizes are the
`Content-Length` reported by the servers.

## Attaching on Kaggle (what the notebook does)

The notebook trains on three datasets, all **attached** as Kaggle inputs rather than
downloaded — so Kaggle's 20 GB working-directory quota never applies:

| Dataset | Kaggle | Role |
|---|---|---|
| FlyingThings3D | `kiraarsene/flying-things-3d` | **TRAIN split only**; TEST is the paper's Table IV benchmark, held out |
| KITTI Eigen split | `awsaf49/kitti-eigen-split-dataset` | training only — the paper reports no KITTI accuracy |
| Middlebury | `minhanhtruong/middleburystereodataset` | training (the paper trains on the Middlebury training set) |

```python
DATASETS = {"sceneflow": 0.50, "kitti": 0.25, "middlebury": 0.25}
ATTACHED = {
    "sceneflow":  "/kaggle/input/flying-things-3d",
    "kitti":      "/kaggle/input/kitti-eigen-split-dataset",
    "middlebury": "/kaggle/input/middleburystereodataset",
}
SCENEFLOW_SPLIT = "TRAIN"     # holds out the paper's evaluation split
```

**Layout does not matter.** Every loader searches its attached directory for the pair of
views at any nesting depth (`stereo/data/discovery.py`) and prints the real tree if it cannot:

| Dataset | Layouts handled |
|---|---|
| FlyingThings3D | `frames_finalpass\|frames_cleanpass/TRAIN/<A\|B\|C>/<scene>/left`, the flat `FlyingThings3D_subset` release, or bare `left`/`right` trees with no pass directory |
| KITTI | raw / Eigen (`<date>/<date>_drive_NNNN_sync/image_02/data`), 2015 (`image_2`), 2012 (`colored_0`) |
| Middlebury | any directory containing `im0.png` + `im1.png` |

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
