# Datasets: download and preparation

Every URL below was checked to resolve and serve the stated content. Sizes are the
`Content-Length` reported by the servers.

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

### Scene Flow FlyingThings3D — `python -m stereo.data.download sceneflow`
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
