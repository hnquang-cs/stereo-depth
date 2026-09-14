"""Named evaluation protocols.

A protocol is the complete set of choices that make two numbers comparable:
which split, at what resolution, with which valid-pixel mask, which metrics are
*primary*, and whether post-processing is on.  Comparing numbers produced under
different protocols is the single easiest way to publish a wrong table, so the
protocol is recorded in every result file.

What the original work actually reports
---------------------------------------
Reading the paper together with the reference implementation gives:

* **Scene Flow FlyingThings3D, test split** (paper Table IV): ``EPE (px)`` and
  ``% Bad (1.0)``.  The paper's configuration for this dataset is "a maximum
  disparity of 256 and a cost volume downsample of four", so the reference
  valid-pixel mask upper bound is ``(256/4 - 1) * 4 - 1 = 251``.  Reported
  result: **EPE 0.936, %Bad(1.0) 10.0**.
* **Middlebury 2014, test split** (paper Table V, via the official leaderboard):
  ``bad 2.0``, ``bad 4.0``, ``avgerr``, ``rms``, ``A90``, ``A95``, each as
  ``nonocc/all``.  Configuration: "a maximum disparity of 512 and a cost volume
  downsample-rate of eight".  Reported result: **12.7/17.4, 7.26/11.0,
  3.6/5.77, 14.2/19.7, 3.82/14.5, 21.0/35.5**.
* Both tables use "only the raw output of our learned model" -- the
  matchability-based post-processing is explicitly excluded, which is why
  ``postprocess`` defaults to ``False`` here and post-processed numbers are
  reported as a separate row.
* Metrics are computed on **disparity**, at native resolution, with no scale
  alignment anywhere.

Limits of local reproduction, stated up front
---------------------------------------------
* The Middlebury **test** ground truth is not public.  ``middlebury2014`` below
  therefore evaluates the **training** split.  Its numbers are not comparable to
  the paper's Table V row and are labelled ``split="training"`` everywhere.
* The Middlebury leaderboard aggregates with per-image weights published on the
  website and not present in the SDK.  Both an unweighted per-image mean and a
  global per-pixel mean are reported, and neither is the weighted average.
* The other methods in the paper's Table IV use their own valid-pixel masks
  (commonly ``0 < d_gt < 192``).  ``sceneflow_literature_mask`` reproduces that
  variant so the difference can be measured rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence


@dataclass
class EvaluationProtocol:
    """One fully-specified benchmark protocol."""

    name: str
    dataset_type: str
    #: Official split this protocol scores.
    split: str
    #: Metric keys promoted to the "primary" section of the report, in order.
    primary_metrics: Sequence[str]
    #: Human-readable source of the primary metric definitions.
    primary_source: str
    #: Upper bound of the valid-pixel mask, in disparity pixels. ``None`` disables it.
    max_disparity: Optional[float] = None
    #: How ``max_disparity`` was chosen, recorded in the result file.
    max_disparity_source: str = ""
    min_disparity: float = 1e-3
    #: Evaluate the occluded pixels separately using the dataset's nonocc mask.
    evaluate_nonocc: bool = False
    #: Drop pixels whose match falls outside the image (reference implementation's option).
    ignore_edge: bool = False
    #: Native resolution; ``None`` means "do not resize", which is the default everywhere.
    resize: Optional[Sequence[int]] = None
    #: Run the matchability post-processing. The paper's tables are raw output.
    postprocess: bool = False
    #: Also compute the diagnostic depth metrics when calibration is available.
    depth_metrics: bool = False
    depth_range: Sequence[float] = (1e-3, 80.0)
    #: Never true unless a benchmark mandates it; recorded in the output either way.
    median_scaling: bool = False
    notes: str = ""
    dataset_options: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _reference_max_disparity(num_disparities: int, downsample: int) -> float:
    """The reference implementation's ``max_disparity`` for the valid mask."""
    return (num_disparities // downsample - 1) * downsample - 1


PROTOCOLS: Dict[str, EvaluationProtocol] = {
    "sceneflow": EvaluationProtocol(
        name="sceneflow",
        dataset_type="sceneflow",
        split="TEST",
        primary_metrics=("global_epe", "global_bad_1"),
        primary_source="paper Table IV (EPE px, % Bad 1.0) with the reference "
                       "implementation's global per-pixel aggregation",
        max_disparity=_reference_max_disparity(256, 4),
        max_disparity_source="paper: Scene Flow uses num_disparities=256, downsample=4 -> "
                             "(256/4 - 1) * 4 - 1 = 251, the reference implementation's mask bound",
        depth_metrics=False,
        dataset_options={"split": "TEST"},
        notes="Paper reports EPE 0.936 and %Bad(1.0) 10.0 on this split.",
    ),
    "sceneflow_literature_mask": EvaluationProtocol(
        name="sceneflow_literature_mask",
        dataset_type="sceneflow",
        split="TEST",
        primary_metrics=("global_epe", "global_bad_1"),
        primary_source="same metrics as 'sceneflow' but with the d_gt < 192 mask used by "
                       "PSMNet/AANet, so the mask difference can be measured",
        max_disparity=192.0,
        max_disparity_source="common Scene Flow convention in the comparison methods of Table IV",
        dataset_options={"split": "TEST"},
        notes="Protocol variant, NOT the paper's mask. Report separately.",
    ),
    "middlebury2014": EvaluationProtocol(
        name="middlebury2014",
        dataset_type="middlebury",
        split="training",
        primary_metrics=("image_bad_2", "image_bad_4", "image_avgerr", "image_rms", "image_A90", "image_A95"),
        primary_source="Middlebury 2014 / MiddEval3 evaluation SDK, the metric set of paper Table V",
        max_disparity=None,
        max_disparity_source="Middlebury SDK scores every finite ground-truth pixel",
        evaluate_nonocc=True,
        depth_metrics=False,
        notes="The paper's Table V is the hidden TEST split via the official leaderboard. "
              "This protocol scores the public TRAINING split, so it is NOT directly "
              "comparable to the published row.",
    ),
    "eth3d": EvaluationProtocol(
        name="eth3d",
        dataset_type="eth3d",
        split="two_view_training",
        primary_metrics=("image_bad_1", "image_avgerr", "image_rms"),
        primary_source="ETH3D two-view benchmark (bad 1.0 / AvgErr); NOT a paper metric",
        max_disparity=None,
        max_disparity_source="ETH3D ground truth is sparse; every measured pixel is scored",
        evaluate_nonocc=True,
        notes="Secondary benchmark. The paper does not report ETH3D.",
    ),
    "kitti2015": EvaluationProtocol(
        name="kitti2015",
        dataset_type="kitti",
        split="training",
        primary_metrics=("image_d1", "global_epe"),
        primary_source="KITTI D1 outlier rate (>3 px and >5%); NOT a paper metric",
        max_disparity=None,
        max_disparity_source="KITTI ground truth validity comes from the 16-bit PNG (0 = no data)",
        depth_metrics=True,
        depth_range=(1e-3, 80.0),
        dataset_options={"version": "2015", "occlusion": "occ"},
        notes="Secondary benchmark. The paper reports only KITTI runtimes (Table III), no accuracy.",
    ),
    "kitti2012": EvaluationProtocol(
        name="kitti2012",
        dataset_type="kitti",
        split="training",
        primary_metrics=("image_bad_3", "global_epe"),
        primary_source="KITTI 2012 bad-3.0 outlier rate; NOT a paper metric",
        max_disparity=None,
        max_disparity_source="KITTI ground truth validity comes from the 16-bit PNG (0 = no data)",
        depth_metrics=True,
        dataset_options={"version": "2012", "occlusion": "occ"},
        notes="Secondary benchmark.",
    ),
    "custom_folder": EvaluationProtocol(
        name="custom_folder",
        dataset_type="folder",
        split="user",
        primary_metrics=("global_epe", "global_bad_1"),
        primary_source="reference-implementation metrics applied to a user-supplied dataset",
        max_disparity=None,
        max_disparity_source="user dataset; no benchmark bound",
        depth_metrics=True,
        notes="For benchmarking a custom capture that has ground truth.",
    ),
}


def get_protocol(name: str) -> EvaluationProtocol:
    if name not in PROTOCOLS:
        raise ValueError(f"unknown protocol {name!r}; known: {sorted(PROTOCOLS)}")
    return PROTOCOLS[name]


#: Published numbers, transcribed from the paper.  Nothing here was measured
#: locally; every consumer must label these rows "published".
PUBLISHED_RESULTS: Dict[str, Dict[str, object]] = {
    "sceneflow": {
        "source": "Shankar et al., Table IV, Sceneflow FlyingThings test set",
        "metrics": {"global_epe": 0.936, "global_bad_1": 10.0},
        "status": "published",
    },
    "middlebury2014_test": {
        "source": "Shankar et al., Table V, Middlebury 2014 test set (official leaderboard), nonocc/all",
        "metrics": {
            "image_bad_2_nonocc": 12.7, "image_bad_2_all": 17.4,
            "image_bad_4_nonocc": 7.26, "image_bad_4_all": 11.0,
            "image_avgerr_nonocc": 3.6, "image_avgerr_all": 5.77,
            "image_rms_nonocc": 14.2, "image_rms_all": 19.7,
            "image_A90_nonocc": 3.82, "image_A90_all": 14.5,
            "image_A95_nonocc": 21.0, "image_A95_all": 35.5,
        },
        "status": "published",
        "caveat": "TEST split (hidden ground truth). Not reproducible locally; the "
                  "'middlebury2014' protocol scores the TRAINING split instead.",
    },
}
