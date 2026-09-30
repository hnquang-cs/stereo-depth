"""Assembly of the complete label-free training objective.

    L_total = w_photo  * L_photometric          (both directions)
            + w_lr     * L_left_right
            + w_smooth * L_smoothness           (both directions)
            + w_low    * (the same two terms on the low-resolution disparity)
            + w_conf   * L_confidence           (label-free matchability target)

There is deliberately no ``L_supervised_disparity`` / ``L_supervised_depth``
term and no ground-truth tensor anywhere in this file.

Two design notes worth reading before changing anything
-------------------------------------------------------
*The photometric loss is not masked by the occlusion mask.*  It would be easy
to multiply it by the left-right agreement mask, but since the loss is a
*masked mean*, a model that makes its two disparity maps disagree everywhere
would drive that mask -- and therefore the loss -- to zero.  Following Monodepth,
occlusions are handled by the left-right consistency term instead, and the
photometric term is masked only by things the network cannot manipulate: the
valid-warp region and the image border.

The low-resolution terms exist because the soft-argmin output is where the cost
volume -- and therefore matchability -- is shaped.  Without them the only path
to the cost volume is through the refinement network, which learns to ignore a
bad coarse input rather than fix it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..config import LossWeights
from ..geometry import RESIZE_ALIGN_CORNERS
from ..losses import (ConfidenceLoss, DisparityLoss, NsceLoss, LeftRightConsistencyLoss, PhotometricLoss,SmoothnessLoss, )

@dataclass
class ObjectiveState:
    """Per-iteration schedule inputs."""
    iteration: int = 0
    epoch: int = 0
    #: 0 during the photometric warm-up, 1 afterwards.
    warmup_scale: float = 1.0

class LabelFreeObjective:
    """Computes the total loss and the log dictionary for one batch."""

    def __init__(self, weights: LossWeights,
                 downsample: int = 4,
                 lr_occlusion_threshold: float = 1.0,
                 photometric_reliability_threshold: float = 0.15):
        self.weights = weights
        self.downsample = downsample
        self.photometric_reliability_threshold = photometric_reliability_threshold
        self.photometric = PhotometricLoss()
        self.smoothness = SmoothnessLoss(normalize=True)
        self.consistency = LeftRightConsistencyLoss()
        self.confidence = ConfidenceLoss()
        self.disparity_loss = DisparityLoss()
        self.nsce = NsceLoss()
        self.lr_occlusion_threshold = lr_occlusion_threshold

    # -- individual terms --------------------------------------------------- #

    def _photometric_pair(self, outputs, left_image, right_image, key="disparity",
                          valid_mask=None):
        left_terms = self.photometric(left_image, right_image, outputs["left"][key], "left",
                                      extra_mask=valid_mask)
        right_terms = self.photometric(right_image, left_image, outputs["right"][key], "right",
                                       extra_mask=valid_mask)
        return left_terms, right_terms

    def _smoothness_pair(self, outputs, left_image, right_image, key="disparity",
                         unlabelled=None):
        """Smoothness, applied only where there is no label.

        Smoothness is a prior standing in for supervision: it says "disparity
        varies slowly except at image edges" because nothing else constrains the
        field. Where ground truth exists, it constrains it far better, and the
        prior only biases the answer -- measurably so, since a flat field is
        exactly what smoothness rewards. So the two are complementary rather than
        additive: smooth-L1 where there is a label, smoothness where there is not.

        ``unlabelled`` is 1 on unlabelled pixels. It comes from the LEFT view's
        mask and is applied to both, which is approximate for the right view by up
        to one disparity -- immaterial for a gate that only marks which regions
        are supervised.
        """
        loss_left = self.smoothness(outputs["left"][key], left_image, unlabelled)
        loss_right = self.smoothness(outputs["right"][key], right_image, unlabelled)
        return 0.5 * (loss_left + loss_right)

    @staticmethod
    def _resample_mask(valid_mask, like):
        """Nearest-resample a ``(B, 1, H, W)`` mask onto another map's grid."""
        if valid_mask is None or valid_mask.shape[-2:] == like.shape[-2:]:
            return valid_mask
        return torch.nn.functional.interpolate(valid_mask, size=like.shape[-2:], mode="nearest")

    # -- full objective ----------------------------------------------------- #

    def __call__(self,
                 student_outputs: Dict[str, Dict[str, torch.Tensor]],
                 images: Dict[str, torch.Tensor],
                 state: ObjectiveState,
                 max_disparity: float = 1e9,
                 valid_mask: Optional[torch.Tensor] = None,
                 labels: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """Args:
            student_outputs: ``{"left": {...}, "right": {...}}`` from the student.
            images: ``{"left", "right"}`` -- the **clean** (un-jittered) pair,
                which is what the photometric term reconstructs.
            state: schedule values for this iteration.
            max_disparity: model's search-range bound, used to reject
                out-of-range predictions.
            labels: ``{"disparity_gt", "valid_gt_mask"}`` when any sample in the
                batch carries a label. Samples without one have an all-zero mask,
                so the supervised terms ignore them and no branching is needed.
            valid_mask: ``(B, 1, H, W)``, 1 on real pixels and 0 on the rows the
                collate padded a ragged (aspect-preserving) batch with. Without
                it the padded rows are free to match each other perfectly, which
                would reward the network for whatever it does there.

        Returns:
            ``{"loss": scalar, "logs": {...}, "aux": {...}}``.
        """
        left_image, right_image = images["left"], images["right"]
        logs: Dict[str, float] = {}
        #: term -> its WEIGHTED contribution to the total. Raw values answer "how
        #: big is this residual"; these answer "what is the optimiser actually
        #: following", which is the question that kept coming up.
        parts: Dict[str, float] = {}

        # ---- photometric reconstruction, full resolution ------------------ #
        photo_left, photo_right = self._photometric_pair(student_outputs, left_image, right_image,
                                                         valid_mask=valid_mask)
        photometric_loss = 0.5 * (photo_left["loss"] + photo_right["loss"])
        logs["photometric"] = float(photometric_loss.detach())
        logs["photometric_l1"] = float(0.5 * (photo_left["l1"] + photo_right["l1"]).detach())
        logs["photometric_ssim"] = float(0.5 * (photo_left["ssim"] + photo_right["ssim"]).detach())
        logs["valid_warp_ratio"] = float(photo_left["valid_warp"].mean().detach())

        # ---- which pixels carry a label ----------------------------------- #
        # Computed here because smoothness is gated by it: the prior applies only
        # where supervision does not.
        label_mask = None
        if labels is not None and self.weights.uses_labels_for(labels):
            label_mask = labels.get("valid_gt_mask")
            if label_mask is None:
                label_mask = torch.ones_like(labels["disparity_gt"])
            if valid_mask is not None:
                label_mask = label_mask * valid_mask
            logs["labelled_ratio"] = float(label_mask.mean())

        unlabelled = None
        if label_mask is not None:
            unlabelled = 1.0 - label_mask
            if valid_mask is not None:
                unlabelled = unlabelled * valid_mask
        elif valid_mask is not None:
            unlabelled = valid_mask

        # ---- edge-aware smoothness, where there is no label ---------------- #
        smoothness_loss = self._smoothness_pair(student_outputs, left_image, right_image,
                                                unlabelled=unlabelled)
        logs["smoothness"] = float(smoothness_loss.detach())

        # ---- left-right consistency --------------------------------------- #
        # Normalised by the search range before weighting. The raw term is in
        # PIXELS while the photometric term is an image residual in [0, 1], so
        # weighting them against each other directly compares quantities two
        # orders of magnitude apart -- and because a *constant* disparity field
        # is perfectly left-right consistent (error exactly 0) while any real
        # structured field is not, an oversized weight here does not merely
        # dominate: it actively rewards collapsing the prediction to a constant.
        # Monodepth avoids this by construction, its disparity being a fraction
        # of image width; normalising restores that scale-free behaviour and
        # makes the weight independent of resolution and disparity range.
        consistency = self.consistency(student_outputs["left"]["disparity"],
                                       student_outputs["right"]["disparity"])
        logs["left_right"] = float(consistency["loss"].detach())
        consistency_normalised = consistency["loss"] / max(max_disparity, 1.0)

        total = (self.weights.photometric * photometric_loss
                 + self.weights.smoothness * smoothness_loss
                 + state.warmup_scale * self.weights.left_right * consistency_normalised)
        parts["photo"] = float((self.weights.photometric * photometric_loss).detach())
        parts["smooth"] = float((self.weights.smoothness * smoothness_loss).detach())
        parts["lr"] = float((state.warmup_scale * self.weights.left_right
                             * consistency_normalised).detach())

        # ---- the same two terms at cost-volume resolution ----------------- #
        if self.weights.low_resolution > 0.0:
            small_size = student_outputs["left"]["disparity_small"].shape[-2:]
            small_left = F.interpolate(left_image, size=small_size, mode="bilinear",
                                       align_corners=RESIZE_ALIGN_CORNERS)
            small_right = F.interpolate(right_image, size=small_size, mode="bilinear",
                                        align_corners=RESIZE_ALIGN_CORNERS)
            small_photo_left, small_photo_right = self._photometric_pair(
                student_outputs, small_left, small_right, key="disparity_small")
            small_photometric = 0.5 * (small_photo_left["loss"] + small_photo_right["loss"])
            small_smoothness = self._smoothness_pair(student_outputs, small_left, small_right,
                                                     key="disparity_small")
            logs["photometric_small"] = float(small_photometric.detach())
            parts["lowres"] = 0.0     # filled below
            total = total + self.weights.low_resolution * (
                self.weights.photometric * small_photometric + self.weights.smoothness * small_smoothness)

        # ---- shape the cost volume from photometric evidence --------------- #
        # ---- supervised, on whichever samples carry a label ------------------ #
        if labels is not None and (self.weights.supervised > 0.0 or self.weights.nsce > 0.0):
            target = labels["disparity_gt"]
            mask = labels.get("valid_gt_mask")
            mask = torch.ones_like(target) if mask is None else mask
            if valid_mask is not None:
                mask = mask * valid_mask                 # exclude batch padding
            logs["labelled_ratio"] = float(mask.mean())

            if self.weights.supervised > 0.0:
                terms = self.disparity_loss(student_outputs["left"]["disparity"], target, mask)
                # Normalised by the search range before weighting, exactly as
                # left_right and range_penalty are. Smooth-L1 is a PIXEL quantity
                # and the photometric term is an image residual in [0, 1], so
                # weighting them against each other directly compares numbers two
                # orders of magnitude apart: measured at width 320, an unnormalised
                # smooth-L1 was 99% of the objective and the photometric term got
                # so little gradient that its SSIM did not move at all.
                # Normalising also makes the weight independent of TRAIN_WIDTH and
                # DISPARITY_RANGE, so changing either does not silently rebalance
                # the objective.
                contribution = self.weights.supervised * terms["loss"] / max(max_disparity, 1.0)
                total = total + contribution
                parts["sL1"] = float(contribution.detach())
                logs["supervised"] = float(terms["loss"].detach())
                logs["epe"] = float(terms["epe"])
            if self.weights.nsce > 0.0 and "cost" in student_outputs["left"]:
                terms = self.nsce(student_outputs["left"]["cost"], target, mask, self.downsample)
                total = total + self.weights.nsce * terms["loss"]
                parts["nsce"] = float((self.weights.nsce * terms["loss"]).detach())
                logs["nsce"] = float(terms["loss"].detach())
                logs["nsce_in_range"] = float(terms["in_range_ratio"])

        # ---- label-free matchability supervision -------------------------- #
        if self.weights.confidence > 0.0 and state.warmup_scale > 0.0:
            reliability = _reliability_target(
                student_outputs["left"]["disparity"].detach(),
                student_outputs["right"]["disparity"].detach(),
                photo_left["residual"].detach(),
                photo_left["mask"].detach(),
                self.lr_occlusion_threshold,
                self.photometric_reliability_threshold)
            confidence_terms = self.confidence(student_outputs["left"]["matchability"], reliability)
            confidence_contribution = (state.warmup_scale * self.weights.confidence
                                       * confidence_terms["loss"])
            total = total + confidence_contribution
            parts["conf"] = float(confidence_contribution.detach())
            logs["confidence_loss"] = float(confidence_terms["loss"].detach())
            logs["mean_confidence"] = float(confidence_terms["mean_confidence"])
            logs["mean_reliability"] = float(confidence_terms["mean_reliability"])

        # ---- keep predictions inside the cost volume's search range --------- #
        if self.weights.range_penalty > 0.0:
            excess = sum(
                torch.clamp(student_outputs[d]["disparity"] - max_disparity, min=0.0).mean()
                for d in student_outputs) / max(len(student_outputs), 1)
            # Normalised by the range so the term is scale-free across resolutions.
            range_loss = excess / max(max_disparity, 1.0)
            total = total + self.weights.range_penalty * range_loss
            parts["range"] = float((self.weights.range_penalty * range_loss).detach())
            logs["range_penalty"] = float(range_loss.detach())

        # How far the refinement moves the prediction away from the cost volume's
        # own estimate. The cost volume is the only part that does actual stereo
        # MATCHING; the refinement is a 2D network over the reference image, so if
        # it overrides the coarse estimate wholesale the model is regressing
        # disparity from appearance rather than matching, which reconstructs well
        # while being geometrically wrong.
        with torch.no_grad():
            small = student_outputs["left"]["disparity_small"]
            full = student_outputs["left"]["disparity"]
            scale = full.shape[-1] / small.shape[-1]
            base = F.interpolate(small, size=full.shape[-2:], mode="bilinear",
                                 align_corners=RESIZE_ALIGN_CORNERS) * scale
            logs["cost_volume_mean"] = float(base.mean())
            logs["refine_delta"] = float((full - base).abs().mean())

        disparity = student_outputs["left"]["disparity"].detach()
        logs["disparity_mean"] = float(disparity.mean())
        logs["disparity_min"] = float(disparity.min())
        logs["disparity_max"] = float(disparity.max())
        logs["total"] = float(total.detach())

        logs.update({f"part/{name}": value for name, value in parts.items()})
        return {"loss": total, "logs": logs, "parts": parts,
                "aux": {"photometric_left": photo_left, "consistency": consistency}}


def _reliability_target(disparity_left, disparity_right, photometric_residual, warp_mask,
                        lr_threshold: float, photometric_threshold: float) -> torch.Tensor:
    """Label-free binary reliability used as the matchability target.

    A pixel is reliable when its two disparity estimates agree geometrically,
    its warp landed inside the image, and its photometric residual is small.
    Built from images and model outputs only -- never from ground truth.
    """
    from ..losses import occlusion_mask
    agree = occlusion_mask(disparity_left, disparity_right, "left", lr_threshold)
    photometric_ok = (photometric_residual < photometric_threshold).to(agree.dtype)
    return agree * photometric_ok * warp_mask
