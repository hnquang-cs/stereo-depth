"""Assembly of the complete label-free training objective.

    L_total = w_photo  * L_photometric          (both directions)
            + w_lr     * L_left_right
            + w_smooth * L_smoothness           (both directions)
            + w_low    * (the same two terms on the low-resolution disparity)
            + w_pseudo * ramp * L_pseudo        (Stage 2 only)
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

*Pseudo-label masks come only from the teacher.*  The same escape route exists
for the pseudo-label term, and the defence is that its mask is computed
entirely from the EMA teacher's outputs under ``no_grad``.  The student cannot
shrink that mask within a step; it can only influence it through the EMA, with
a 1/(1-decay)-step lag.

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

from ..config import LossWeights, TeacherConfig
from ..geometry import RESIZE_ALIGN_CORNERS
from ..losses import (ConfidenceLoss, CostVolumeLoss, LeftRightConsistencyLoss, PhotometricLoss,
                      PseudoLabelLoss, SmoothnessLoss, build_pseudo_label_mask)


@dataclass
class ObjectiveState:
    """Per-iteration schedule inputs."""
    iteration: int = 0
    epoch: int = 0
    #: 0 during the photometric warm-up, 1 afterwards.
    warmup_scale: float = 1.0
    #: Pseudo-label ramp in [0, 1]; 0 disables the term entirely.
    pseudo_scale: float = 0.0


class LabelFreeObjective:
    """Computes the total loss and the log dictionary for one batch."""

    def __init__(self, weights: LossWeights, teacher_config: TeacherConfig,
                 lr_occlusion_threshold: float = 1.0):
        self.weights = weights
        self.teacher_config = teacher_config
        self.photometric = PhotometricLoss()
        self.smoothness = SmoothnessLoss(normalize=True)
        self.consistency = LeftRightConsistencyLoss()
        self.pseudo = PseudoLabelLoss()
        self.confidence = ConfidenceLoss()
        self.cost_volume = CostVolumeLoss()
        self.lr_occlusion_threshold = lr_occlusion_threshold

    # -- individual terms --------------------------------------------------- #

    def _photometric_pair(self, outputs, left_image, right_image, key="disparity"):
        left_terms = self.photometric(left_image, right_image, outputs["left"][key], "left")
        right_terms = self.photometric(right_image, left_image, outputs["right"][key], "right")
        return left_terms, right_terms

    def _smoothness_pair(self, outputs, left_image, right_image, key="disparity"):
        loss_left = self.smoothness(outputs["left"][key], left_image)
        loss_right = self.smoothness(outputs["right"][key], right_image)
        return 0.5 * (loss_left + loss_right)

    # -- full objective ----------------------------------------------------- #

    def __call__(self,
                 student_outputs: Dict[str, Dict[str, torch.Tensor]],
                 images: Dict[str, torch.Tensor],
                 state: ObjectiveState,
                 teacher_outputs: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
                 max_disparity: float = 1e9) -> Dict[str, Any]:
        """Args:
            student_outputs: ``{"left": {...}, "right": {...}}`` from the student.
            images: ``{"left", "right"}`` -- the **clean** (un-jittered) pair,
                which is what the photometric term reconstructs.
            state: schedule values for this iteration.
            teacher_outputs: teacher predictions, already detached, or ``None``.
            max_disparity: model's search-range bound, used to reject
                out-of-range pseudo-labels.

        Returns:
            ``{"loss": scalar, "logs": {...}, "aux": {...}}``.
        """
        left_image, right_image = images["left"], images["right"]
        logs: Dict[str, float] = {}

        # ---- photometric reconstruction, full resolution ------------------ #
        photo_left, photo_right = self._photometric_pair(student_outputs, left_image, right_image)
        photometric_loss = 0.5 * (photo_left["loss"] + photo_right["loss"])
        logs["photometric"] = float(photometric_loss.detach())
        logs["photometric_l1"] = float(0.5 * (photo_left["l1"] + photo_right["l1"]).detach())
        logs["photometric_ssim"] = float(0.5 * (photo_left["ssim"] + photo_right["ssim"]).detach())
        logs["valid_warp_ratio"] = float(photo_left["valid_warp"].mean().detach())

        # ---- edge-aware smoothness ---------------------------------------- #
        smoothness_loss = self._smoothness_pair(student_outputs, left_image, right_image)
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
            total = total + self.weights.low_resolution * (
                self.weights.photometric * small_photometric + self.weights.smoothness * small_smoothness)

        # ---- shape the cost volume from photometric evidence --------------- #
        # The cost volume is the only part that performs actual stereo matching.
        # Its gradient through soft-argmin only says "move the expected
        # disparity"; this says "the match is at index k", which is what the
        # paper's NSCE loss provides from ground truth and this derives from the
        # images alone.
        if self.weights.cost_volume > 0.0 and "cost" in student_outputs["left"]:
            cost = student_outputs["left"]["cost"]
            cost_size = cost.shape[-2:]
            cost_left = F.interpolate(left_image, size=cost_size, mode="bilinear",
                                      align_corners=RESIZE_ALIGN_CORNERS)
            cost_right = F.interpolate(right_image, size=cost_size, mode="bilinear",
                                       align_corners=RESIZE_ALIGN_CORNERS)
            cost_terms = self.cost_volume(cost, cost_left, cost_right, "left")
            total = total + self.weights.cost_volume * cost_terms["loss"]
            logs["cost_volume_loss"] = float(cost_terms["loss"].detach())
            logs["cost_target_confidence"] = float(cost_terms["target_confidence"])
            logs["cost_supervised_ratio"] = float(cost_terms["supervised_ratio"])

        # ---- label-free matchability supervision -------------------------- #
        if self.weights.confidence > 0.0 and state.warmup_scale > 0.0:
            reliability = _reliability_target(
                student_outputs["left"]["disparity"].detach(),
                student_outputs["right"]["disparity"].detach(),
                photo_left["residual"].detach(),
                photo_left["mask"].detach(),
                self.lr_occlusion_threshold,
                self.teacher_config.filter.photometric_threshold)
            confidence_terms = self.confidence(student_outputs["left"]["matchability"], reliability)
            total = total + state.warmup_scale * self.weights.confidence * confidence_terms["loss"]
            logs["confidence_loss"] = float(confidence_terms["loss"].detach())
            logs["mean_confidence"] = float(confidence_terms["mean_confidence"])
            logs["mean_reliability"] = float(confidence_terms["mean_reliability"])

        # ---- teacher pseudo-labels ---------------------------------------- #
        pseudo_valid_ratio = 0.0
        if teacher_outputs is not None and state.pseudo_scale > 0.0 and self.weights.pseudo > 0.0:
            pseudo = self._pseudo_term(student_outputs, teacher_outputs, images, max_disparity)
            # Also a pixel-scale quantity; normalised for the same reason.
            pseudo_normalised = pseudo["loss"] / max(max_disparity, 1.0)
            total = total + state.pseudo_scale * self.weights.pseudo * pseudo_normalised
            logs["pseudo"] = float(pseudo["loss"].detach())
            pseudo_valid_ratio = float(pseudo["valid_ratio"])
        logs["pseudo_valid_ratio"] = pseudo_valid_ratio
        logs["pseudo_scale"] = state.pseudo_scale

        # ---- keep predictions inside the cost volume's search range --------- #
        if self.weights.range_penalty > 0.0:
            excess = sum(
                torch.clamp(student_outputs[d]["disparity"] - max_disparity, min=0.0).mean()
                for d in student_outputs) / max(len(student_outputs), 1)
            # Normalised by the range so the term is scale-free across resolutions.
            range_loss = excess / max(max_disparity, 1.0)
            total = total + self.weights.range_penalty * range_loss
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

        return {"loss": total, "logs": logs,
                "aux": {"photometric_left": photo_left, "consistency": consistency}}

    # -- pseudo-label term -------------------------------------------------- #

    def _pseudo_term(self, student_outputs, teacher_outputs, images, max_disparity) -> Dict[str, torch.Tensor]:
        """Masked smooth-L1 against the filtered teacher disparity, both directions."""
        losses, ratios = [], []
        for direction in ("left", "right"):
            if direction not in teacher_outputs or direction not in student_outputs:
                continue
            teacher = teacher_outputs[direction]
            with torch.no_grad():
                mask = self._teacher_mask(teacher, teacher_outputs, images, direction, max_disparity)
            terms = self.pseudo(student_outputs[direction]["disparity"], teacher["disparity"], mask)
            losses.append(terms["loss"])
            ratios.append(terms["valid_ratio"])
        if not losses:
            zero = torch.zeros((), device=student_outputs["left"]["disparity"].device)
            return {"loss": zero, "valid_ratio": zero}
        return {"loss": sum(losses) / len(losses), "valid_ratio": sum(ratios) / len(ratios)}

    @torch.no_grad()
    def _teacher_mask(self, teacher, teacher_outputs, images, direction, max_disparity) -> torch.Tensor:
        """Reliability mask built purely from teacher outputs and the input images."""
        from ..geometry import warp_left_to_right, warp_right_to_left

        opposite = "right" if direction == "left" else "left"
        disparity = teacher["disparity"]

        lr_error = None
        if opposite in teacher_outputs:
            if direction == "left":
                warped, _ = warp_right_to_left(teacher_outputs[opposite]["disparity"], disparity)
            else:
                warped, _ = warp_left_to_right(teacher_outputs[opposite]["disparity"], disparity)
            lr_error = torch.abs(disparity - warped)

        target_image = images[direction]
        source_image = images[opposite]
        photo = self.photometric(target_image, source_image, disparity, direction)
        return build_pseudo_label_mask(
            disparity_teacher=disparity,
            config=self.teacher_config.filter,
            max_disparity=max_disparity,
            confidence=teacher.get("confidence"),
            lr_error=lr_error,
            photometric_residual=photo["residual"],
            valid_warp=photo["mask"])


@torch.no_grad()
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
