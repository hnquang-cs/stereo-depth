"""EMA teacher for Stage 2 self-training.

    theta_teacher <- m * theta_teacher + (1 - m) * theta_student

The teacher never receives gradients: it is constructed with
``requires_grad_(False)``, updated in-place under ``torch.no_grad()``, and only
ever run inside ``torch.no_grad()``.  Its predictions are pseudo-labels, not
ground truth, and they are filtered before use (see
:mod:`stereo.losses.pseudo_label`).

Batch-normalisation buffers are averaged the same way as the weights so that the
teacher's normalisation statistics track the student's rather than freezing at
initialisation; integer buffers (``num_batches_tracked``) are copied.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable

import torch
import torch.nn as nn


class EmaTeacher:
    """Exponential moving average of a student model."""

    def __init__(self, student: nn.Module, decay: float = 0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.model = copy.deepcopy(student)
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def update(self, student: nn.Module, decay: float = None) -> None:
        """One EMA step.  ``decay`` overrides the configured value if given."""
        decay = self.decay if decay is None else decay
        student_params = dict(student.named_parameters())
        for name, teacher_param in self.model.named_parameters():
            teacher_param.mul_(decay).add_(student_params[name].detach(), alpha=1.0 - decay)

        student_buffers = dict(student.named_buffers())
        for name, teacher_buffer in self.model.named_buffers():
            source = student_buffers[name]
            if teacher_buffer.dtype.is_floating_point:
                teacher_buffer.mul_(decay).add_(source.detach(), alpha=1.0 - decay)
            else:
                teacher_buffer.copy_(source)

    @torch.no_grad()
    def predict(self, left: torch.Tensor, right: torch.Tensor,
                directions: Iterable[str] = ("left", "right")) -> Dict[str, Dict[str, torch.Tensor]]:
        """Teacher inference.  Every returned tensor is detached by construction."""
        self.model.eval()
        outputs = self.model(left, right, directions=directions)
        return {direction: {key: value.detach() for key, value in output.items()}
                for direction, output in outputs.items()}

    def to(self, device) -> "EmaTeacher":
        self.model.to(device)
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state)


def pseudo_label_weight(epoch: int, start_epoch: int, ramp_epochs: int) -> float:
    """Linear ramp of the pseudo-label weight, 0 before ``start_epoch``.

    Ramping rather than switching avoids a step change in the objective at the
    moment the teacher turns on, which is where self-training runs tend to
    destabilise.
    """
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return float(min(1.0, (epoch - start_epoch + 1) / ramp_epochs))
