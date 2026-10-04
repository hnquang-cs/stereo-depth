"""Quantize a trained model and measure what it costs.

The paper's system runs on a robot, where the question is not only "how accurate"
but "how accurate per millisecond". This module answers both, by evaluating the
same checkpoint at three precisions and reporting accuracy beside latency.

Two schemes, because they trade off differently:

**Dynamic** quantizes weights to int8 and leaves activations in float, deciding
each tensor's scale at runtime. One call, no calibration data. It only converts
Linear and LSTM layers, though, so on a network that is almost entirely
convolution it changes very little -- which is itself worth measuring rather than
assuming.

**Static** quantizes weights AND activations, using a calibration pass over real
images to fix the activation ranges ahead of time. It converts convolutions, so
it is the one that actually shrinks and accelerates this model, and the one a
deployment would use.

Both are CPU-only in PyTorch (the fbgemm/qnnpack backends), so latency is
measured on CPU for every precision to keep the comparison fair.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

#: Modules dynamic quantization converts. Listed explicitly so the report can say
#: how many of them the model actually has.
DYNAMIC_TARGETS = (nn.Linear, nn.LSTM, nn.GRU)


@dataclass
class LatencyReport:
    """Wall-clock per forward pass, in milliseconds."""
    mean_ms: float
    median_ms: float
    runs: int

    def __str__(self) -> str:
        return f"{self.mean_ms:.1f} ms mean, {self.median_ms:.1f} ms median ({self.runs} runs)"


@dataclass
class QuantizedModel:
    """A model at one precision, with its size and latency.

    ``model`` is ``None`` when the scheme could not be applied on this host; the
    note says why. Such a variant is reported, but neither timed nor scored.
    """
    name: str
    model: Optional[nn.Module]
    size_mb: float
    latency: Optional[LatencyReport] = None
    note: str = ""


def model_size_mb(model: nn.Module) -> float:
    """Serialised size, which is what a quantized model actually saves."""
    import io

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1e6


@torch.no_grad()
@torch.no_grad()
def measure_latency(model: nn.Module, example: Dict[str, torch.Tensor],
                    runs: int = 10, warmup: int = 3) -> LatencyReport:
    """Time one prediction exactly as evaluation makes it.

    That is :func:`~stereo.model.inference.predict_left_disparity`: the input
    resized to the width the model was trained at, the network run there, and
    the disparity scaled back to the input's size. Timing the network on the
    input as given would time a size it never runs at -- 960 x 540 where the
    scored prediction runs at 224 wide.

    Warm-up runs are discarded: the first forward pays for lazy allocation and
    kernel selection, which a deployed model pays once rather than per frame.
    """
    from ..model.inference import predict_left_disparity

    model.eval()
    for _ in range(warmup):
        predict_left_disparity(model, example["left"], example["right"])
    timings: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        predict_left_disparity(model, example["left"], example["right"])
        timings.append((time.perf_counter() - start) * 1000.0)
    timings.sort()
    return LatencyReport(mean_ms=sum(timings) / len(timings),
                         median_ms=timings[len(timings) // 2], runs=runs)


def quantize_dynamic(model: nn.Module) -> QuantizedModel:
    """int8 weights, float activations. No calibration needed."""
    converted = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).cpu().eval(), set(DYNAMIC_TARGETS), dtype=torch.qint8)
    targets = sum(1 for module in model.modules() if isinstance(module, DYNAMIC_TARGETS))
    note = (f"{targets} quantizable module(s); dynamic quantization converts only "
            f"Linear/RNN layers, and this model is almost entirely convolution"
            if targets < 5 else f"{targets} quantizable modules")
    return QuantizedModel("dynamic int8", converted, model_size_mb(converted), note=note)


class QuantizableStereo(nn.Module):
    """A stereo model with quantize/dequantize boundaries.

    Static post-training quantization observes activations and replaces modules
    with int8 versions, but it needs to know where float ends and int8 begins.
    Without these stubs ``prepare``/``convert`` leave the model running in float
    and the exercise measures nothing.

    Only the disparity output is dequantized; the auxiliary outputs are returned
    as the wrapped model produced them.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = model
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, left: torch.Tensor, right: torch.Tensor, **kwargs):
        """Every entry point quantizes: evaluation calls ``model(left, right,
        directions=...)``, and int8 layers given float input fail."""
        outputs = self.model(self.quant(left), self.quant(right), **kwargs)
        return {direction: {**output, "disparity": self.dequant(output["disparity"])}
                for direction, output in outputs.items()}

    def forward_left(self, left: torch.Tensor, right: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self(left, right, directions=("left",))["left"]

    def __getattr__(self, name):
        """Forward num_disparities, scale and friends to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["model"], name)


def available_backend() -> str:
    """The quantized backend this machine supports.

    fbgemm is x86, qnnpack is ARM. Picking the wrong one fails at convert time
    with "quantized engine is not supported", which is a property of the host
    rather than of the model.
    """
    supported = list(torch.backends.quantized.supported_engines)
    for candidate in ("fbgemm", "qnnpack", "x86"):
        if candidate in supported:
            return candidate
    raise RuntimeError(f"no quantized backend available; torch reports {supported}")


def quantize_static(model: nn.Module, calibration: Iterable[Dict[str, torch.Tensor]],
                    max_batches: int = 32, backend: Optional[str] = None) -> QuantizedModel:
    """int8 weights AND activations, calibrated on real images.

    Args:
        calibration: batches to run through the model so the observers see the
            activation ranges this data actually produces. A few hundred images
            is ample; the ranges are percentiles, not a fit.
    """
    backend = backend or available_backend()
    torch.backends.quantized.engine = backend
    prepared = QuantizableStereo(copy.deepcopy(model).cpu().eval()).eval()
    prepared.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    torch.ao.quantization.prepare(prepared, inplace=True)

    from ..model.inference import predict_left_disparity

    seen = 0
    with torch.no_grad():
        for batch in calibration:
            if seen >= max_batches:
                break
            # Through the scored path, so the observers see the activations of
            # the size the network actually runs at.
            predict_left_disparity(prepared, batch["left"], batch["right"])
            seen += 1
    if seen == 0:
        raise ValueError("static quantization needs calibration data and got none")

    converted = torch.ao.quantization.convert(prepared, inplace=False)
    return QuantizedModel("static int8", converted, model_size_mb(converted),
                          note=f"calibrated on {seen} batch(es), backend {backend}")


def build_variants(model: nn.Module, calibration: Iterable[Dict[str, torch.Tensor]],
                   example: Dict[str, torch.Tensor], runs: int = 10,
                   include: Iterable[str] = ("fp32", "dynamic", "static")) -> List[QuantizedModel]:
    """The model at each precision, each timed on the same example.

    A quantization scheme that fails on this architecture is reported rather than
    raised: that it cannot be applied is a result, not an error.
    """
    wanted = set(include)
    variants: List[QuantizedModel] = []
    baseline = copy.deepcopy(model).cpu().eval()
    if "fp32" in wanted:
        variants.append(QuantizedModel("fp32", baseline, model_size_mb(baseline),
                                       note="unquantized baseline"))
    if "dynamic" in wanted:
        try:
            variants.append(quantize_dynamic(baseline))
        except Exception as error:                       # pragma: no cover
            variants.append(QuantizedModel("dynamic int8", None, float("nan"),
                                           note=f"unavailable: {error}"))
    if "static" in wanted:
        try:
            variants.append(quantize_static(baseline, calibration))
        except Exception as error:
            variants.append(QuantizedModel("static int8", None, float("nan"),
                                           note=f"unavailable: {error}"))

    for variant in variants:
        if variant.model is None:
            continue                  # timing a stand-in would report the wrong model
        try:
            variant.latency = measure_latency(variant.model, example, runs=runs)
        except Exception as error:
            # Converting is not running: an operation the int8 model hands a
            # quantized tensor it does not accept fails only here.
            variant.note = f"unavailable: converts, but cannot run ({error})"
            variant.model = None
    return variants
