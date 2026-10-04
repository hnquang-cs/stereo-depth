"""Quantization: accuracy and latency at each precision.

The paper's system runs on a robot, so the question is accuracy per millisecond,
not accuracy alone. These tests pin the machinery and the honest reporting; the
numbers themselves are measured in the evaluation mode.
"""

import pytest
import torch

from stereo.evaluation.quantization import (DYNAMIC_TARGETS, QuantizableStereo, available_backend,
                                            build_variants, measure_latency, model_size_mb,
                                            quantize_dynamic)
from stereo.model import StereoNet, StereoNetConfig


def tiny_model():
    return StereoNet(StereoNetConfig(num_disparities=32, canonical_width=96,
                                     backbone_width=4, feature_channels=4)).eval()


def example_batch(size=(48, 96)):
    torch.manual_seed(0)
    return {"left": torch.rand(1, 3, *size), "right": torch.rand(1, 3, *size)}


def test_dynamic_quantization_reports_that_it_converts_nothing_here():
    """An honest null result. Dynamic quantization only converts Linear and RNN
    layers, and this architecture is entirely convolutional -- so it changes
    nothing, and saying so is more useful than reporting an unchanged number as
    if it were a measurement."""
    model = tiny_model()
    targets = sum(1 for module in model.modules() if isinstance(module, DYNAMIC_TARGETS))
    assert targets == 0, "architecture changed; revisit what dynamic quantization can do"

    variant = quantize_dynamic(model)
    assert "0 quantizable module" in variant.note


def test_latency_is_measured_after_warmup():
    """The first forward pays for allocation and kernel selection, which a
    deployed model pays once rather than per frame."""
    report = measure_latency(tiny_model(), example_batch(), runs=3, warmup=2)
    assert report.runs == 3
    assert report.mean_ms > 0 and report.median_ms > 0


def test_quantizable_wrapper_keeps_the_model_usable():
    """Static quantization needs quant/dequant boundaries, and wrapping must not
    change the output or hide the model's attributes."""
    model = tiny_model()
    wrapped = QuantizableStereo(model).eval()
    batch = example_batch()
    with torch.no_grad():
        plain = model.forward_left(batch["left"], batch["right"])["disparity"]
        through = wrapped.forward_left(batch["left"], batch["right"])["disparity"]
    assert torch.allclose(plain, through, atol=1e-5)
    assert wrapped.num_disparities == model.num_disparities
    assert wrapped.max_disparity == model.max_disparity


def test_build_variants_reports_a_failed_scheme_rather_than_raising():
    """A scheme the host cannot run is a result, not an error: QNNPACK on ARM
    cannot convert these convolutions, fbgemm on x86 can, and an evaluation
    should report that rather than abort."""
    model = tiny_model()
    batch = example_batch()
    variants = build_variants(model, [batch] * 2, batch, runs=2)

    names = [variant.name for variant in variants]
    assert names == ["fp32", "dynamic int8", "static int8"]
    for variant in variants:
        assert variant.note, f"{variant.name} should say what it did"
        if variant.model is None:
            # Unavailable on this host: reported, but not timed -- timing a
            # stand-in reported the fp32 model's latency under the int8 name.
            assert variant.latency is None and variant.note.startswith("unavailable")
        else:
            assert variant.latency is not None, f"{variant.name} was not timed"


def test_sizes_are_measured_from_the_serialised_weights():
    model = tiny_model()
    assert model_size_mb(model) == pytest.approx(
        sum(p.numel() for p in model.parameters()) * 4 / 1e6, rel=0.2)


@pytest.mark.skipif("fbgemm" not in torch.backends.quantized.supported_engines,
                    reason="static quantization of conv needs fbgemm (x86)")
def test_static_quantization_shrinks_the_model():
    from stereo.evaluation.quantization import quantize_static

    model = tiny_model()
    batch = example_batch()
    variant = quantize_static(model, [batch] * 2, backend="fbgemm")
    assert variant.size_mb < model_size_mb(model) * 0.6, (
        f"int8 should be roughly a quarter the size of fp32: "
        f"{variant.size_mb:.2f} MB vs {model_size_mb(model):.2f} MB")


def test_backend_selection_matches_the_host():
    assert available_backend() in torch.backends.quantized.supported_engines


def test_latency_is_timed_at_the_size_the_scored_prediction_runs():
    """Evaluation resizes to the model's width before running it; timing the
    network on a 960 x 540 input timed a size no scored prediction runs at."""
    model = tiny_model()                                  # canonical width 96
    widths = []
    original = model.forward

    def recording(left, right, **kwargs):
        widths.append(left.shape[-1])
        return original(left, right, **kwargs)

    model.forward = recording
    measure_latency(model, example_batch(size=(96, 192)), runs=2, warmup=1)
    assert set(widths) == {96}


def test_a_scheme_that_converts_but_cannot_run_is_reported_not_timed(monkeypatch):
    """Converting is not running: an int8 model can reject quantized tensors in
    an operation only a forward reaches. Report that, and score nothing with it."""
    from stereo.evaluation import quantization

    class Broken(torch.nn.Module):
        canonical_width = 96

        def forward(self, left, right, **kwargs):
            raise RuntimeError("quantized::cat is not implemented")

    monkeypatch.setattr(quantization, "quantize_static", lambda model, calibration: quantization.QuantizedModel(
        "static int8", Broken(), 1.0, note="calibrated"))
    batch = example_batch()
    static = build_variants(tiny_model(), [batch], batch, runs=1)[-1]
    assert static.name == "static int8" and static.model is None and static.latency is None
    assert static.note.startswith("unavailable: converts, but cannot run")
