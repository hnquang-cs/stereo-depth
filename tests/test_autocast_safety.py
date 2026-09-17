"""Guard against operations torch refuses to run under mixed precision.

Some ops raise outright under autocast -- ``binary_cross_entropy`` is the one
that bit us: it ran fine in every CPU test (which use AMP off) and then killed a
Kaggle run at the exact iteration the confidence loss switched on.

That failure mode is invisible to this test suite, because the ban is raised by
CUDA autocast and there is no GPU here. So instead of trying to execute it, this
checks the source property that prevents it: every call to a banned op must sit
inside an ``autocast(..., enabled=False)`` block.
"""

import ast
import os

import pytest

#: Operations torch refuses to autocast, or that are unsafe in half precision.
#: https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float32
AUTOCAST_BANNED = {"binary_cross_entropy", "BCELoss"}

#: Packages whose code runs inside the AMP-wrapped training step.
GUARDED_PACKAGES = ("stereo/losses", "stereo/training", "stereo/model")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _python_files():
    for package in GUARDED_PACKAGES:
        directory = os.path.join(ROOT, package)
        for name in sorted(os.listdir(directory)):
            if name.endswith(".py"):
                yield os.path.join(directory, name)


def _disables_autocast(node: ast.AST) -> bool:
    """True if this subtree contains ``autocast(..., enabled=False)``."""
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        name = inner.func.attr if isinstance(inner.func, ast.Attribute) else getattr(
            inner.func, "id", "")
        if name != "autocast":
            continue
        for keyword in inner.keywords:
            if keyword.arg == "enabled" and isinstance(keyword.value, ast.Constant) \
                    and keyword.value.value is False:
                return True
    return False


def _called_names(node: ast.AST):
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            yield inner.func.attr if isinstance(inner.func, ast.Attribute) else getattr(
                inner.func, "id", "")


@pytest.mark.parametrize("path", list(_python_files()), ids=lambda p: os.path.relpath(p, ROOT))
def test_banned_ops_are_wrapped_in_an_autocast_disabled_block(path):
    tree = ast.parse(open(path).read())

    offenders = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        used = AUTOCAST_BANNED & set(_called_names(function))
        if used and not _disables_autocast(function):
            offenders.append(f"{function.name}() calls {sorted(used)}")

    assert not offenders, (
        f"{os.path.relpath(path, ROOT)}: these run inside the AMP training step and call "
        f"an op torch refuses to autocast, without disabling autocast first:\n  "
        + "\n  ".join(offenders)
        + "\nWrap the computation in "
          "'with torch.autocast(device_type=..., enabled=False):' and cast to float32, "
          "as stereo/losses/confidence.py and stereo/model/cost_volume.py do.")


def test_the_guard_actually_catches_a_violation(tmp_path):
    """A guard that cannot fail is worthless; prove this one detects the bug."""
    bad = tmp_path / "bad.py"
    bad.write_text("import torch.nn.functional as F\n"
                   "def forward(x, t):\n"
                   "    return F.binary_cross_entropy(x, t)\n")
    tree = ast.parse(bad.read_text())
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    assert AUTOCAST_BANNED & set(_called_names(function))
    assert not _disables_autocast(function)

    good = ("import torch\nimport torch.nn.functional as F\n"
            "def forward(x, t):\n"
            "    with torch.autocast(device_type='cuda', enabled=False):\n"
            "        return F.binary_cross_entropy(x.float(), t.float())\n")
    function = next(n for n in ast.walk(ast.parse(good)) if isinstance(n, ast.FunctionDef))
    assert _disables_autocast(function)
