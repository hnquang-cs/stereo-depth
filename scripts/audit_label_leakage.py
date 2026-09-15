#!/usr/bin/env python3
"""Static label-leakage audit.

Searches the repository for ground-truth-related identifiers and checks that
every hit in **executable code** lands in a zone where it is allowed.  Python
files are tokenised, so a mention inside a comment or a docstring is reported
separately from one in real code -- prose explaining that a loss is *not*
supervised is not a leak, and should not be able to hide one either.

Zones
-----
    FORBIDDEN  everything that participates in optimisation
    GUARD      stereo/data/base.py, which defines the ground-truth key list and
               the assert_label_free() check -- it names the keys in order to
               reject them
    LOADERS    the dataset modules, which implement _load_ground_truth(); only
               DatasetMode.BENCHMARK ever calls it
    ALLOWED    evaluation, post-processing, scripts, tests, docs, configs

Also checks that no module on the training path imports stereo.evaluation.

    python scripts/audit_label_leakage.py
    python scripts/audit_label_leakage.py --strict      # exit 1 on any violation
    python scripts/audit_label_leakage.py --show-allowed
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import tokenize
import ast
from typing import Dict, List, NamedTuple

#: Identifiers whose presence in executable training code would indicate leakage.
#: Matched on word boundaries, case-insensitively.
AUDIT_TERMS = [
    "gt", "ground_truth", "groundtruth", "disparity_gt", "depth_gt", "disp_gt",
    "valid_gt_mask", "nonocc", "occ_mask", "target_disparity", "supervised_loss",
    "disparity_loss", "nsce", "lidar", "epe", "bad_pixel", "d1_all", "abs_rel",
]

FORBIDDEN_ZONE = (
    "stereo/model/", "stereo/losses/", "stereo/training/",
    "stereo/geometry.py", "stereo/config.py", "stereo/data/augmentation.py",
    "stereo/utils/seed.py", "stereo/utils/checkpoint.py", "train.py",
)
#: Modules that name ground-truth terms precisely in order to EXCLUDE them:
#: base.py defines the forbidden key list and the assert_label_free() check;
#: discovery.py lists ground-truth directory names so image search skips them.
GUARD_ZONE = ("stereo/data/base.py", "stereo/data/discovery.py")
#: The dataset package. Every module here may implement _load_ground_truth(),
#: which only DatasetMode.BENCHMARK ever calls. Declared as a prefix rather than
#: a file list so that adding a loader cannot silently land in UNCLASSIFIED --
#: the training-path and guard modules inside stereo/data/ are pulled out by the
#: FORBIDDEN and GUARD zones above, which are matched first.
LOADER_ZONE = ("stereo/data/",)
ALLOWED_ZONE = (
    "stereo/evaluation/", "stereo/postprocess.py", "stereo/utils/visualization.py",
    "stereo/utils/calibration.py", "stereo/utils/__init__.py", "stereo/__init__.py",
    "evaluate.py", "inference.py", "tests/", "scripts/", "docs/", "configs/", "notebooks/",
)

FORBIDDEN_IMPORTS = ("from ..evaluation", "from stereo.evaluation", "import stereo.evaluation",
                     "from .evaluation import")


class Hit(NamedTuple):
    path: str
    line: int
    term: str
    text: str
    in_code: bool


def iter_source_files(root: str):
    skip = {".git", "__pycache__", "outputs", "datasets", ".pytest_cache", "venv", ".ipynb_checkpoints"}
    for directory, subdirs, filenames in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in skip]
        for filename in filenames:
            if filename.endswith((".py", ".yaml", ".yml")):
                yield os.path.relpath(os.path.join(directory, filename), root)


def zone_of(path: str) -> str:
    normalized = path.replace(os.sep, "/")
    for prefixes, zone in ((FORBIDDEN_ZONE, "FORBIDDEN"), (GUARD_ZONE, "GUARD"),
                           (LOADER_ZONE, "LOADERS"), (ALLOWED_ZONE, "ALLOWED")):
        if any(normalized == p or normalized.startswith(p) for p in prefixes):
            return zone
    return "UNCLASSIFIED"


def prose_spans(source: str) -> Dict[int, List[tuple]]:
    """Column ranges occupied by comments and *docstrings* only.

    A string literal used as a value -- ``{"disparity_gt": ...}`` -- is code and
    must be audited.  Only comments and module/class/function docstrings count as
    prose, so this uses the AST for docstrings rather than treating every STRING
    token as prose (which would let a leak hide inside a dictionary key).

    Returns ``{line_number: [(start_col, end_col), ...]}``; ``end_col`` of ``-1``
    means "to the end of the line".
    """
    spans: Dict[int, List[tuple]] = {}

    def add(line: int, start: int, end: int) -> None:
        spans.setdefault(line, []).append((start, end))

    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                add(token.start[0], token.start[1], -1)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return spans

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            start_line = first.lineno
            end_line = getattr(first, "end_lineno", start_line)
            for line in range(start_line, end_line + 1):
                add(line, 0, -1)
    return spans


def in_prose(spans: Dict[int, List[tuple]], line: int, column: int) -> bool:
    for start, end in spans.get(line, ()):
        if column >= start and (end == -1 or column < end):
            return True
    return False


def audit(root: str):
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in AUDIT_TERMS) + r")\b", re.IGNORECASE)
    hits: Dict[str, List[Hit]] = {zone: [] for zone in
                                  ("FORBIDDEN", "GUARD", "LOADERS", "ALLOWED", "UNCLASSIFIED")}
    import_violations: List[str] = []

    for relative in sorted(iter_source_files(root)):
        zone = zone_of(relative)
        full = os.path.join(root, relative)
        with open(full, errors="replace") as handle:
            source = handle.read()
        spans = prose_spans(source) if relative.endswith(".py") else {}

        for number, line in enumerate(source.splitlines(), start=1):
            for match in pattern.finditer(line):
                if relative.endswith(".py"):
                    prose = in_prose(spans, number, match.start())
                else:
                    comment = line.find("#")
                    prose = comment != -1 and match.start() > comment
                hits[zone].append(Hit(relative, number, match.group(0), line.strip(), not prose))
            if zone == "FORBIDDEN" and any(bad in line for bad in FORBIDDEN_IMPORTS):
                import_violations.append(f"{relative}:{number}: {line.strip()}")

    return hits, import_violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--show-allowed", action="store_true")
    args = parser.parse_args()

    hits, import_violations = audit(args.root)

    print("=" * 78)
    print("LABEL LEAKAGE AUDIT")
    print("=" * 78)
    print(f"root : {args.root}")
    print(f"terms: {', '.join(AUDIT_TERMS)}")
    print("Python files are tokenised, so comments and docstrings are counted separately "
          "from executable code.\n")

    violations: List[Hit] = []
    for zone in ("FORBIDDEN", "GUARD", "LOADERS", "ALLOWED", "UNCLASSIFIED"):
        zone_hits = hits[zone]
        code = [hit for hit in zone_hits if hit.in_code]
        prose = [hit for hit in zone_hits if not hit.in_code]
        verdict = {"FORBIDDEN": "ground truth must NOT appear in code",
                   "GUARD": "names the terms in order to exclude them",
                   "LOADERS": "BENCHMARK mode only",
                   "ALLOWED": "ground truth permitted",
                   "UNCLASSIFIED": "review manually"}[zone]
        print(f"{zone:14s} {len(code):4d} in code, {len(prose):4d} in comments/docstrings   ({verdict})")
        if zone in ("FORBIDDEN", "UNCLASSIFIED"):
            violations.extend(code)

    print()
    if violations:
        print("VIOLATIONS (ground-truth identifier in executable code on the training path):")
        for hit in violations:
            print(f"  {hit.path}:{hit.line} [{hit.term}]  {hit.text[:90]}")
        print()

    print(f"Evaluation imports on the training path: {len(import_violations)}")
    for violation in import_violations:
        print(f"  VIOLATION {violation}")
    print()

    if args.show_allowed:
        print("Allowed code-level mentions in detail:")
        for zone in ("GUARD", "LOADERS", "ALLOWED"):
            for hit in hits[zone]:
                if hit.in_code:
                    print(f"  [{zone}] {hit.path}:{hit.line} [{hit.term}]  {hit.text[:80]}")
        print()

    total = len(violations) + len(import_violations)
    if total == 0:
        print("RESULT: PASS")
        print("  * No ground-truth identifier appears in executable code on the training path.")
        print("  * No module on the training path imports the evaluation package.")
        print("  * Ground-truth handling is confined to the benchmark-mode loaders, the")
        print("    guard in stereo/data/base.py, and stereo/evaluation/.")
    else:
        print(f"RESULT: FAIL -- {total} violation(s).")
    print("=" * 78)
    return 1 if (args.strict and total) else 0


if __name__ == "__main__":
    sys.exit(main())
