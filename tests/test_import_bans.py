"""Architectural import bans (ARC_BUILD.md section 1.3), enforced in CI.

A ban is (package_path_prefix, forbidden_module_prefix). Every .py file under
the package path is parsed and every import it performs is resolved to an
absolute dotted module name; if that name falls under the forbidden prefix the
build fails.

WHY a test and not a convention: convention fails under deadline pressure. A
build that fails on a forbidden import is evidence, not a promise.

Bans whose package directory does not exist yet SKIP rather than pass, so the
gap is visible in `pytest -v` output instead of silently reading as green.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# The ban list. Extended as milestones land; never relaxed.
# ---------------------------------------------------------------------------
BANS: list[tuple[str, str]] = [
    ("arc/allocator", "arc.llm_service"),
    ("arc/gate", "arc.llm_service"),
    ("arc/gate", "arc.models"),
    # The Gate is pure: no ledger, no database, no I/O on the evaluation path.
    ("arc/gate", "arc.ledger"),
    ("arc/money", "arc.llm_service"),
    ("arc/inngest_fns", "arc.channels"),
    ("arc/sentinel", "arc.channels"),
    # The anti-circularity guard, both ways. The simulated world must not know
    # about the policy that will be measured against it...
    ("arc/simulator", "arc.allocator"),
    ("arc/simulator", "arc.forecaster"),
    ("arc/simulator", "arc.gate"),
    # ...and the policy must not reach into the world it is measured in.
    # Only the evaluation harness may, and it is not on this list.
    ("arc/allocator", "arc.simulator"),
    ("arc/forecaster", "arc.simulator"),
    ("arc/sentinel", "arc.simulator"),
    ("arc/gate", "arc.simulator"),
    ("arc/conductor", "arc.simulator"),
    ("arc/channels", "arc.simulator"),
    ("arc/ingest", "arc.simulator"),
    ("arc/ledger", "arc.simulator"),
    ("arc/core", "arc.simulator"),
    ("arc/events", "arc.simulator"),
    ("arc/llm_service", "arc.simulator"),
]

SKIP_DIRS = {"__pycache__", ".venv", "node_modules", ".git", ".ruff_cache"}

DYNAMIC_IMPORTERS = {"import_module", "__import__"}


@dataclass(frozen=True)
class Violation:
    path: Path
    lineno: int
    module: str

    def __str__(self) -> str:
        rel = self.path.relative_to(REPO_ROOT).as_posix()
        return f"{rel}:{self.lineno} imports {self.module}"


def _package_of(path: Path, root: Path) -> str:
    """Dotted package containing `path` - the base for relative imports."""
    return ".".join(path.relative_to(root).parts[:-1])


def _resolve(module: str | None, level: int, package: str) -> str:
    """Resolve an ImportFrom target to an absolute dotted module name."""
    if level == 0:
        return module or ""
    parts = package.split(".") if package else []
    if level - 1 > len(parts):
        return ""  # escapes the repo root; nothing we can attribute
    base = parts[: len(parts) - (level - 1)]
    if module:
        base = base + module.split(".")
    return ".".join(base)


def _is_under(module: str, prefix: str) -> bool:
    """Prefix match on module boundaries: `arc.llm_services` is not `arc.llm_service`."""
    return module == prefix or module.startswith(prefix + ".")


def _imported_modules(tree: ast.AST, package: str) -> list[tuple[int, str]]:
    """Every absolute module name this AST imports, static or dynamic."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve(node.module, node.level, package)
            if resolved:
                found.append((node.lineno, resolved))
        elif isinstance(node, ast.Call):
            # importlib.import_module("x.y") / __import__("x.y") are imports too.
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name in DYNAMIC_IMPORTERS and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.append((node.lineno, arg.value))
    return found


def find_violations(package_dir: Path, forbidden: str, root: Path) -> list[Violation]:
    """Every import under `package_dir` that resolves beneath `forbidden`."""
    violations: list[Violation] = []
    for path in sorted(package_dir.rglob("*.py")):
        if SKIP_DIRS & set(path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package = _package_of(path, root)
        violations.extend(
            Violation(path, lineno, module)
            for lineno, module in _imported_modules(tree, package)
            if _is_under(module, forbidden)
        )
    return violations


# ---------------------------------------------------------------------------
# The guard itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("package_path", "forbidden"),
    BANS,
    ids=[f"{p}!->{f}" for p, f in BANS],
)
def test_import_ban(package_path: str, forbidden: str) -> None:
    package_dir = REPO_ROOT / package_path
    if not package_dir.is_dir():
        pytest.skip(f"{package_path} not created yet - ban is pending, not satisfied")

    violations = find_violations(package_dir, forbidden, REPO_ROOT)
    assert not violations, "\n".join(
        [f"{package_path} must not import {forbidden}:", *(f"  {v}" for v in violations)]
    )


# ---------------------------------------------------------------------------
# Tests of the detector, so the guard is not trusted on an empty tree
# ---------------------------------------------------------------------------
def _write(root: Path, rel: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


@pytest.mark.parametrize(
    "body",
    [
        "import arc.llm_service\n",
        "import arc.llm_service.client as c\n",
        "from arc.llm_service import redactor\n",
        "from arc.llm_service.redactor import Redactor\n",
        "from importlib import import_module\nm = import_module('arc.llm_service')\n",
        "m = __import__('arc.llm_service.client')\n",
    ],
    ids=["plain", "aliased", "from-pkg", "from-mod", "importlib", "dunder"],
)
def test_detector_catches_every_import_form(tmp_path: Path, body: str) -> None:
    _write(tmp_path, "arc/allocator/policy.py", body)
    found = find_violations(tmp_path / "arc/allocator", "arc.llm_service", tmp_path)
    assert len(found) == 1, f"missed a banned import in:\n{body}"


def test_detector_resolves_relative_imports(tmp_path: Path) -> None:
    """`from ..llm_service import x` must not slip past absolute matching."""
    _write(tmp_path, "arc/allocator/sub/deep.py", "from ...llm_service import redactor\n")
    found = find_violations(tmp_path / "arc/allocator", "arc.llm_service", tmp_path)
    assert [v.module for v in found] == ["arc.llm_service"]


def test_detector_respects_module_boundaries(tmp_path: Path) -> None:
    """A longer name that merely starts with the prefix is not a violation."""
    _write(tmp_path, "arc/allocator/policy.py", "import arc.llm_service_helpers\n")
    assert find_violations(tmp_path / "arc/allocator", "arc.llm_service", tmp_path) == []


def test_detector_allows_permitted_imports(tmp_path: Path) -> None:
    _write(tmp_path, "arc/allocator/policy.py", "import numpy\nfrom arc.gate import Gate\n")
    assert find_violations(tmp_path / "arc/allocator", "arc.llm_service", tmp_path) == []


def test_ban_list_targets_the_arc_package() -> None:
    """Guards against a typo silently disabling a ban forever."""
    for package_path, forbidden in BANS:
        assert package_path.startswith("arc/"), package_path
        assert forbidden.startswith("arc."), forbidden


# ---------------------------------------------------------------------------
# The call-level ban on ground truth
#
# A module ban stops `import arc.simulator`. It does not stop a `World` handed
# in as a parameter from having `counterfactual()` called on it, and that call
# is the circularity bug in its purest form: a forecaster that reads the answer
# key produces a headline number measuring nothing. So the names themselves are
# banned everywhere except the two packages entitled to them.
# ---------------------------------------------------------------------------
GROUND_TRUTH_NAMES: frozenset[str] = frozenset(
    {
        "counterfactual",  # ground-truth P(pay) under any action
        "_latent",  # the private door to LatentState
        "LatentState",  # the type itself
        "sleeping_dogs",  # ground-truth negative-uplift accounts
    }
)

# The world that owns the ground truth, and the evaluation harness that
# validates the DR estimator against it. Nothing else, ever.
GROUND_TRUTH_ALLOWED: frozenset[str] = frozenset({"simulator", "proving_ground"})

# Packages that must be checked once they exist. Listed rather than only swept
# so a milestone that has not landed yet shows as a skip instead of a silent
# absence, exactly like the module bans above.
GROUND_TRUTH_BANNED_PACKAGES: tuple[str, ...] = (
    "allocator",
    "events",
    "forecaster",
    "sentinel",
    "gate",
    "conductor",
    "channels",
    "ingest",
    "ledger",
    "core",
    "llm_service",
    "inngest_fns",
)


def ground_truth_references(path: Path) -> list[Violation]:
    """Every reference to the ground-truth surface in one file.

    Catches attribute access, a bare name pulled in by an import, the import
    itself, and `getattr(world, "counterfactual")` - because a ban that only
    matches the obvious spelling is a ban a deadline will find its way around.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in GROUND_TRUTH_NAMES:
            found.append(Violation(path, node.lineno, node.attr))
        elif isinstance(node, ast.Name) and node.id in GROUND_TRUTH_NAMES:
            found.append(Violation(path, node.lineno, node.id))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name in GROUND_TRUTH_NAMES:
                    found.append(Violation(path, node.lineno, alias.name))
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name == "getattr" and len(node.args) >= 2:
                target = node.args[1]
                if isinstance(target, ast.Constant) and target.value in GROUND_TRUTH_NAMES:
                    found.append(Violation(path, node.lineno, str(target.value)))
    return found


def _describe(violation: Violation) -> str:
    """`Violation.__str__` says "imports", which is the wrong verb here: the
    whole point of this ban is that the answer key is reached without one."""
    rel = violation.path.relative_to(REPO_ROOT).as_posix()
    return f"{rel}:{violation.lineno} reaches {violation.module}"


def _ground_truth_violations(package_dir: Path) -> list[Violation]:
    violations: list[Violation] = []
    for path in sorted(package_dir.rglob("*.py")):
        if SKIP_DIRS & set(path.parts):
            continue
        violations.extend(ground_truth_references(path))
    return violations


@pytest.mark.parametrize("package", GROUND_TRUTH_BANNED_PACKAGES)
def test_ground_truth_not_reachable_from(package: str) -> None:
    package_dir = REPO_ROOT / "arc" / package
    if not package_dir.is_dir():
        pytest.skip(f"arc/{package} not created yet - ban is pending, not satisfied")

    violations = _ground_truth_violations(package_dir)
    assert not violations, "\n".join(
        [
            f"arc/{package} must not touch simulator ground truth "
            f"({', '.join(sorted(GROUND_TRUTH_NAMES))}):",
            *(f"  {_describe(v)}" for v in violations),
        ]
    )


def test_every_package_is_covered_by_the_ground_truth_ban() -> None:
    """A package created later must not appear unguarded.

    The parameterised test above only checks a list. This sweeps whatever is
    actually on disk, so a new subpackage is caught the day it lands rather
    than the day somebody remembers to add it.
    """
    offenders: list[Violation] = []
    for child in sorted((REPO_ROOT / "arc").iterdir()):
        if not child.is_dir() or child.name in GROUND_TRUTH_ALLOWED:
            continue
        if child.name in SKIP_DIRS:
            continue
        offenders.extend(_ground_truth_violations(child))

    assert not offenders, "ground truth reached outside the harness:\n" + "\n".join(
        f"  {v}" for v in offenders
    )


@pytest.mark.parametrize(
    "body",
    [
        "def f(world, at):\n    return world.counterfactual('a', None, at)\n",
        "from arc.simulator.world import LatentState\n",
        "from arc.simulator.world import sleeping_dogs\nx = sleeping_dogs\n",
        "def f(world):\n    return world._latent('acct_1')\n",
        "def f(world):\n    return getattr(world, 'counterfactual')('a', None, None)\n",
        "def f(w):\n    fn = w.counterfactual\n    return fn\n",
    ],
    ids=["call", "import-type", "import-fn", "private-door", "getattr", "squirrelled"],
)
def test_ground_truth_detector_catches_every_route(tmp_path: Path, body: str) -> None:
    """The ban is only worth anything if it survives an evasion attempt.

    Each of these is a way a forecaster could reach the answer key without
    ever writing `import arc.simulator`, which is exactly why the module ban
    alone is not enough.
    """
    _write(tmp_path, "arc/forecaster/uplift.py", body)
    assert _ground_truth_violations(tmp_path / "arc/forecaster"), f"missed:\n{body}"


def test_ground_truth_detector_allows_the_observable_path(tmp_path: Path) -> None:
    """`observe()` and `ObservableState` are the agent's path and stay legal."""
    _write(
        tmp_path,
        "arc/forecaster/uplift.py",
        "def features(world, account_id, at):\n"
        "    obs = world.observe(account_id, at)\n"
        "    return obs.prior_bounces_90d, obs.issuer_id\n",
    )
    assert _ground_truth_violations(tmp_path / "arc/forecaster") == []


def test_ground_truth_allowlist_is_the_harness_only() -> None:
    """Guards against the allowlist quietly growing a third member."""
    assert frozenset({"simulator", "proving_ground"}) == GROUND_TRUTH_ALLOWED


# ---------------------------------------------------------------------------
# The carve-out, and the proof it did not widen
#
# `proving_ground` is entitled to ground truth: the doubly-robust estimator
# validates itself against the simulator's counterfactuals, and reporting an
# estimate without its own error is a claim rather than a measurement.
#
# THE FAILURE MODE OF AN EXEMPTION IS THAT IT COVERS MORE THAN IT WAS WRITTEN
# FOR. `test_ground_truth_allowlist_is_the_harness_only` pins the membership of
# the allowlist. These pin its EFFECT: the two packages that must never reach
# the answer key are checked to still fail on the identical body the harness is
# allowed to contain.
# ---------------------------------------------------------------------------
GROUND_TRUTH_FORBIDDEN_AFTER_CARVE_OUT: tuple[str, ...] = ("forecaster", "allocator")


@pytest.mark.parametrize("package", GROUND_TRUTH_FORBIDDEN_AFTER_CARVE_OUT)
@pytest.mark.parametrize(
    "body",
    [
        "def v(world, at):\n    return world.counterfactual('a', None, at)\n",
        "from arc.simulator.world import LatentState\n",
        "def v(world):\n    return world._latent('acct_1')\n",
    ],
    ids=["counterfactual", "latent-type", "private-door"],
)
def test_ground_truth_ban_still_fires_after_the_carve_out(
    tmp_path: Path, package: str, body: str
) -> None:
    """The exemption is for the harness alone and did not widen to the policy.

    A forecaster that reads the answer key produces a headline number
    measuring nothing, and an allocator that reads it optimises against truth
    it will not have in production. Both must still fail on the same code the
    harness may legally contain.
    """
    _write(tmp_path, f"arc/{package}/reaches.py", body)
    violations = _ground_truth_violations(tmp_path / "arc" / package)
    assert violations, (
        f"arc/{package} was allowed to reach ground truth after the "
        f"proving_ground carve-out:\n{body}"
    )


@pytest.mark.parametrize(
    "body",
    [
        "def v(world, at):\n    return world.counterfactual('a', None, at)\n",
        "from arc.simulator.world import LatentState\n",
    ],
    ids=["counterfactual", "latent-type"],
)
def test_the_harness_may_contain_exactly_what_the_policy_may_not(tmp_path: Path, body: str) -> None:
    """The other half of the carve-out: it does work where it is meant to.

    Without this the pair above would pass just as happily if the exemption
    had been deleted, and the suite would be asserting nothing about it.
    """
    _write(tmp_path, "arc/proving_ground/estimator.py", body)
    offenders = []
    for child in sorted((tmp_path / "arc").iterdir()):
        if not child.is_dir() or child.name in GROUND_TRUTH_ALLOWED:
            continue
        offenders.extend(_ground_truth_violations(child))
    assert not offenders, "the sweep flagged the harness, which is entitled to ground truth"


def test_the_two_forbidden_packages_are_swept_in_reality_not_only_in_tmp() -> None:
    """The real tree, not a fixture. The bans above are only worth their run.

    A ban demonstrated on a temporary directory proves the detector works. This
    proves it is pointed at the packages that exist on disk right now.
    """
    for package in GROUND_TRUTH_FORBIDDEN_AFTER_CARVE_OUT:
        assert package in GROUND_TRUTH_BANNED_PACKAGES, f"arc/{package} dropped off the banned list"
        assert package not in GROUND_TRUTH_ALLOWED
        directory = REPO_ROOT / "arc" / package
        assert directory.is_dir(), f"arc/{package} does not exist, so the ban is not being run"
        assert not _ground_truth_violations(directory)
