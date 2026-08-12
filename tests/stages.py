"""Disk-stored intermediate stage checkpoints for the heavy tests.

The problem
-----------
``pytest -m ''`` costs **3 h 23 min**, and it is dominated by four-component atomic solves that
have nothing to do with most of what is being changed: a lanthanide is 15-60 minutes, and
editing ``kuiva/ci/strings.py`` cannot possibly move it. Re-running it anyway is the single
largest waste in the development loop, and it grows every time a stage is added downstream.

So a stage's output is recorded on disk and replayed — but **only when replaying it cannot
hide the thing under test**. Two independent gates decide that, and both must open:

1. ⚠ **Has the code that produces this stage changed?** Each stage declares the ``kuiva``
   modules it is built from; the fingerprint is the transitive **import closure** of those
   modules, hashed from their *sources*. Touch ``kuiva/amf/decouple.py`` and every
   ``amf_correction`` checkpoint is stale on the next run, automatically. Touch
   ``kuiva/ci/strings.py`` and none of them are, because it is not in the closure. This is the
   whole mechanism: it is not a version number someone has to remember to bump.
2. ⚠ **Is this stage the subject of the test?** A checkpoint *replaces a computation*, so a
   test whose assertions are about that computation must never be served one — it would assert
   on numbers the recorded run produced and pass whatever the code now does. A test declares
   its subject with ``@pytest.mark.stage_under_test("amf_atomic")``; that stage **and every
   stage downstream of it** are then computed for real, however fresh the checkpoints are.

Beyond those, the conservative defaults:

* **The whole mechanism is off unless asked for.** A fresh clone, CI and a plain ``pytest``
  compute everything, so no committed result can depend on a file in a git-ignored directory.
  This is what keeps "a test's result may never depend on a persistent cache" true of
  the suite as it is normally run.
* **The caller's own source counts too.** The builder is a closure in a test module, so that
  module's source (docstrings stripped — see :func:`normalized_digest`) is folded into the
  fingerprint. Editing a test invalidates its own checkpoints, which is the wrong-way-safe
  choice: the cost of an unnecessary recompute is time, the cost of a stale one is a false
  pass.
* **Nothing here may fail a test.** Every read failure is a miss and every write failure is a
  warning, exactly as ``kuiva/amf/cache.py`` is built (the persistent-cache boundary).

What is stored, and what is not
-------------------------------
A checkpoint payload is a **plain dict** of arrays and scalars: the *result* of a stage, not a
live object. That is a deliberate restriction. Serializing a ``ScalarX2CData`` or an
``AtomicDiracSolution`` would mean a second, unversioned definition of those objects living in
the test suite, and would store 1 GB per decontracted lanthanide for four numbers anybody
actually asserts on. The rule — checkpoint what is *cheap and precious*, regenerate what is
neither — points the same way here as it does for the AMF correction cache.

Usage
-----
::

    payload = stages.checkpoint(
        "amf_atomic",
        key={"element": "Ne", "basis": "dyallv2z", "interaction": "coulomb"},
        build=lambda: {"e_tot": solution.e_tot, "spinors": solution.occupied_energies()},
    )

``build`` is called only on a miss. The returned mapping is what it returned, or what an
earlier run's did.

Command line
------------
``python tests/stages.py`` prints what is stored, which entries are still valid under the
current sources, and which stage each belongs to; ``--purge`` removes them, ``--stages``
lists the registry with its dependency closures.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO / "kuiva"

#: File-layout version. Bump when the HDF5 layout changes — never for a change of physics,
#: which the source fingerprint already covers automatically.
SCHEMA = 1

#: ``off`` (default) | ``on`` | ``read`` | ``refresh`` | ``trust``. See :func:`mode`.
ENV_MODE = "KUIVA_TEST_CHECKPOINTS"
#: Where entries live. Default ``tests/checkpoints`` (git-ignored).
ENV_DIR = "KUIVA_TEST_CHECKPOINT_DIR"

DEFAULT_DIR = REPO / "tests" / "checkpoints"
MODES = ("off", "on", "read", "refresh", "trust")


# =============================================================================================
# The stage registry
# =============================================================================================

@dataclass(frozen=True)
class Stage:
    """One coarse step of a test calculation, and the code its result depends on.

    Attributes
    ----------
    name : str
        Registry key, and the subdirectory entries are stored in.
    modules : tuple of str
        The ``kuiva`` modules this stage is computed by. The fingerprint is taken over the
        transitive **import closure** of these, so naming the entry points is enough — the
        closure picks up everything they reach and, more importantly, nothing they do not.
    upstream : tuple of str
        Stages whose output this one consumes. Used for one thing: when a test declares a
        stage as its subject, that stage *and everything downstream of it* are recomputed.
    version : int
        Manual override for the rare change the fingerprint cannot see — a different
        convention in the *payload* (a renamed field, a different unit) with no source change
        in ``kuiva``. Bump it and every entry for the stage is stale.
    what : str
        One line for ``--stages`` and for the session summary.
    """
    name: str
    modules: Tuple[str, ...]
    upstream: Tuple[str, ...] = ()
    version: int = 1
    what: str = ""


#: ⚠ **The registry is the documentation of what may be replayed.** Adding a stage is a
#: statement that its output is reproducible from its declared modules and its key alone; if
#: that is not true the checkpoint is a lie that no test can catch. When in doubt, do not add
#: the stage — the cost of leaving it out is that it is computed every time.
STAGES: Dict[str, Stage] = {}


def register(stage: Stage) -> Stage:
    if stage.name in STAGES:
        raise ValueError("stage {!r} is already registered".format(stage.name))
    STAGES[stage.name] = stage
    return stage


register(Stage(
    name="amf_atomic",
    modules=("kuiva.amf.atomic", "kuiva.amf.backend", "kuiva.amf.pyscf_dhf",
             "kuiva.amf.configuration"),
    what="four-component atomic Dirac-Hartree-Fock (the dominant cost of the slow suite)",
))

register(Stage(
    name="amf_correction",
    modules=("kuiva.amf.correction", "kuiva.amf.decouple", "kuiva.amf.atomic"),
    upstream=("amf_atomic",),
    what="X2C decoupling of the atomic mean field, and its molecular assembly",
))

register(Stage(
    name="two_component_scf",
    modules=("kuiva.amf.correction", "kuiva.spinor.expand"),
    upstream=("amf_correction",),
    what="self-consistent two-component SCF on the corrected Hamiltonian",
))

register(Stage(
    name="scalar_scf",
    modules=("kuiva.interface.pyscf_bridge", "kuiva.interface.api", "kuiva.basis.registry"),
    what="the scalar-relativistic X2C SCF of the front-end ",
))

register(Stage(
    name="soc_ingestion",
    modules=("kuiva.interface.pyscf_bridge", "kuiva.spinor.expand"),
    upstream=("scalar_scf", "amf_correction"),
    what="two-component spin-orbit ingestion ",
))

register(Stage(
    name="integrals",
    modules=("kuiva.integrals.transform", "kuiva.orth.canonical", "kuiva.spinor.expand"),
    upstream=("scalar_scf",),
    what="orthonormal working basis and the AO->spinor-MO integral transform ",
))

# ⚠ Declared before the modules exist, deliberately. A stage that appears only when its
# implementation lands arrives with no downstream relationships recorded, and the first thing
# anyone does is checkpoint over the very step under development. These say what the pipeline
# is, so `stage_under_test` already means something for work that has not started.
register(Stage(
    name="preopt",
    modules=("kuiva.mcscf.preopt", "kuiva.ci.strings"),
    upstream=("integrals", "soc_ingestion"),
    what="cheap-CI pre-optimization - not yet consumed by any test",
))

register(Stage(
    name="casscf",
    modules=("kuiva.mcscf.orbopt", "kuiva.mcscf.preopt"),
    upstream=("preopt",),
    what="state-averaged CASSCF orbital optimization - not yet consumed by any test",
))


def downstream_closure(names: Iterable[str]) -> Set[str]:
    """``names`` plus every stage that (transitively) consumes one of them.

    This is what makes ``stage_under_test`` sound: if the atomic solve is the subject, a
    *correction* checkpoint built on top of a previous atomic solve is equally unusable, and
    so is anything built on that.
    """
    wanted = {n for n in names if n in STAGES}
    unknown = sorted(set(names) - wanted)
    if unknown:
        raise KeyError("unknown stage(s) {}; known: {}".format(unknown, sorted(STAGES)))
    grown = True
    while grown:
        grown = False
        for stage in STAGES.values():
            if stage.name not in wanted and wanted.intersection(stage.upstream):
                wanted.add(stage.name)
                grown = True
    return wanted


# =============================================================================================
# Source fingerprints: the automatic staleness rule
# =============================================================================================

def normalized_digest(path: Path) -> str:
    """SHA-256 of a Python source file with **docstrings removed**.

    Comments and docstrings do not change what a module computes, and a lanthanide re-solve is
    35 minutes; invalidating one because a docstring gained a sentence would make the whole
    mechanism something people turn off. The normalization is an ``ast`` round-trip, which also
    makes reformatting (line breaks, quote style) invisible — anything that survives it is a
    change to the code.

    Falls back to the raw bytes if the file cannot be parsed or unparsed, because a digest that
    is too *strict* only costs time.
    """
    raw = path.read_bytes()
    try:
        tree = ast.parse(raw)
        _strip_docstrings(tree)
        text = ast.unparse(tree) # Python 3.9+
    except Exception:                                                        # noqa: BLE001
        text = raw.decode("utf-8", "replace")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]


def package_modules() -> Dict[str, Path]:
    """``{module name: file}`` for every module of the ``kuiva`` package."""
    modules: Dict[str, Path] = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(REPO).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules[".".join(parts)] = path
    return modules


def _module_package(name: str, is_package: bool) -> str:
    return name if is_package else name.rpartition(".")[0]


def _module_edges(name: str, path: Path, modules: Set[str],
                  packages: Set[str]) -> Tuple[Set[str], Set[str]]:
    """``(followed, touched)`` ``kuiva`` modules for one source file.

    ⚠ **Two kinds of edge, and the distinction is what keeps the closures useful.** Importing
    ``kuiva.amf.backend`` *executes* ``kuiva/amf/__init__.py``, which re-exports most of the
    package. Following that edge would put ``kuiva.amf.decouple`` in the closure of everything
    that touches any part of ``kuiva.amf``, and the fingerprint would degenerate into "did
    anything in the package change" — which is the mechanism this module exists to avoid. But
    a re-export cannot change what ``backend`` computes.

    So a package reached only as the *ancestor* of an imported submodule is **touched**: its
    own source is hashed, since an ``__init__`` may contain code, but its edges are not
    followed. A package whose own namespace supplies the imported name —
    ``from kuiva.amf import amf_correction``, where ``amf_correction`` is not a submodule — is
    **followed**, because then the re-export is how the name was obtained.
    """
    followed: Set[str] = set()
    touched: Set[str] = set()

    def note_ancestors(target: str) -> None:
        parent = target.rpartition(".")[0]
        while parent:
            if parent in packages:
                touched.add(parent)
            parent = parent.rpartition(".")[0]

    try:
        tree = ast.parse(path.read_bytes())
    except SyntaxError:                                                      # pragma: no cover
        return followed, touched
    package = _module_package(name, name in packages)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] != "kuiva":
                    continue
                if alias.name in modules:
                    followed.add(alias.name)
                note_ancestors(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                                # relative: resolve against the file
                base = package
                for _ in range(node.level - 1):
                    base = base.rpartition(".")[0]
                if node.module:
                    base = "{}.{}".format(base, node.module) if base else node.module
            elif node.module and node.module.split(".")[0] == "kuiva":
                base = node.module
            else:
                continue
            if not base or base not in modules:
                continue
            note_ancestors(base)
            touched.add(base)
            for alias in node.names:
                submodule = "{}.{}".format(base, alias.name)
                if submodule in modules:
                    followed.add(submodule)              # a submodule: the real dependency
                else:
                    followed.add(base)                   # a name defined or re-exported by it
    return followed, touched


def import_graph() -> Dict[str, Tuple[Set[str], Set[str]]]:
    """``{module: (followed, touched)}``, restricted to the ``kuiva`` package."""
    modules = package_modules()
    names = set(modules)
    packages = {n for n, p in modules.items() if p.name == "__init__.py"}
    graph: Dict[str, Tuple[Set[str], Set[str]]] = {}
    for name, path in modules.items():
        followed, touched = _module_edges(name, path, names, packages)
        graph[name] = (followed - {name}, touched - followed - {name})
    return graph


_GRAPH_CACHE: Dict[str, object] = {}


def source_closure(roots: Sequence[str]) -> Tuple[str, ...]:
    """Every ``kuiva`` module whose source can change what ``roots`` compute, sorted.

    The point of the whole design: code outside this set provably cannot change the result of
    a stage built from ``roots``.
    """
    graph = _GRAPH_CACHE.get("graph")
    if graph is None:
        graph = import_graph()
        _GRAPH_CACHE["graph"] = graph
    unknown = [r for r in roots if r not in graph]
    if unknown:
        raise KeyError("no such kuiva module(s): {}".format(sorted(unknown)))
    seen: Set[str] = set()
    extra: Set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        followed, touched = graph.get(name, (set(), set()))
        stack.extend(followed)
        extra.update(touched)
    return tuple(sorted(seen | extra))


def stage_sources(name: str) -> Tuple[str, ...]:
    """The full source closure of a stage: its own modules and every upstream stage's.

    Upstream modules are included explicitly rather than left to the import graph. They are
    usually reached anyway — ``kuiva.amf.correction`` imports ``kuiva.amf.atomic`` — but a
    stage boundary that happens not to be an import edge (the front-end feeding the CI through
    plain arrays, which is exactly what the boundary asks for) would otherwise go unnoticed.
    """
    stage = STAGES[name]
    modules: Set[str] = set(stage.modules)
    for up in downstream_closure_upstream(name):
        modules.update(STAGES[up].modules)
    return source_closure(sorted(modules))


def downstream_closure_upstream(name: str) -> Set[str]:
    """Every stage ``name`` (transitively) consumes."""
    wanted: Set[str] = set()
    stack = list(STAGES[name].upstream)
    while stack:
        up = stack.pop()
        if up in wanted:
            continue
        wanted.add(up)
        stack.extend(STAGES[up].upstream)
    return wanted


_FINGERPRINTS: Dict[str, str] = {}


def stage_fingerprint(name: str) -> str:
    """Digest of everything outside the key that the stage's result depends on.

    Covers the source closure, the stage's manual :attr:`Stage.version`, and the versions of
    the two libraries the numbers actually come out of. ⚠ It does **not** cover the basis-set
    data files, the compiler, or MKL — a checkpoint is not a claim of bitwise reproducibility
    across machines, and the entries are per-machine and git-ignored for that reason.
    """
    cached = _FINGERPRINTS.get(name)
    if cached is not None:
        return cached
    stage = STAGES[name]
    modules = package_modules()
    parts: List[str] = ["stage={}".format(stage.name), "version={}".format(stage.version),
                        "schema={}".format(SCHEMA)]
    for module in stage_sources(name):
        parts.append("{}={}".format(module, normalized_digest(modules[module])))
    parts.append("numpy={}".format(np.__version__))
    parts.append("pyscf={}".format(_pyscf_version()))
    parts.append("kernels={}".format(_kernel_backend_token()))
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    _FINGERPRINTS[name] = digest
    return digest


def _pyscf_version() -> str:
    try:
        import pyscf
    except ImportError:                                                      # pragma: no cover
        return "absent"
    return str(pyscf.__version__)


def _kernel_backend_token() -> str:
    """⚠ The kernel-backend provenance tuple ``(preferred_backend, native_build_id)``.

    The source fingerprint above cannot see a compiled ``.so``, and a threaded reduction
    is only 1e-13-equal to the NumPy one — so without this token ``--checkpoints=on``
    could replay a NumPy-epoch stage into a native-epoch run and mask a parity defect
. Rendered by :func:`kuiva.util.native.fingerprint_token`,
    which owns the backend state.
    """
    from kuiva.util import native

    return native.fingerprint_token()


# =============================================================================================
# Mode, location and per-session state
# =============================================================================================

_STATE: Dict[str, Any] = {
    "mode": None,               # None -> read from the environment
    "directory": None,
    "under_test": frozenset(),
    "events": [],               # one record per checkpoint() call
    "once": set(),              # names of session-once actions already taken
}


def once(name: str) -> bool:
    """``True`` the first time it is called with ``name`` in this session, ``False`` after.

    For the session-once half of a per-test fixture — purging a directory in ``refresh`` mode
    must happen once, not once per test, or each test would erase what the previous one wrote.
    """
    taken = _STATE["once"]
    if name in taken:
        return False
    taken.add(name)
    return True


def mode() -> str:
    """``off`` | ``on`` | ``read`` | ``refresh`` | ``trust``.

    ``off``
        Compute everything, store nothing. **The default**, so that ``pytest`` on a fresh
        clone is the full calculation and no committed result can depend on a git-ignored
        file.
    ``on``
        Replay valid entries, write missing ones — except for a test's declared subject.
    ``read``
        As ``on``, but never writes. For running against a checkpoint set you do not want
        modified.
    ``refresh``
        Ignore what is stored, recompute, overwrite. How checkpoint data is produced.
    ``trust``
        ⚠ As ``on``, **including a test's declared subject**. This deliberately downgrades
        those tests from "does the code produce the right answer" to "does the recorded run
        still match the reference", and it is the mode that makes the slow suite cheap. The
        source fingerprint still applies, so a downgraded test is only ever replaying a run
        made by *this* code; what is given up is sensitivity to everything the fingerprint
        cannot see — the basis-set data files, MKL, the machine. The terminal summary names
        every stage it was used on. Never use it to decide that something works.
    """
    configured = _STATE["mode"]
    if configured is None:
        configured = os.environ.get(ENV_MODE, "off").strip().lower()
        if configured in ("1", "true", "yes"):
            configured = "on"
        elif configured in ("0", "false", "no", ""):
            configured = "off"
        if configured not in MODES:
            raise ValueError("${}={!r}: expected one of {}".format(
                ENV_MODE, configured, ", ".join(MODES)))
        _STATE["mode"] = configured
    return configured


def set_mode(value: str) -> None:
    if value not in MODES:
        raise ValueError("mode must be one of {}".format(", ".join(MODES)))
    _STATE["mode"] = value


def directory() -> Path:
    configured = _STATE["directory"]
    if configured is None:
        env = os.environ.get(ENV_DIR)
        configured = Path(os.path.expanduser(env)) if env else DEFAULT_DIR
        _STATE["directory"] = configured
    return Path(configured)


def set_directory(path) -> None:
    _STATE["directory"] = Path(path)


def set_under_test(names: Iterable[str]) -> frozenset:
    """Declare the stages this test is the test *of*; returns the previous value.

    They and everything downstream are computed for real. Called by the autouse fixture in
    ``conftest.py`` from ``@pytest.mark.stage_under_test`` marks.
    """
    previous = _STATE["under_test"]
    _STATE["under_test"] = frozenset(downstream_closure(names)) if names else frozenset()
    return previous


def restore_under_test(previous) -> None:
    _STATE["under_test"] = frozenset(previous)


def under_test() -> frozenset:
    return _STATE["under_test"]


def amf_cache_directory() -> Optional[Path]:
    """Where ``$KUIVA_AMF_CACHE`` should point for this run, or ``None`` for "off".

    ⚠ **Not a second cache — Kuiva's own one, made safe for the suite.**
    ``kuiva/amf/cache.py`` already exists so that a lanthanide is solved once ever, and it is
    the only thing that can skip the four-component solve *inside* ``amf_correction``, which
    is where the hours are. ``tests/conftest.py`` turns it off for the whole session for two
    good reasons (pollution of the user's real cache, and call-count assertions that must see
    a cold start), and this re-enables it without giving either back:

    * **A fingerprinted directory.** Entries live under the ``amf_correction`` stage
      fingerprint, so any change to the atomic solver, the decoupling or the assembly starts
      cold. That is strictly stronger than the manual ``cache.FORMULA_VERSION``, which stays as
      the guarantee for *users*, who have no fingerprint.
    * **Off whenever it would hide the subject.** A test marked
      ``stage_under_test("amf_atomic")`` or ``("amf_correction")`` gets ``off``, so every
      call-count and cold-start assertion still sees exactly what it saw before.

    Returns ``None`` in ``off`` mode, which is the default and is what a fresh clone does.
    """
    if mode() == "off":
        return None
    if under_test().intersection(("amf_atomic", "amf_correction")):
        return None
    return directory() / "amf_correction" / stage_fingerprint("amf_correction")[:16]


def events() -> Tuple[Dict[str, Any], ...]:
    """One record per :func:`checkpoint` call this session — for the terminal summary."""
    return tuple(_STATE["events"])


def clear_events() -> None:
    _STATE["events"] = []


# =============================================================================================
# The seam
# =============================================================================================

#: In-process memo, so one session computes a stage once however many tests consume it — the
#: job the module-level dictionaries in the individual test files used to do, one per file.
#: Keyed by the fingerprint as well, so a monkeypatched source in a test of this machinery
#: cannot be served a stale answer.
_MEMO: Dict[Tuple[str, str, str], Dict[str, Any]] = {}


def clear_memo() -> None:
    _MEMO.clear()


def checkpoint(stage: str, key: Mapping[str, Any], build: Callable[[], Mapping[str, Any]],
               *, extra_sources: Sequence[str] = ()) -> Dict[str, Any]:
    """Return the recorded output of ``stage`` for ``key``, computing it if need be.

    Parameters
    ----------
    stage : str
        A key of :data:`STAGES`.
    key : mapping
        Everything about *this instance* of the stage: element, basis, configuration,
        interaction, active space... Stored in full and compared on read, so an incomplete key
        presents as a miss rather than as another system's numbers (the rule
        ``kuiva/amf/cache.py`` states as point 2 of its docstring).
    build : callable
        Called on a miss. Must return a mapping of arrays and scalars — see the module
        docstring for why the payload is a plain mapping and not the live object.
    extra_sources : sequence of str
        Repository-relative paths of *test-side* sources the builder depends on beyond its own
        module — a helper imported from another test file. The calling module is always
        included automatically.
    """
    if stage not in STAGES:
        raise KeyError("unknown stage {!r}; known: {}".format(stage, sorted(STAGES)))
    key = dict(key)
    fingerprint = _full_fingerprint(stage, extra_sources)
    # ⚠ The filename deliberately does **not** carry the fingerprint: a stale entry is then
    # overwritten by the run that supersedes it, rather than the directory growing one copy per
    # edit of the source. Staleness is decided by what is inside the file.
    path = _entry_path(stage, key, _key_digest(stage, key))

    current = mode()
    subject = stage in under_test()
    # ⚠ Being the subject blocks the **read**, never the write. Blocking the write would mean
    # that whichever test happened to run first decided whether a checkpoint ever existed —
    # and the tests that are subjects are exactly the ones that compute the expensive stages,
    # so the entries most worth having would be the ones never written.
    blocked = subject and current != "trust"

    memo = _MEMO.get((stage, path.name, fingerprint))
    if memo is not None:
        # An earlier test in this session already computed or replayed it. This is what the
        # module-level dictionaries in the test files used to do, one per file.
        _record(stage, key, "memo", path, 0.0, 0.0)
        return dict(memo)

    payload = None
    if current in ("on", "read", "trust") and not blocked:
        payload = _load(path, stage=stage, key=key, fingerprint=fingerprint)

    if payload is not None:
        _MEMO[(stage, path.name, fingerprint)] = payload
        _record(stage, key, "trusted" if subject else "hit", path, 0.0, 0.0)
        return dict(payload)

    wall, cpu = time.time(), time.process_time()
    payload = _as_payload(build())
    wall, cpu = time.time() - wall, time.process_time() - cpu
    _MEMO[(stage, path.name, fingerprint)] = payload

    if current in ("on", "refresh", "trust"):
        _store(path, stage=stage, key=key, fingerprint=fingerprint, payload=payload,
               wall=wall, cpu=cpu)
    reason = ("under test" if blocked else
              "checkpoints off" if current == "off" else
              "refresh" if current == "refresh" else "miss")
    _record(stage, key, reason, path, wall, cpu)
    return dict(payload)


def _record(stage: str, key: Mapping[str, Any], outcome: str, path: Path,
            wall: float, cpu: float) -> None:
    _STATE["events"].append({"stage": stage, "key": dict(key), "outcome": outcome,
                             "path": str(path), "wall": wall, "cpu": cpu})


def _full_fingerprint(stage: str, extra_sources: Sequence[str]) -> str:
    """The stage fingerprint plus the test-side sources of this particular builder."""
    parts = [stage_fingerprint(stage)]
    for rel in sorted(set(_caller_sources()).union(extra_sources)):
        path = REPO / rel
        if path.is_file():
            parts.append("{}={}".format(rel, normalized_digest(path)))
        else:                                                                # pragma: no cover
            parts.append("{}=absent".format(rel))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _caller_sources() -> Tuple[str, ...]:
    """Repository-relative path of the module that called :func:`checkpoint`."""
    frame = sys._getframe(1)
    while frame is not None:
        filename = frame.f_globals.get("__file__")
        if filename and Path(filename).resolve() != Path(__file__).resolve():
            try:
                return (str(Path(filename).resolve().relative_to(REPO)),)
            except ValueError:                                               # pragma: no cover
                return ()
        frame = frame.f_back
    return ()                                                                # pragma: no cover


def _key_digest(stage: str, key: Mapping[str, Any]) -> str:
    text = json.dumps({"stage": stage, "key": _jsonable(key)}, sort_keys=True, default=repr)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonable(key: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, value in key.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
        elif isinstance(value, (list, tuple)):
            out[name] = [_scalar(v) for v in value]
        else:
            out[name] = repr(value)
    return out


def _scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _slug(key: Mapping[str, Any]) -> str:
    """A readable prefix so a directory listing says something without opening files."""
    bits = []
    for name in sorted(key):
        value = key[name]
        if isinstance(value, (str, int, bool)) and not isinstance(value, bool):
            bits.append(str(value))
        elif isinstance(value, bool):
            bits.append("{}{}".format("" if value else "no-", name))
    text = "-".join(bits)[:48]
    return "".join(c if (c.isalnum() or c in "-_+.") else "_" for c in text) or "entry"


def _entry_path(stage: str, key: Mapping[str, Any], digest: str) -> Path:
    return directory() / stage / "{}-{}.h5".format(_slug(key), digest[:12])


# --- payload encoding -----------------------------------------------------------------------

def _as_payload(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a builder's return value.

    ⚠ **Checked in every mode, including ``off``.** Storability is a property of the test, not
    of the run: leaving it to the write would mean a builder that cannot be checkpointed
    passes on every developer machine and fails only for whoever first turns checkpoints on —
    and it would fail there as a *warning*, since a write may never raise.
    """
    if not isinstance(value, Mapping):
        raise TypeError("a stage builder must return a mapping of arrays and scalars, got {}"
                        .format(type(value).__name__))
    payload = dict(value)
    for name, item in payload.items():
        if not isinstance(name, str):
            raise TypeError("stage payload keys must be strings, got {!r}".format(name))
        if _payload_kind(item) is None:
            raise TypeError(
                "stage payloads hold arrays and scalars only; {!r} is {}. Reduce it to what "
                "the test asserts on (see tests/stages.py)".format(name, type(item).__name__))
    return payload


def _payload_kind(value: Any) -> Optional[str]:
    """The storage kind of a payload value, or ``None`` if it has none."""
    if value is None:
        return "none"
    if isinstance(value, np.ndarray):
        return "array"
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except Exception:                                                    # noqa: BLE001
            return None
        return "list" if array.dtype != np.dtype("O") else None
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, (int, np.integer)):
        return "int"
    if isinstance(value, (float, np.floating)):
        return "float"
    if isinstance(value, str):
        return "str"
    return None


def _h5py():
    try:
        import h5py
    except ImportError:                                                      # pragma: no cover
        return None
    return h5py


def _load(path: Path, *, stage: str, key: Mapping[str, Any],
          fingerprint: str) -> Optional[Dict[str, Any]]:
    """The stored payload, or ``None``. Never raises — every failure is a miss."""
    h5py = _h5py()
    if h5py is None or not path.is_file():
        return None
    try:
        with h5py.File(str(path), "r") as f:
            if int(f.attrs.get("schema", -1)) != SCHEMA:
                return None
            if _text(f.attrs.get("stage")) != stage:
                return None
            if _text(f.attrs.get("fingerprint")) != fingerprint:
                return None
            stored_key = json.loads(_text(f.attrs.get("key", "{}")))
            if stored_key != _jsonable(key):
                # ⚠ A warning, not a silent miss: two different keys reaching one filename
                # means the digest is not separating them, which is a defect and not wear.
                _warn("checkpoint {} does not describe the key it was found for (stored {}, "
                      "wanted {}); recomputing".format(path.name, stored_key, _jsonable(key)))
                return None
            return _read_group(f["payload"])
    except Exception as exc:                                                 # noqa: BLE001
        _warn("cannot read the stage checkpoint {} ({}: {}); recomputing"
              .format(path, type(exc).__name__, exc))
        return None


def _store(path: Path, *, stage: str, key: Mapping[str, Any], fingerprint: str,
           payload: Mapping[str, Any], wall: float, cpu: float) -> bool:
    """Write a payload. Never raises: a checkpoint may not fail a calculation."""
    h5py = _h5py()
    if h5py is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp", dir=str(path.parent))
        os.close(fd)
        try:
            with h5py.File(tmp, "w") as f:
                f.attrs["schema"] = SCHEMA
                f.attrs["stage"] = stage
                f.attrs["fingerprint"] = fingerprint
                # The stage half on its own, so that ``--status`` can say *which* of the two
                # halves has moved without a test run to tell it. Never read by :func:`_load`,
                # which compares the combined value.
                f.attrs["stage_fingerprint"] = stage_fingerprint(stage)
                f.attrs["key"] = json.dumps(_jsonable(key), sort_keys=True)
                f.attrs["written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                f.attrs["wall_seconds"] = float(wall)
                f.attrs["cpu_seconds"] = float(cpu)
                _write_group(f.create_group("payload"), payload)
            os.replace(tmp, str(path))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except Exception as exc:                                                 # noqa: BLE001
        _warn("cannot write the stage checkpoint {} ({}: {}); continuing without it"
              .format(path, type(exc).__name__, exc))
        return False
    return True


#: Payload value kinds, stored beside each entry so a restore returns the type that was put in
#: rather than whatever h5py happens to hand back.
_KIND_ATTR = "kinds"


def _write_group(group, payload: Mapping[str, Any]) -> None:
    kinds: Dict[str, str] = {}
    for name, value in payload.items():
        kind = _payload_kind(value)          # already validated by :func:`_as_payload`
        kinds[name] = kind
        if kind == "array":
            group.create_dataset(name, data=value)
        elif kind == "list":
            group.create_dataset(name, data=np.asarray(value))
        elif kind == "bool":
            group.attrs[name] = int(value)
        elif kind == "int":
            group.attrs[name] = int(value)
        elif kind == "float":
            group.attrs[name] = float(value)
        elif kind == "str":
            group.attrs[name] = value
    group.attrs[_KIND_ATTR] = json.dumps(kinds, sort_keys=True)


def _read_group(group) -> Dict[str, Any]:
    kinds = json.loads(_text(group.attrs[_KIND_ATTR]))
    payload: Dict[str, Any] = {}
    for name, kind in kinds.items():
        if kind == "none":
            payload[name] = None
        elif kind == "array":
            payload[name] = np.array(group[name])
        elif kind == "list":
            payload[name] = np.array(group[name]).tolist()
        elif kind == "bool":
            payload[name] = bool(group.attrs[name])
        elif kind == "int":
            payload[name] = int(group.attrs[name])
        elif kind == "float":
            payload[name] = float(group.attrs[name])
        elif kind == "str":
            payload[name] = _text(group.attrs[name])
        else:                                                                # pragma: no cover
            raise ValueError("unknown payload kind {!r}".format(kind))
    return payload


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return "" if value is None else str(value)


def _warn(message: str) -> None:
    """Warn through Kuiva's logging so it obeys the output grammar like everything else."""
    try:
        from kuiva.util.logging import get_logger
        get_logger("kuiva.tests.stages").warning("%s", message)
    except Exception:                                                        # pragma: no cover
        sys.stderr.write("*** WARNING [tests.stages] {}\n".format(message))


# =============================================================================================
# Reporting
# =============================================================================================

def summary_lines() -> List[str]:
    """The end-of-session block: what was replayed, what was not, and why."""
    records = events()
    if not records:
        return []
    lines = ["stage checkpoints: mode={} dir={}".format(mode(), directory())]
    by_stage: Dict[str, Dict[str, Any]] = {}
    for record in records:
        entry = by_stage.setdefault(
            record["stage"],
            {"hit": 0, "trusted": 0, "memo": 0, "computed": 0, "cpu": 0.0, "why": set()})
        if record["outcome"] in ("hit", "trusted", "memo"):
            entry[record["outcome"]] += 1
        else:
            entry["computed"] += 1
            entry["cpu"] += record["cpu"]
            entry["why"].add(record["outcome"])
    for name in sorted(by_stage):
        entry = by_stage[name]
        why = ", ".join(sorted(entry["why"])) if entry["why"] else "-"
        lines.append(
            "  {:<18s} {:3d} replayed  {:3d} reused  {:3d} computed ({:.1f} CPU s; {})"
            .format(name, entry["hit"] + entry["trusted"], entry["memo"], entry["computed"],
                    entry["cpu"], why))
    trusted = sorted(n for n, e in by_stage.items() if e["trusted"])
    if trusted:
        # ⚠ Loud on purpose. In this mode the tests that validate these stages asserted on a
        # recorded run rather than on a fresh one, and a summary that did not say so would
        # make a 20-minute suite look like a 3-hour one.
        lines.append("  *** WARNING: --checkpoints=trust replayed the stage(s) under test "
                     "({}); those tests did NOT validate the computation".format(
                         ", ".join(trusted)))
    return lines


def stored_entries() -> List[Dict[str, Any]]:
    """Every entry on disk, with whether it is still valid under the current sources."""
    h5py = _h5py()
    root = directory()
    found: List[Dict[str, Any]] = []
    if h5py is None or not root.is_dir():
        return found
    for stage_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        stage = stage_dir.name
        current = stage_fingerprint(stage) if stage in STAGES else None
        for path in sorted(stage_dir.glob("*.h5")):
            record: Dict[str, Any] = {"stage": stage, "path": path,
                                      "size_mb": path.stat().st_size / 1024.0 ** 2}
            try:
                with h5py.File(str(path), "r") as f:
                    record["key"] = json.loads(_text(f.attrs.get("key", "{}")))
                    record["written"] = _text(f.attrs.get("written"))
                    record["cpu_seconds"] = float(f.attrs.get("cpu_seconds", 0.0))
                    stored = _text(f.attrs.get("stage_fingerprint"))
                # ⚠ This says only that the **stage** fingerprint still matches — that no
                # ``kuiva`` module in the closure has changed. The other half of the
                # fingerprint is the calling test module, which nothing outside a run can
                # know, so an ``ok`` entry may still miss at test time. The reverse cannot
                # happen: a ``STALE`` entry will certainly be recomputed.
                record["stage_current"] = current is not None and stored == current
            except Exception as exc:                                         # noqa: BLE001
                record["error"] = "{}: {}".format(type(exc).__name__, exc)
                record["stage_current"] = False
            found.append(record)
    return found


def amf_cache_generations() -> List[Dict[str, Any]]:
    """The per-fingerprint X2CAMF correction caches on disk, newest fingerprint marked.

    ⚠ These are the entries that skip the four-component solve, and they are **not** the
    ``.h5`` files :func:`stored_entries` lists — they live one level deeper, under the
    fingerprint they were written for. A superseded fingerprint's directory is inert but keeps
    its disk, so it is reported rather than silently accumulated. It is not deleted
    automatically: switching between two branches would then pay a lanthanide solve every
    time you switched back.
    """
    root = directory() / "amf_correction"
    if not root.is_dir():
        return []
    current = stage_fingerprint("amf_correction")[:16]
    generations = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(path.glob("*.h5"))
        generations.append({
            "path": path, "fingerprint": path.name, "current": path.name == current,
            "entries": len(files),
            "size_mb": sum(f.stat().st_size for f in files) / 1024.0 ** 2})
    return generations


def purge(stage: Optional[str] = None) -> int:
    """Delete stored entries, optionally of one stage only. Returns how many were removed."""
    root = directory()
    if not root.is_dir():
        return 0
    removed = 0
    for path in sorted(root.rglob("*.h5")):
        # The stage is the first path component under the root — ``amf_correction`` entries
        # sit one level deeper, under their fingerprint.
        owner = path.relative_to(root).parts[0]
        if stage is not None and owner != stage:
            continue
        try:
            path.unlink()
            removed += 1
        except OSError as exc:                                               # pragma: no cover
            _warn("cannot remove {}: {}".format(path, exc))
    return removed


# =============================================================================================
# Command line
# =============================================================================================

def _main(argv: Sequence[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stages", action="store_true",
                        help="list the stage registry and each stage's source closure")
    parser.add_argument("--purge", nargs="?", const="", metavar="STAGE",
                        help="delete stored entries (optionally of one stage only)")
    parser.add_argument("--dir", help="checkpoint directory (default tests/checkpoints)")
    args = parser.parse_args(argv)

    if args.dir:
        set_directory(args.dir)

    if args.stages:
        for name in sorted(STAGES):
            stage = STAGES[name]
            closure = stage_sources(name)
            print("{}  (v{})".format(name, stage.version))
            print("    {}".format(stage.what))
            print("    upstream: {}".format(", ".join(stage.upstream) or "-"))
            print("    fingerprint: {}".format(stage_fingerprint(name)[:16]))
            print("    closure ({} modules): {}".format(len(closure), ", ".join(closure)))
        return 0

    if args.purge is not None:
        removed = purge(args.purge or None)
        print("removed {} checkpoint(s) from {}".format(removed, directory()))
        return 0

    entries = stored_entries()
    generations = amf_cache_generations()
    print("checkpoint directory: {}".format(directory()))
    if not entries and not generations:
        print("  (empty)")
        return 0
    for entry in entries:
        flag = "ok  " if entry.get("stage_current") else "STALE"
        print("  {} {:<16s} {:>7.2f} MB  {:>9.1f} CPU s  {}".format(
            flag, entry["stage"], entry["size_mb"], entry.get("cpu_seconds", 0.0),
            json.dumps(entry.get("key", {}), sort_keys=True)))
    if entries:
        print("  {} stage entries, {:.1f} MB, {:.0f} CPU s recorded".format(
            len(entries), sum(e["size_mb"] for e in entries),
            sum(e.get("cpu_seconds", 0.0) for e in entries)))
    for generation in generations:
        print("  {} X2CAMF corrections  {:>7.2f} MB  {:3d} entries  {}".format(
            "ok  " if generation["current"] else "STALE", generation["size_mb"],
            generation["entries"], generation["fingerprint"]))
    return 0


if __name__ == "__main__":                                                   # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))
