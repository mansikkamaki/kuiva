"""Tier 0: the stage-checkpoint machinery (``tests/stages.py``).

⚠ **This file guards a mechanism that can make other tests stop testing anything.** A stage
checkpoint replaces a computation with numbers a previous run wrote; if the rules that decide
*when* it may do so are wrong, the failure is a suite that passes while the code is broken —
the worst failure mode a test suite has. So the assertions here are about the two gates and
their conservatism, not about HDF5 round-tripping:

* the source fingerprint moves when a module **in** the stage's closure changes and does not
  move when one **outside** it does — the property the whole design rests on;
* a stage declared as the subject of a test is never replayed, nor is anything downstream;
* the default is off, so a fresh clone computes everything;
* every failure mode — corrupt file, mismatched key, unwritable directory — is a **miss**, and
  never an exception (a cache may not fail a calculation).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import stages


@pytest.fixture(autouse=True)
def _cold_memo():
    """⚠ Every test here reuses the key ``{"element": "Ne"}``, and the in-process memo of
    :func:`stages.checkpoint` is deliberately session-lived — so without this each test would
    be answered by the previous one and would assert nothing."""
    stages.clear_memo()
    yield
    stages.clear_memo()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A scratch checkpoint directory in mode ``on``, restored afterwards."""
    monkeypatch.setattr(stages, "_STATE", dict(stages._STATE, events=[]))
    stages.set_directory(tmp_path)
    stages.set_mode("on")
    stages.set_under_test(())
    return tmp_path


class Counter:
    """A builder that records how many times it actually ran."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return dict(self.payload)


# --- gate 1: the source fingerprint --------------------------------------------------------

def test_a_stage_closure_contains_what_produces_it_and_not_the_rest():
    """⚠ **The headline property, stated as an assertion.**

    The reason for the whole mechanism is that editing the CI has no bearing on a
    four-component atomic solve, so the atomic checkpoints must survive it. Stated the other
    way round for the same stage: the four-component backend *is* what produces it, so a
    change there must invalidate.
    """
    closure = set(stages.stage_sources("amf_atomic"))
    assert "kuiva.amf.pyscf_dhf" in closure
    assert "kuiva.amf.backend" in closure
    assert "kuiva.ci.strings" not in closure
    assert "kuiva.mcscf.orbopt" not in closure
    assert "kuiva.integrals.transform" not in closure


def test_every_registered_stage_names_modules_that_exist():
    """A typo, or a module renamed out from under a stage, must fail here and not silently
    shrink the closure it is supposed to be watching."""
    for name in stages.STAGES:
        assert stages.stage_sources(name), name


def test_a_stage_closure_includes_its_upstream_stages():
    """A correction is computed *from* an atomic solve, so the code that produces the solve is
    part of what its own checkpoint depends on — whether or not the two happen to be joined by
    an import edge (stage boundaries must *not* be import edges)."""
    upstream = set(stages.stage_sources("amf_atomic"))
    assert upstream.issubset(set(stages.stage_sources("amf_correction")))


def test_the_fingerprint_moves_with_a_module_inside_the_closure(monkeypatch):
    before = stages.stage_fingerprint("amf_atomic")
    real = stages.normalized_digest

    def perturbed(path):
        digest = real(path)
        return "changed" if path.name == "pyscf_dhf.py" else digest

    monkeypatch.setattr(stages, "normalized_digest", perturbed)
    monkeypatch.setattr(stages, "_FINGERPRINTS", {})
    assert stages.stage_fingerprint("amf_atomic") != before


def test_the_fingerprint_ignores_a_module_outside_the_closure(monkeypatch):
    """The saving, in one assertion: a CI edit does not re-solve a lanthanide."""
    before = stages.stage_fingerprint("amf_atomic")
    real = stages.normalized_digest

    def perturbed(path):
        return "changed" if path.name == "strings.py" else real(path)

    monkeypatch.setattr(stages, "normalized_digest", perturbed)
    monkeypatch.setattr(stages, "_FINGERPRINTS", {})
    assert stages.stage_fingerprint("amf_atomic") == before


def test_a_stale_fingerprint_is_a_miss_and_the_entry_is_replaced(store, monkeypatch):
    key = {"element": "Ne"}
    builder = Counter({"e": 1.0})
    stages.checkpoint("amf_atomic", key, builder)
    assert builder.calls == 1
    stages.checkpoint("amf_atomic", key, builder)
    assert builder.calls == 1                                   # replayed

    real = stages.normalized_digest
    monkeypatch.setattr(stages, "normalized_digest",
                        lambda p: "changed" if p.name == "backend.py" else real(p))
    monkeypatch.setattr(stages, "_FINGERPRINTS", {})
    stages.checkpoint("amf_atomic", key, builder)
    assert builder.calls == 2                                   # recomputed
    # ⚠ Replaced, not accumulated: the filename carries the key, not the fingerprint, so a
    # long-lived checkpoint directory cannot grow one entry per edit of the source.
    assert len(list((store / "amf_atomic").glob("*.h5"))) == 1


def test_docstring_edits_do_not_invalidate(tmp_path):
    """The normalization that keeps the mechanism usable: a comment or a docstring cannot
    change what a module computes, and invalidating a 35-minute lanthanide solve for one is
    how a feature like this gets switched off permanently."""
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text('"""One docstring."""\nX = 1  # a comment\n')
    b.write_text('"""A different docstring, longer."""\nX = 1\n')
    assert stages.normalized_digest(a) == stages.normalized_digest(b)
    b.write_text('"""One docstring."""\nX = 2\n')
    assert stages.normalized_digest(a) != stages.normalized_digest(b)


# --- gate 2: the stage under test ----------------------------------------------------------

def test_downstream_closure_follows_the_pipeline():
    """Declaring the atomic solve as the subject must also block the correction built on it,
    and the two-component SCF built on that. A checkpoint one step downstream of a broken
    stage is exactly as unusable as one of the stage itself."""
    assert stages.downstream_closure(["amf_atomic"]) >= {
        "amf_atomic", "amf_correction", "two_component_scf", "soc_ingestion"}
    assert "scalar_scf" not in stages.downstream_closure(["amf_atomic"])


def test_an_unknown_stage_is_refused():
    with pytest.raises(KeyError):
        stages.downstream_closure(["no_such_stage"])
    with pytest.raises(KeyError):
        stages.checkpoint("no_such_stage", {}, dict)


def test_a_stage_under_test_is_never_replayed(store):
    key = {"element": "Ne"}
    builder = Counter({"e": 1.0})
    stages.checkpoint("amf_atomic", key, builder)
    assert builder.calls == 1

    stages.clear_memo()                              # a later session, reading what was stored
    stages.set_under_test(["amf_atomic"])
    stages.checkpoint("amf_atomic", key, builder)
    assert builder.calls == 2, "a test's own subject must be computed, not replayed"


def test_a_downstream_stage_is_not_replayed_either(store):
    key = {"element": "Ne"}
    builder = Counter({"e": 1.0})
    stages.checkpoint("amf_correction", key, builder)
    stages.clear_memo()
    stages.set_under_test(["amf_atomic"])
    stages.checkpoint("amf_correction", key, builder)
    assert builder.calls == 2


@pytest.mark.stage_under_test("amf_atomic")
def test_the_marker_reaches_the_store():
    """The fixture in ``conftest.py`` is what connects the mark to the store; without it the
    mark would be decoration and every declared subject would still be replayed."""
    assert "amf_atomic" in stages.under_test()
    assert "amf_correction" in stages.under_test()


def test_the_marker_does_not_leak_into_the_next_test():
    assert stages.under_test() == frozenset()


def test_every_solve_count_assertion_declares_its_stage():
    """⚠ **The structural guard, and the reason it has to exist.**

    ``cache_statistics()["solves"]`` is how this suite verifies caching honestly (
    timing is not evidence). With the stage-managed X2CAMF cache underneath it, every one of
    those counters reads **zero** and every such test passes vacuously. Remembering to mark
    them is not a plan — a test added next year would silently join the vacuous ones — so the
    marks are checked here, from the sources.
    """
    import ast
    from pathlib import Path

    unmarked = []
    here = Path(__file__).resolve()
    for path in sorted(here.parent.glob("test_*.py")):
        if path == here:                      # this file names the counter only to look for it
            continue
        tree = ast.parse(path.read_bytes())
        module_marked = any(
            "stage_under_test" in ast.dump(node) for node in tree.body
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", "") == "pytestmark" for t in node.targets))
        if module_marked:
            continue
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            body = ast.dump(node)
            if "cache_statistics" not in body:
                continue
            if not any("stage_under_test" in ast.dump(d) for d in node.decorator_list):
                unmarked.append("{}::{}".format(path.name, node.name))
    assert not unmarked, (
        "these tests assert on a solve/hit count but do not declare a stage under test, so a "
        "stage-managed cache would answer them with zero: {}. Add "
        '@pytest.mark.stage_under_test("amf_atomic").'.format(unmarked))


# --- the conservative default --------------------------------------------------------------

def test_checkpoints_are_off_unless_asked_for(tmp_path, monkeypatch):
    """⚠ A fresh clone, and a plain ``pytest``, must compute everything. This is what keeps
    "a test's result may never depend on a persistent cache" true of the suite as it
    is normally run."""
    monkeypatch.setattr(stages, "_STATE", dict(stages._STATE, mode=None, events=[]))
    monkeypatch.delenv(stages.ENV_MODE, raising=False)
    stages.set_directory(tmp_path)
    assert stages.mode() == "off"

    builder = Counter({"e": 1.0})
    for _ in range(3):
        stages.clear_memo()
        stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)
    assert builder.calls == 3
    assert not list(tmp_path.rglob("*.h5")), "nothing may be written while checkpoints are off"


def test_read_mode_never_writes(store):
    stages.set_mode("read")
    builder = Counter({"e": 1.0})
    stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)
    assert builder.calls == 1
    assert not list(store.rglob("*.h5"))


def test_refresh_recomputes_and_overwrites(store):
    builder = Counter({"e": 1.0})
    stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)
    stages.clear_memo()
    stages.set_mode("refresh")
    stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)
    assert builder.calls == 2
    stages.clear_memo()
    stages.set_mode("on")
    stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)
    assert builder.calls == 2                                   # the refreshed entry is used


def test_an_invalid_mode_is_refused(monkeypatch):
    monkeypatch.setattr(stages, "_STATE", dict(stages._STATE, mode=None))
    monkeypatch.setenv(stages.ENV_MODE, "maybe")
    with pytest.raises(ValueError):
        stages.mode()


# --- payloads -------------------------------------------------------------------------------

def test_a_payload_round_trips_by_value_and_by_type(store):
    payload = {"energies": np.linspace(-1.0, 1.0, 7), "complex": np.array([1 + 2j, 3 - 4j]),
               "e_tot": -128.5, "n": 7, "converged": True, "label": "Ne 2p",
               "levels": [1.0, 2.0, 3.0], "absent": None}
    builder = Counter(payload)
    stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)
    restored = stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)

    assert builder.calls == 1
    assert set(restored) == set(payload)
    assert np.array_equal(restored["energies"], payload["energies"])
    assert np.array_equal(restored["complex"], payload["complex"])
    assert restored["e_tot"] == payload["e_tot"]
    assert restored["n"] == 7 and isinstance(restored["n"], int)
    assert restored["converged"] is True
    assert restored["label"] == "Ne 2p"
    assert restored["levels"] == [1.0, 2.0, 3.0]
    assert restored["absent"] is None


def test_a_payload_that_is_not_a_mapping_is_refused(store):
    with pytest.raises(TypeError):
        stages.checkpoint("amf_atomic", {"element": "Ne"}, lambda: (1.0, 2.0))


def test_a_live_object_in_a_payload_is_refused_with_advice(store):
    """⚠ The restriction that keeps this from becoming a second, unversioned definition of
    Kuiva's data classes — and from storing 1 GB per lanthanide for four numbers."""
    with pytest.raises(TypeError, match="arrays and scalars"):
        stages.checkpoint("amf_atomic", {"element": "Ne"}, lambda: {"solution": object()})


# --- every failure is a miss ----------------------------------------------------------------

def test_a_different_key_gets_a_different_entry(store):
    builder = Counter({"e": 1.0})
    stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)
    stages.checkpoint("amf_atomic", {"element": "Ar"}, builder)
    assert builder.calls == 2
    assert len(list((store / "amf_atomic").glob("*.h5"))) == 2


def test_a_corrupt_entry_is_a_miss_not_an_exception(store, kuiva_caplog):
    builder = Counter({"e": 1.0})
    stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)
    entry = next((store / "amf_atomic").glob("*.h5"))
    entry.write_bytes(b"not an HDF5 file at all")
    stages.clear_memo()

    assert stages.checkpoint("amf_atomic", {"element": "Ne"}, builder) == {"e": 1.0}
    assert builder.calls == 2
    assert any("stage checkpoint" in r.message for r in kuiva_caplog.records)


def test_a_truncated_entry_is_a_miss(store):
    builder = Counter({"e": 1.0})
    stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)
    entry = next((store / "amf_atomic").glob("*.h5"))
    entry.write_bytes(entry.read_bytes()[: len(entry.read_bytes()) // 2])
    stages.clear_memo()
    stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)
    assert builder.calls == 2


def test_an_entry_describing_another_key_warns_and_misses(store, kuiva_caplog):
    """⚠ Not a silent miss. Two keys landing on one filename means the digest has stopped
    separating them, which is a defect in the key and not ordinary staleness — the same
    distinction ``kuiva/amf/cache.py`` makes."""
    h5py = pytest.importorskip("h5py")
    builder = Counter({"e": 1.0})
    stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)
    entry = next((store / "amf_atomic").glob("*.h5"))
    with h5py.File(str(entry), "r+") as f:
        f.attrs["key"] = json.dumps({"element": "Ar"}, sort_keys=True)
    stages.clear_memo()

    stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)
    assert builder.calls == 2
    assert any("does not describe the key" in r.message for r in kuiva_caplog.records)


def test_an_unwritable_directory_warns_and_the_test_still_runs(tmp_path, monkeypatch,
                                                               kuiva_caplog):
    """A checkpoint may never fail a calculation."""
    monkeypatch.setattr(stages, "_STATE", dict(stages._STATE, events=[]))
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    stages.set_directory(blocked)
    stages.set_mode("on")
    try:
        result = stages.checkpoint("amf_atomic", {"element": "Ne"}, lambda: {"e": 1.0})
    finally:
        blocked.chmod(0o700)
    assert result == {"e": 1.0}
    assert any("cannot write" in r.message for r in kuiva_caplog.records)


def test_a_missing_directory_is_simply_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "_STATE", dict(stages._STATE, events=[]))
    stages.set_directory(tmp_path / "nothing" / "here")
    stages.set_mode("read")
    assert stages.checkpoint("amf_atomic", {"element": "Ne"}, lambda: {"e": 1.0}) == {"e": 1.0}


# --- reporting -------------------------------------------------------------------------------

def test_the_session_summary_says_what_was_replayed_and_why(store):
    """An unreported replay is indistinguishable from a computation that silently stopped
    happening, so the summary is part of the mechanism rather than decoration."""
    builder = Counter({"e": 1.0})
    stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)
    stages.clear_memo()
    stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)
    stages.clear_memo()
    stages.set_under_test(["amf_atomic"])
    stages.checkpoint("amf_atomic", {"element": "Ne"}, builder)

    text = "\n".join(stages.summary_lines())
    assert "amf_atomic" in text
    assert "1 replayed" in text and "2 computed" in text
    assert "under test" in text


def test_stored_entries_report_staleness(store, monkeypatch):
    stages.checkpoint("amf_atomic", {"element": "Ne"}, lambda: {"e": 1.0})
    entries = stages.stored_entries()
    assert len(entries) == 1 and entries[0]["stage_current"] is True
    assert entries[0]["key"] == {"element": "Ne"}

    real = stages.normalized_digest
    monkeypatch.setattr(stages, "normalized_digest",
                        lambda p: "changed" if p.name == "backend.py" else real(p))
    monkeypatch.setattr(stages, "_FINGERPRINTS", {})
    assert stages.stored_entries()[0]["stage_current"] is False


def test_purge_removes_entries(store):
    stages.checkpoint("amf_atomic", {"element": "Ne"}, lambda: {"e": 1.0})
    stages.checkpoint("scalar_scf", {"key": "ne"}, lambda: {"e": 1.0})
    assert stages.purge("scalar_scf") == 1
    assert stages.purge() == 1
    assert stages.purge() == 0
