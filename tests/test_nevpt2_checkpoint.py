"""SC-NEVPT2 per-class checkpointing: what it stores, what it refuses, and what it saves.

The tests are ordered by what they can *fail on*.

1. **The mechanism, not the observable.** A restart that produces the right numbers proves
   nothing on its own — a run that silently recomputed everything produces them too. Two tests
   here forbid the evaluation itself (``no_class_evaluation``) and one counts the classes a
   partial restart evaluates, so "it resumed" is asserted rather than inferred.
2. **The refusals**, each with a companion that must *pass*, because a guard that cannot fire
   proves as little as one that always does.
3. **What is not in the file.** The design claim is that a checkpoint here is kilobytes and
   carries no reference; both are asserted against the file, not against the docstring.
"""
from __future__ import annotations

import contextlib

import numpy as np
import pytest

from kuiva.io.checkpoint import CheckpointError
from kuiva.pt.checkpoint import (NEVPT2Checkpoint, check_resumable, options_digest,
                                 read_nevpt2_checkpoint, reference_digest,
                                 write_nevpt2_checkpoint)
from kuiva.pt.classes import ClassResult, available_classes, excitation_class
from kuiva.pt.nevpt2 import assemble_from_checkpoint, sc_nevpt2
from kuiva.util.signals import StopRequested
from test_nevpt2 import spinor_setup

pytest.importorskip("h5py")

#: Two states, so the state loop has something to skip, and a CAS small enough that the whole
#: correction is a fraction of a second. ⚠ A checkpoint test wants many cheap `(state, class)`
#: pairs, not an expensive one.
N_STATES = 2


@pytest.fixture(scope="module")
def lih():
    return spinor_setup("Li 0 0 0; H 0 0 1.6", "sto-3g", ncas=2, nelecas=2)


@pytest.fixture(scope="module")
def states(lih):
    from kuiva.mcscf.casci import casci
    return casci(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"], lih["nelecas"],
                 n_states=N_STATES, e_nuc=lih["e_nuc"], report=False, enforce_kramers=False)


def run(lih, states, **kwargs):
    return sc_nevpt2(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"],
                     states.vectors, lih["nelecas"], energies=states.energies,
                     e_nuc=lih["e_nuc"], report=False, **kwargs)


@contextlib.contextmanager
def no_class_evaluation(monkeypatch):
    """Make every registered class raise if it is evaluated at all.

    ⚠ This is the test, not a speed-up: a resumed run that quietly recomputed the classes
    returns exactly the same numbers as one that read them, so the only way to assert that a
    restart *skipped* work is to make the work impossible.
    """
    import dataclasses

    def refuse(ctx):
        raise AssertionError("a class was evaluated when the checkpoint should have supplied it")

    # ⚠ The registry entry is a frozen dataclass, so the lookup is what is replaced rather than
    # the entry: `_evaluate_classes` resolves each class through this name.
    def refusing(name):
        return dataclasses.replace(excitation_class(name), evaluate=refuse)

    monkeypatch.setattr("kuiva.pt.nevpt2.excitation_class", refusing)
    yield


class CountingStop:
    """A stop cause that fires after ``after`` classes. Duck-types
    :class:`kuiva.util.signals.SignalStop` the way the policy takes it."""

    def __init__(self, after: int) -> None:
        self.after, self.seen, self.announced = after, 0, False

    def observe(self, wall):                        # pragma: no cover - the deadline half
        pass

    def should_stop(self, write_seconds: float = 0.0) -> bool:
        self.seen += 1
        return self.seen > self.after

    def announce(self, info, *, wrote=None, write_seconds: float = 0.0) -> None:
        self.announced = True


# --- 1. a checkpointed run is the same run ---------------------------------------------------

def test_checkpointing_changes_no_number(lih, states, tmp_path):
    """⚠ Bitwise, not to a tolerance: the checkpoint observes the run and must not enter it."""
    plain = run(lih, states)
    watched = run(lih, states, checkpoint=tmp_path / "e2.h5")

    assert np.array_equal(plain.e2, watched.e2)
    for name in plain.class_energies:
        np.testing.assert_array_equal(plain.class_energies[name], watched.class_energies[name])
    assert (tmp_path / "e2.h5").is_file()


def test_the_finished_file_holds_every_state_and_class(lih, states, tmp_path):
    run(lih, states, checkpoint=tmp_path / "e2.h5")
    stored = read_nevpt2_checkpoint(tmp_path / "e2.h5")

    assert stored.complete
    assert stored.n_states == N_STATES
    assert stored.served_classes == tuple(available_classes())
    assert stored.n_done == N_STATES * len(available_classes())
    assert stored.first_incomplete_state() == N_STATES


# --- 2. the mechanism: a restart skips what the file holds ------------------------------------

def test_a_complete_restart_evaluates_no_class_at_all(lih, states, tmp_path, monkeypatch):
    """The strongest statement this file makes: with every pair already on disk, a restart
    reproduces the correction while class evaluation is *forbidden*."""
    reference = run(lih, states, checkpoint=tmp_path / "e2.h5")

    with no_class_evaluation(monkeypatch):
        resumed = run(lih, states, restart=tmp_path / "e2.h5")

    np.testing.assert_array_equal(reference.e2, resumed.e2)
    assert resumed.complete == reference.complete


def test_a_partial_restart_finishes_exactly_what_is_missing(lih, states, tmp_path):
    """A table with the last state removed is resumed to bit-identical numbers.

    ⚠ The removal is of a **whole state**, which is what a real interruption leaves: the
    driver finishes a state before it writes the served-class list, so a half-written state is
    not a state the file will claim to have.
    """
    reference = run(lih, states, checkpoint=tmp_path / "e2.h5")
    stored = read_nevpt2_checkpoint(tmp_path / "e2.h5")
    stored.entries = {key: value for key, value in stored.entries.items()
                      if key[0] != N_STATES - 1}
    stored.e_casscf[N_STATES - 1] = np.nan
    write_nevpt2_checkpoint(tmp_path / "e2.h5", stored)
    assert not read_nevpt2_checkpoint(tmp_path / "e2.h5").complete

    # ⚠ Both arguments, and deliberately the same path: `restart=` reads and `checkpoint=`
    # writes, so a resumed run only keeps protecting itself when it is told to.
    resumed = run(lih, states, restart=tmp_path / "e2.h5", checkpoint=tmp_path / "e2.h5")

    np.testing.assert_array_equal(reference.e2, resumed.e2)
    assert read_nevpt2_checkpoint(tmp_path / "e2.h5").complete


def test_a_restart_rewrites_the_pairs_it_was_given(lih, states, tmp_path):
    """⚠ A resumed run that dies again must leave behind everything the first one did, not
    only its own share — so the restored entries are written back, not merely used."""
    run(lih, states, checkpoint=tmp_path / "first.h5")
    stored = read_nevpt2_checkpoint(tmp_path / "first.h5")
    stored.entries = {key: value for key, value in stored.entries.items() if key[0] == 0}
    stored.served_classes = tuple(available_classes())
    stored.e_casscf[1:] = np.nan
    write_nevpt2_checkpoint(tmp_path / "first.h5", stored)

    run(lih, states, restart=tmp_path / "first.h5", checkpoint=tmp_path / "second.h5")

    second = read_nevpt2_checkpoint(tmp_path / "second.h5")
    assert second.complete, "the resumed file must hold the first run's states too"


# --- 3. the refusals -------------------------------------------------------------------------

def test_a_restart_against_re_solved_ci_vectors_is_refused(lih, states, tmp_path):
    """⚠ The refusal that carries the physics: inside a degenerate manifold the CI's basis is
    arbitrary, so resuming across a re-solved reference would compute some members in one basis
    and some in another and the barycentre would belong to neither run."""
    run(lih, states, checkpoint=tmp_path / "e2.h5")
    rotated = np.array(states.vectors)
    rotated[0] *= np.exp(0.3j)                     # a phase: same state, different vector

    with pytest.raises(ValueError, match="different reference"):
        sc_nevpt2(lih["factors"], lih["h_ao"], lih["coeff"], lih["spaces"], rotated,
                  lih["nelecas"], energies=states.energies, e_nuc=lih["e_nuc"],
                  report=False, restart=tmp_path / "e2.h5")


def test_a_restart_with_a_different_setting_is_refused(lih, states, tmp_path):
    run(lih, states, checkpoint=tmp_path / "e2.h5")

    with pytest.raises(ValueError, match="different settings"):
        run(lih, states, restart=tmp_path / "e2.h5", shift=1e-3)


def test_the_same_settings_are_not_refused(lih, states, tmp_path):
    """⚠ The companion: a guard that refuses everything is not a guard."""
    run(lih, states, checkpoint=tmp_path / "e2.h5")
    run(lih, states, restart=tmp_path / "e2.h5")


def test_a_restart_with_a_different_class_list_is_refused(lih, states, tmp_path):
    run(lih, states, checkpoint=tmp_path / "e2.h5", classes=["Sijrs"])

    with pytest.raises(ValueError, match="different settings"):
        run(lih, states, restart=tmp_path / "e2.h5", classes=["Sijrs", "Sr"])


def test_a_missing_file_is_an_error_not_a_fresh_start(lih, states, tmp_path):
    """⚠ A read failure on an explicitly requested restart propagates: quietly starting over
    wastes exactly the hours the file existed to protect."""
    with pytest.raises(CheckpointError, match="no NEVPT2 checkpoint"):
        run(lih, states, restart=tmp_path / "absent.h5")


def test_a_wrong_schema_version_is_refused(lih, states, tmp_path):
    import h5py

    run(lih, states, checkpoint=tmp_path / "e2.h5")
    with h5py.File(tmp_path / "e2.h5", "r+") as handle:
        handle.attrs["schema_version"] = 99

    with pytest.raises(CheckpointError, match="schema version"):
        read_nevpt2_checkpoint(tmp_path / "e2.h5")


def test_check_resumable_refuses_a_different_state_count():
    table = NEVPT2Checkpoint(reference_key="a", options_key="b", n_states=4,
                             class_names=("Sijrs",))
    with pytest.raises(ValueError, match="holds 4 states"):
        check_resumable(table, reference_key="a", options_key="b", n_states=2,
                        class_names=("Sijrs",), path="x.h5")


# --- 4. assembling a finished correction ------------------------------------------------------

def test_a_finished_table_reassembles_the_result_without_computing(lih, states, tmp_path,
                                                                   monkeypatch):
    reference = run(lih, states, checkpoint=tmp_path / "e2.h5")

    with no_class_evaluation(monkeypatch):
        rebuilt = assemble_from_checkpoint(tmp_path / "e2.h5", report=False)

    np.testing.assert_array_equal(reference.e2, rebuilt.e2)
    np.testing.assert_array_equal(reference.e_casscf, rebuilt.e_casscf)
    np.testing.assert_array_equal(reference.total_energies, rebuilt.total_energies)
    assert rebuilt.complete == reference.complete
    assert rebuilt.n_frozen == reference.n_frozen
    assert rebuilt.n_deleted == reference.n_deleted
    for name in reference.class_energies:
        np.testing.assert_array_equal(reference.class_energies[name],
                                      rebuilt.class_energies[name])
        np.testing.assert_array_equal(reference.class_norms[name], rebuilt.class_norms[name])


def test_assembling_an_incomplete_table_is_refused(lih, states, tmp_path):
    """⚠ It would otherwise arrive as a *partial* E2 and read like a method limitation rather
    than an unfinished run."""
    run(lih, states, checkpoint=tmp_path / "e2.h5")
    stored = read_nevpt2_checkpoint(tmp_path / "e2.h5")
    stored.entries = {key: value for key, value in stored.entries.items() if key[0] == 0}
    write_nevpt2_checkpoint(tmp_path / "e2.h5", stored)

    with pytest.raises(ValueError, match="restart point, not a finished correction"):
        assemble_from_checkpoint(tmp_path / "e2.h5", report=False)


# --- 5. stopping between classes ---------------------------------------------------------------

def test_a_stop_between_states_leaves_a_resumable_file(lih, states, tmp_path):
    stopper = CountingStop(after=0)

    with pytest.raises(StopRequested, match="stopped after state 0"):
        run(lih, states, checkpoint=tmp_path / "e2.h5", signals=stopper)

    assert stopper.announced
    stored = read_nevpt2_checkpoint(tmp_path / "e2.h5")
    assert not stored.complete and stored.first_incomplete_state() == 1

    resumed = run(lih, states, restart=tmp_path / "e2.h5", checkpoint=tmp_path / "e2.h5")
    np.testing.assert_array_equal(run(lih, states).e2, resumed.e2)


# --- 6. what the file does NOT contain ----------------------------------------------------------

def test_the_file_carries_no_reference_and_stays_small(lih, states, tmp_path):
    """The design claim, asserted against the bytes: no orbitals, no CI vectors, and a size
    that does not scale with either."""
    import h5py

    run(lih, states, checkpoint=tmp_path / "e2.h5")
    with h5py.File(tmp_path / "e2.h5", "r") as handle:
        assert set(handle.keys()) == {"table", "reference"}
        assert set(handle["reference"].keys()) == {"e_casscf", "eps_inactive", "eps_virtual"}
    assert (tmp_path / "e2.h5").stat().st_size < 128 * 1024


def test_the_digests_separate_what_they_should(lih, states):
    """⚠ A digest that never differs is not a digest. Each of the four inputs is varied alone."""
    base = reference_digest(lih["coeff"], lih["spaces"], lih["nelecas"],
                            energies=states.energies, civecs=states.vectors)
    assert base == reference_digest(lih["coeff"], lih["spaces"], lih["nelecas"],
                                    energies=states.energies, civecs=states.vectors)

    moved = np.array(lih["coeff"])
    moved[0, 0] += 1e-12
    assert reference_digest(moved, lih["spaces"], lih["nelecas"], energies=states.energies,
                            civecs=states.vectors) != base
    shifted = np.array(states.energies) + 1e-12
    assert reference_digest(lih["coeff"], lih["spaces"], lih["nelecas"], energies=shifted,
                            civecs=states.vectors) != base
    rotated = np.array(states.vectors) * np.exp(0.1j)
    assert reference_digest(lih["coeff"], lih["spaces"], lih["nelecas"],
                            energies=states.energies, civecs=rotated) != base

    assert options_digest(fock="state-averaged") != options_digest(fock="state-specific")
    assert options_digest(classes=["a"]) != options_digest(classes=["a", "b"])


def test_a_class_result_round_trips_including_its_nones(tmp_path):
    """⚠ ``energy=None`` is not zero and the driver may not add it as one, so the file must
    bring back the ``None`` rather than a plausible number."""
    entry = ClassResult(name="Sijrs", norm=1.25, energy=None, n_perturbers=7,
                        n_dropped=2, min_denominator=None, min_signed_denominator=-0.5,
                        max_imaginary=3e-14)
    table = NEVPT2Checkpoint(reference_key="r", options_key="o", n_states=1,
                             class_names=("Sijrs",), served_classes=("Sijrs",),
                             entries={(0, "Sijrs"): entry},
                             e_casscf=np.array([-7.5]))
    write_nevpt2_checkpoint(tmp_path / "t.h5", table)
    back = read_nevpt2_checkpoint(tmp_path / "t.h5").entries[(0, "Sijrs")]

    assert back.energy is None and back.min_denominator is None
    assert back.norm == entry.norm and back.n_perturbers == 7 and back.n_dropped == 2
    assert back.min_signed_denominator == -0.5 and back.max_imaginary == entry.max_imaginary
