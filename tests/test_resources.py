"""Memory and scratch budgeting.

Two kinds of test here, and the second is the load-bearing one.

**Behaviour** — a limit is read from configuration, an unconfigured driver refuses to start,
an over-large allocation is refused before it happens, and the refusal message says enough for
the user to act on it.

**Accuracy of the sizing functions** — every estimate is pinned against the ``nbytes`` of the
array it predicts, bounded on *both* sides. The lower bound is the safety property: an
estimate below reality lets a calculation start that cannot finish. The upper bound is the
usability property, and it is there because the limit is a hard error: an estimate that drifts
upward would refuse calculations that would have run, which is the failure mode the user
called out when this was designed. A sizing function that grows a "safety factor" fails here.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from kuiva.ci import strings as st
from kuiva.integrals import transform as tf
from kuiva.interface import pyscf_bridge as br
from kuiva.util import resources as res

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def budget():
    """A private budget, so a test never sees another test's reservations."""
    return res.MemoryBudget(res.ResourceLimits(memory_gb=1.0, source="test"))


# --- Configuration ------------------------------------------------------------------------
# The development repository also commits a site defaults.conf at the root; the test that
# pins it lives in tests/test_docs_local.py because that file is not part of a release.


def test_no_configuration_anywhere_refuses_to_run(tmp_path, monkeypatch):
    """User decision: no built-in memory default; a driver refuses rather than guess."""
    monkeypatch.delenv(res.ENV_MEMORY, raising=False)
    monkeypatch.setattr(res, "config_search_path", lambda: [tmp_path / "absent.conf"])
    empty = res.MemoryBudget()
    with pytest.raises(res.ConfigurationError) as excinfo:
        res.ensure_configured(budget=empty)
    message = str(excinfo.value)
    # The refusal has to teach the user how to fix it, or it is just an obstacle.
    assert res.ENV_MEMORY in message and "memory_gb" in message and "[memory]" in message


def test_precedence_explicit_beats_environment_beats_file(tmp_path, monkeypatch):
    cfg = tmp_path / "defaults.conf"
    cfg.write_text("[memory]\nmemory_gb = 3.0\nwarn_fraction = 0.5\n")
    monkeypatch.setattr(res, "config_search_path", lambda: [cfg])

    monkeypatch.delenv(res.ENV_MEMORY, raising=False)
    from_file = res.ResourceLimits.from_config()
    assert from_file.memory_gb == 3.0 and from_file.warn_fraction == 0.5

    monkeypatch.setenv(res.ENV_MEMORY, "5.0")
    assert res.ResourceLimits.from_config().memory_gb == 5.0
    assert res.ResourceLimits.from_config(9.0).memory_gb == 9.0      # explicit wins over both


def test_malformed_configuration_is_an_error_not_a_shrug(tmp_path, monkeypatch):
    cfg = tmp_path / "defaults.conf"
    cfg.write_text("this is not an ini file\n")
    monkeypatch.setattr(res, "config_search_path", lambda: [cfg])
    with pytest.raises(res.ConfigurationError):
        res.ResourceLimits.from_config()


def test_a_limit_must_be_positive():
    with pytest.raises(res.ConfigurationError):
        res.ResourceLimits(memory_gb=0.0)


# --- Reserving and refusing ---------------------------------------------------------------


def test_reservations_accumulate_and_release(budget):
    a = budget.reserve("first", 0.3)
    budget.reserve("second", 0.2)
    assert budget.resident_gb() == pytest.approx(0.5)
    assert budget.available_gb() == pytest.approx(0.5)
    budget.release(a)
    assert budget.resident_gb() == pytest.approx(0.2)


def test_a_phase_releases_what_it_reserved(budget):
    budget.reserve("kept", 0.1)
    with budget.in_phase("integrals"):
        budget.reserve("scratch array", 0.5)
        assert budget.resident_gb() == pytest.approx(0.6)
    # Phases are sequential, so the peak is not the sum over all of them.
    assert budget.resident_gb() == pytest.approx(0.1)


def test_an_oversized_allocation_is_refused_before_it_happens(budget):
    budget.reserve("committed", 0.6)
    with pytest.raises(res.MemoryLimitError) as excinfo:
        budget.reserve("too big", 0.9, note="nao = 400",
                       advice=["use density fitting", "use a smaller basis"])
    message = str(excinfo.value)
    # The user has to be able to tell "raise the limit" from "restructure the calculation",
    # and that needs the whole plan, not just the failed request.
    for expected in ("memory limit", "already committed", "shortfall", "committed",
                     "nao = 400", "use density fitting", "use a smaller basis"):
        assert expected in message
    assert "0.900" in message and "0.600" in message
    assert budget.resident_gb() == pytest.approx(0.6)      # the failure changed nothing


def test_refusal_says_whether_the_machine_could_have_taken_it(budget, monkeypatch):
    monkeypatch.setattr(res, "machine_available_gb", lambda: 64.0)
    with pytest.raises(res.MemoryLimitError) as excinfo:
        budget.reserve("big", 2.0)
    assert "would run this" in str(excinfo.value)

    monkeypatch.setattr(res, "machine_available_gb", lambda: 0.5)
    with pytest.raises(res.MemoryLimitError) as excinfo:
        budget.reserve("big", 2.0)
    assert "raising the limit will not" in str(excinfo.value)


def test_overcommit_downgrades_the_refusal(kuiva_caplog):
    b = res.MemoryBudget(res.ResourceLimits(memory_gb=1.0, source="test",
                                            allow_overcommit=True))
    b.reserve("way too big", 10.0)                     # does not raise
    assert any("allow_overcommit" in r.getMessage() for r in kuiva_caplog.records)


def test_approaching_the_limit_warns_but_proceeds(kuiva_caplog, budget):
    budget.reserve("most of it", 0.8)                  # 80% > warn_fraction 0.7
    assert any(r.levelname == "WARNING" for r in kuiva_caplog.records)
    assert budget.resident_gb() == pytest.approx(0.8)


def test_kernels_do_not_refuse_to_run_unconfigured():
    """Only drivers refuse; a kernel called in isolation still works."""
    unset = res.MemoryBudget()
    unset.require("anything", 1000.0)                  # no raise
    unset.reserve("anything", 1000.0)                  # no raise
    assert unset.transient_gb() == res.FALLBACK_BUFFER_GB


def test_transient_budget_scales_with_what_is_left(budget):
    big = budget.transient_gb()
    budget.reserve("most of it", 0.9)
    assert budget.transient_gb() < big


# --- The lifetime of a reservation --------------------------------------------------------
# A reservation lives as long as the process, because nothing here watches the array it
# accounts for. That is right for one calculation per process and wrong for a loop over
# systems, where the reservations accumulate until one member is refused against a limit its
# predecessors filled — a refusal that reads as though the machine were too small. Both the
# mechanism (the ledger fills) and the fix (a scope) are tested, not only the observable.


def _one_calculation(budget):
    """A calculation that fits comfortably on its own: 0.8 GB of a 1.0 GB limit."""
    budget.reserve("integral factors", 0.5)
    budget.reserve("MO block", 0.3)


def test_a_series_of_calculations_in_one_process_does_not_fill_the_ledger(budget):
    """The point of the whole section: a scripted series runs to the end."""
    for i in range(20):
        with res.calculation("system {}".format(i), budget=budget):
            _one_calculation(budget)
        assert budget.resident_gb() == pytest.approx(0.0)


def test_without_a_scope_the_series_is_refused_and_that_is_the_mechanism(budget):
    """The defect itself: the second member is refused although it fits on its own."""
    _one_calculation(budget)
    with pytest.raises(res.MemoryLimitError):
        _one_calculation(budget)


def test_a_scope_keeps_what_was_live_when_it_started(budget):
    """Identity, not equality: a scope releases its own arrays and nobody else's."""
    outer = budget.reserve("held across calculations", 0.2)
    with res.calculation(budget=budget):
        budget.reserve("held across calculations", 0.2)     # same label, different array
        assert budget.resident_gb() == pytest.approx(0.4)
    assert budget.resident_gb() == pytest.approx(0.2)
    assert [a is outer for a in budget._resident] == [True]


def test_a_scope_does_not_weaken_the_refusal(budget):
    """Scoping changes when a reservation ends, never whether it is checked."""
    with pytest.raises(res.MemoryLimitError):
        with res.calculation("far too big", budget=budget):
            budget.reserve("4-RDM", 400.0)


def test_clear_forgets_reservations_and_keeps_the_limit(budget):
    budget.reserve("previous calculation", 0.6)
    res.clear(budget=budget)
    assert budget.resident_gb() == pytest.approx(0.0)
    assert budget.configured                    # unlike reset(), the limit survives
    _one_calculation(budget)                    # and the next calculation runs


def test_a_refusal_names_the_previous_calculation_rather_than_the_machine(budget, monkeypatch):
    """⚠ The two situations have the same arithmetic and opposite remedies.

    "this calculation is too big" and "the last one never gave its memory back" both read as
    a shortfall against the limit. Reporting only the first is what teaches a user to raise
    the limit blindly — which then defeats the next refusal, the real one.
    """
    monkeypatch.setattr(res, "machine_available_gb", lambda: 64.0)
    budget.begin_calculation("first")
    _one_calculation(budget)
    budget.begin_calculation("second")
    with pytest.raises(res.MemoryLimitError) as excinfo:
        _one_calculation(budget)
    message = str(excinfo.value)
    assert "earlier calculation" in message                     # itemized
    assert "before this calculation began" in message           # and summed
    assert "resources.clear()" in message and "resources.calculation()" in message
    assert "fits in the limit as it stands" in message
    # The machine has 64 GB free, so the old message would have suggested a bigger limit.
    assert "would run this" not in message


def test_a_calculation_that_starts_on_a_full_ledger_warns(kuiva_caplog, budget):
    """The warning fires at the pre-flight, not several phases later in a refusal."""
    _one_calculation(budget)
    res.preflight([res.PhaseEstimate("scf", [res.PlannedAllocation("small", 0.05)])],
                  budget=budget)
    warnings = [r.getMessage() for r in kuiva_caplog.records if r.levelname == "WARNING"]
    assert any("resources.clear()" in w and "0.800 GB" in w for w in warnings)


def test_leftovers_that_threaten_nothing_stay_quiet(kuiva_caplog):
    """⚠ A warning on every second calculation is one the user learns to skip.

    Two small calculations in a roomy limit is the common, harmless case; the warning is
    reserved for the ledger being what puts a run at risk.
    """
    roomy = res.MemoryBudget(res.ResourceLimits(memory_gb=100.0, source="test"))
    _one_calculation(roomy)
    res.preflight([res.PhaseEstimate("scf", [res.PlannedAllocation("small", 0.05)])],
                  budget=roomy)
    assert not [r for r in kuiva_caplog.records
                if r.levelname == "WARNING" and "before this calculation" in r.getMessage()]


def test_a_first_calculation_is_quiet(kuiva_caplog, budget):
    """A fresh process says nothing: the warning is about leftovers, not about reserving."""
    res.preflight([res.PhaseEstimate("scf", [res.PlannedAllocation("small", 0.05)])],
                  budget=budget)
    _one_calculation(budget)
    assert not [r for r in kuiva_caplog.records
                if r.levelname == "WARNING" and "before this calculation" in r.getMessage()]


# --- Sizing functions: accurate in *both* directions --------------------------------------


def _assert_predicts(estimate_gb: float, actual_bytes: int, tol: float = 0.02):
    """The estimate must bound the array from above, but only just."""
    actual_gb = actual_bytes / res.BYTES_PER_GB
    assert estimate_gb >= actual_gb * (1.0 - 1e-9), \
        "estimate {:.6f} GB is BELOW the array's {:.6f} GB".format(estimate_gb, actual_gb)
    assert estimate_gb <= actual_gb * (1.0 + tol), \
        "estimate {:.6f} GB exceeds the array's {:.6f} GB by more than {:.0%}".format(
            estimate_gb, actual_gb, tol)


def test_array_gb_is_exact():
    a = np.empty((7, 11, 13), dtype=np.complex128)
    _assert_predicts(res.array_gb(a.shape, a.dtype), a.nbytes, tol=0.0)


def test_sector_sizing_matches_the_arrays_a_labelled_ci_holds():
    """The two arrays per-irrep selection adds: the per-determinant labels (transient, but
    large enough that a matrix-product formulation would have been the biggest array in the
    solve) and the resident determinant-to-sector index."""
    from kuiva.ci.strings import CASSpace
    from kuiva.symm.groups import C2Z
    from kuiva.symm.sectors import (SectorTable, determinant_labels, determinant_labels_gb,
                                    sector_index_gb)

    space = CASSpace(10, 5, build_map=False)
    labels = np.tile(np.array([[1], [3]], dtype=int), (5, 1))
    det = determinant_labels(space.occupations(), labels, C2Z.moduli)
    _assert_predicts(determinant_labels_gb(space.ndet, C2Z.width), det.nbytes, tol=0.0)
    table = SectorTable.build(space.occupations(), labels, C2Z)
    _assert_predicts(sector_index_gb(space.ndet), table.sector_of.nbytes, tol=0.0)


def test_array_gb_survives_sizes_that_overflow_int64():
    """A 4-RDM of 40 spinors is 40^8 elements: the *point* is to reject it, not to crash."""
    assert res.rdm_gb(40, 4) == pytest.approx(40.0 ** 8 * 16 / res.BYTES_PER_GB)
    assert np.isfinite(res.rdm_gb(64, 4))            # 64^8 * 16 B overflows int64 as a product


def test_rdm_sizing_matches_a_real_rdm():
    n = 6
    g2 = np.zeros((n,) * 4, dtype=np.complex128)
    _assert_predicts(res.rdm_gb(n, 2), g2.nbytes, tol=0.0)


def test_rdm_sizing_pins_the_nevpt2_cliff():
    """The design rules out a cumulant 4-RDM, so this is the number that decides feasibility."""
    assert res.rdm_gb(12, 4) == pytest.approx(6.41, abs=0.02)
    assert res.rdm_gb(14, 4) == pytest.approx(21.99, abs=0.05)
    assert res.rdm_gb(20, 4) == pytest.approx(381.47, abs=0.5)


def test_eri_sizing_matches_pyscfs_array():
    """The earliest check in the program, so it had better be right."""
    pyscf = pytest.importorskip("pyscf")
    mol = pyscf.gto.M(atom="Ne 0 0 0", basis="cc-pvdz", verbose=0)
    eri = mol.intor("int2e", aosym="s8")
    _assert_predicts(br.eri_memory_gb(mol.nao), eri.nbytes, tol=0.0)


def test_factor_and_mo_block_sizing_match_the_arrays():
    nao, naux, n = 12, 40, 9
    factors = tf.ThreeIndexAO(l_packed=np.zeros((naux, tf.npair_of(nao))), nao=nao,
                              origin="cholesky")
    _assert_predicts(tf.factor_memory_gb(nao, naux), factors.l_packed.nbytes, tol=0.0)
    b = np.zeros((naux, n, n), dtype=np.complex128)
    _assert_predicts(tf.mo_block_memory_gb(naux, n, n), b.nbytes, tol=0.0)


def _cas_setup(nao=6, naux=20, n_inactive=2, n_active=4):
    """A CASIntegrals over random-but-consistent arrays — dimensions are all that is tested."""
    from kuiva.mcscf.orbopt import CASIntegrals, OrbitalSpaces

    rng = np.random.default_rng(0)
    factors = tf.ThreeIndexAO(
        l_packed=rng.standard_normal((naux, tf.npair_of(nao))), nao=nao, origin="cholesky")
    n = 2 * nao
    c = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    h_ao = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h_ao = h_ao + h_ao.conj().T
    spaces = OrbitalSpaces.from_counts(n_inactive, n_active, n)
    return factors, CASIntegrals.build(factors, h_ao, c, spaces), c, spaces


def test_cas_integrals_sizing_matches_the_block_the_optimizer_holds():
    """The array that decides whether a multireference calculation fits is ``b_act``.

    ⚠ It carries one index over the **active** space, not over every spinor: the square
    ``B^P_pq`` is not built on any production path, and a plan that budgets one refuses
    calculations that would have run.
    """
    from kuiva.mcscf.orbopt import cas_integrals_memory_gb

    _, ints, _, spaces = _cas_setup()
    _assert_predicts(cas_integrals_memory_gb(ints.b_act.shape[0], ints.n_orb,
                                             spaces.n_active),
                     ints.b_act.nbytes, tol=0.0)


def test_the_second_order_transient_matches_what_a_hessian_product_holds():
    """⚠ The second-order step's peak is *three* copies of ``b_act``, not a fraction of it.

    Counted from the arrays a real Hessian-vector product allocates rather than from the
    constant: the bra-side response, the ket-side transform and their sum are all of ``b_act``'s
    shape and all live at the addition. An optimizer that escalates would otherwise allocate
    four times what the plan budgeted for it — and ``mode="auto"`` escalates on its own.
    """
    from kuiva.mcscf.orbopt import (HESSIAN_RESPONSE_COPIES, cas_integrals_memory_gb,
                                    hessian_response_memory_gb)

    factors, ints, coeff, spaces = _cas_setup()
    naux, n, n_act = ints.b_act.shape[0], ints.n_orb, spaces.n_active
    held = np.stack([np.zeros_like(ints.b_act) for _ in range(HESSIAN_RESPONSE_COPIES)])
    _assert_predicts(hessian_response_memory_gb(naux, n, n_act), held.nbytes, tol=0.0)
    assert hessian_response_memory_gb(naux, n, n_act) == pytest.approx(
        HESSIAN_RESPONSE_COPIES * cas_integrals_memory_gb(naux, n, n_act))

    # And the plan carries it, as a transient rather than as a resident array.
    phase = [p for p in br.memory_plan(nao=100, nelec=60, n_active=n_act)
             if p.name == "active-space integrals"][0]
    hess = [a for a in phase.allocations if "Hessian" in a.label][0]
    assert not hess.resident
    naux_planned = int(br.CHOLESKY_VECTORS_PER_AO * 100)
    assert hess.gb == pytest.approx(hessian_response_memory_gb(naux_planned, 200, n_act))


def test_direct_batch_sizing_matches_the_array_the_evaluator_holds():
    """The integral-direct route's working set, against a real shell-pair batch.

    ⚠ Sized with the **largest** shell on both sides, which is the worst pair that can occur
    and is what the pre-flight has to plan for; the batch actually evaluated for a given pair
    is that or smaller, so the test asserts the bound rather than an equality for one pair.
    """
    pytest.importorskip("pyscf")
    from kuiva.interface.pyscf_bridge import DirectERIMatrix, build_mole
    from kuiva.interface import Molecule

    mol = build_mole(Molecule.from_xyz_string("H 0 0 0\nF 0 0 0.917",
                                              basis="x2c-SVPall-2c"))
    src = DirectERIMatrix(mol)
    biggest = 0
    for k in range(mol.nbas):
        for l in range(k + 1):
            biggest = max(biggest, src._batch(k, l).nbytes)
    _assert_predicts(br.direct_block_memory_gb(mol.nao, src.shell_ao_max), biggest, tol=0.0)


def test_the_plan_drops_the_integral_array_on_the_direct_route():
    """What the route is for: the ``O(nao^4/8)`` line is gone, and what replaces it is a
    transient batch rather than a resident array."""
    conventional = [p for p in br.memory_plan(nao=200, nelec=100)
                    if p.name == "two-electron integrals"][0]
    direct = [p for p in br.memory_plan(nao=200, nelec=100, conventional=False,
                                        direct=True, shell_ao_max=7, n_shells=40)
              if p.name == "two-electron integrals"][0]
    assert any("ERI array" in a.label for a in conventional.allocations)
    assert not any("ERI array" in a.label for a in direct.allocations)
    batch = [a for a in direct.allocations if "shell-pair batch" in a.label][0]
    assert not batch.resident
    # ⚠ The plan states the *same* min(want, allowed) the evaluator computes. Sized from the
    # free budget alone and left unstated, the cache was measured holding gigabytes of
    # batches under a plan that claimed one.
    assert batch.gb == pytest.approx(min(br.direct_cache_memory_gb(200, 7, 40),
                                         res.BUDGET.transient_gb()))
    assert direct.resident_gb() < conventional.resident_gb()
    # The advice on the phase has to name the route that removes the array, or a user
    # refused there cannot act on it.
    assert any("cholesky-direct" in a for a in conventional.advice)


def test_amf_correction_sizing_matches_the_assembled_arrays():
    """The two-electron picture change: ``h_sf`` plus the three ``w_k``, all real."""
    from kuiva.amf.correction import correction_memory_gb, zero_correction

    c = zero_correction(23)
    _assert_predicts(correction_memory_gb(23), c.h_sf.nbytes + c.w.nbytes, tol=0.0)


def test_transform_buffer_sizing_bounds_the_real_kernel():
    """The unblocked buffer must bound what the blocked kernel actually allocates."""
    nao, nket, naux = 10, 6, 25
    per_block = tf._aux_blocksize(nao, nket, naux, tf.transform_buffer_gb(nao, nket, naux))
    assert per_block == naux                        # the whole thing fits in its own estimate
    small = tf._aux_blocksize(nao, nket, naux, tf.transform_buffer_gb(nao, nket, naux) / 5.0)
    assert 1 <= small < naux                        # a smaller budget blocks, never fails


def test_connection_sizing_matches_a_real_connections_object():
    dets = st.Determinants.from_occupations(
        [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 4), (2, 3, 4)], n_spinor=6)
    conn = st.connections(dets)
    actual = sum(a.nbytes for a in (conn.single_i, conn.single_j, conn.single_from,
                                    conn.single_to, conn.single_phase, conn.double_i,
                                    conn.double_j, conn.double_from, conn.double_to,
                                    conn.double_phase))
    _assert_predicts(st.connection_memory_gb(conn.n_single, conn.n_double), actual, tol=0.0)


def test_hamiltonian_sizing_bounds_the_sparse_matrix():
    rng = np.random.default_rng(0)
    n_spinor, n_elec = 8, 4
    dets = st.Determinants.aufbau(n_spinor, n_elec)
    from itertools import combinations
    dets = st.Determinants.from_occupations(list(combinations(range(n_spinor), n_elec)),
                                            n_spinor=n_spinor)
    h = rng.standard_normal((n_spinor, n_spinor))
    h = h + h.T
    eri = np.zeros((n_spinor,) * 4)
    ham = st.hamiltonian_matrix(dets, h.astype(np.complex128), eri.astype(np.complex128))
    stored = ham.data.nbytes + ham.indices.nbytes + ham.indptr.nbytes
    conn = st.connections(dets)
    estimate = st.hamiltonian_memory_gb(dets.ndet, conn.n_single, conn.n_double)
    # The estimate covers the COO triplets *and* the CSR, both live during conversion, so it
    # is legitimately above the CSR alone — but by a factor that is stated, not open-ended.
    assert estimate >= stored / res.BYTES_PER_GB
    assert estimate <= 4.0 * stored / res.BYTES_PER_GB


def test_determinant_sizing_matches_the_arrays():
    from itertools import combinations
    dets = st.Determinants.from_occupations(list(combinations(range(8), 4)), n_spinor=8)
    civecs = np.zeros((dets.ndet, 3), dtype=np.complex128)
    _assert_predicts(st.determinant_memory_gb(dets.ndet, 3),
                     dets.masks.nbytes + civecs.nbytes, tol=0.0)


# --- Checks fire where they are wired in --------------------------------------------------


def test_the_transform_refuses_an_oversized_mo_block(monkeypatch):
    monkeypatch.setattr(res, "BUDGET", res.MemoryBudget(
        res.ResourceLimits(memory_gb=1e-6, source="test")))
    factors = tf.ThreeIndexAO(l_packed=np.zeros((4, tf.npair_of(3))), nao=3,
                              origin="cholesky")
    c = np.zeros((6, 4), dtype=np.complex128)
    with pytest.raises(res.MemoryLimitError) as excinfo:
        tf.transform_3c(factors, c, c)
    assert "three-index MO block" in str(excinfo.value)


def test_the_connection_search_refuses_a_runaway_pair_count(monkeypatch):
    """The pair count is not knowable in advance, so it is checked as it is counted."""
    from itertools import combinations
    monkeypatch.setattr(res, "BUDGET", res.MemoryBudget(
        res.ResourceLimits(memory_gb=1e-7, source="test")))
    dets = st.Determinants.from_occupations(list(combinations(range(10), 5)), n_spinor=10)
    with pytest.raises(res.MemoryLimitError) as excinfo:
        st.connections(dets)
    assert "determinant connections" in str(excinfo.value)


def test_the_eri_reservation_refuses_a_large_basis(monkeypatch):
    monkeypatch.setattr(res, "BUDGET", res.MemoryBudget(
        res.ResourceLimits(memory_gb=0.05, source="test")))
    with pytest.raises(res.MemoryLimitError) as excinfo:
        br._reserve_eri_memory(200)                  # 0.6 GB of ERIs
    message = str(excinfo.value)
    assert "auxbasis" in message and "smaller basis" in message


# --- Pre-flight ---------------------------------------------------------------------------


def test_preflight_peaks_are_sequential_not_cumulative(budget):
    phases = [
        res.PhaseEstimate("a", [res.PlannedAllocation("kept", 0.2),
                                res.PlannedAllocation("buffer", 0.5, resident=False)]),
        res.PhaseEstimate("b", [res.PlannedAllocation("kept too", 0.2)]),
    ]
    # Cumulative would be 0.9; the truth is max(0.2+0.5, 0.4) = 0.7.
    assert res.preflight(phases, budget=budget) == pytest.approx(0.7)


def test_plan_peak_gb_is_preflights_model_without_the_side_effects(budget):
    """The automatic two-electron route selection compares plans *before* committing one to
    the pre-flight, so the two must share one peak model — asserted here by equality, on a
    plan whose cumulative total (0.9) differs from its sequential peak (0.7). And the
    comparison itself must not touch the ledger: no refusal, no calculation start."""
    phases = [
        res.PhaseEstimate("a", [res.PlannedAllocation("kept", 0.2),
                                res.PlannedAllocation("buffer", 0.5, resident=False)]),
        res.PhaseEstimate("ext", governed=False),
        res.PhaseEstimate("b", [res.PlannedAllocation("kept too", 0.2)]),
    ]
    peak = res.plan_peak_gb(phases, budget=budget)
    assert peak == pytest.approx(0.7)
    assert budget.plan_peak_gb == 0.0                 # no side effect on the ledger
    # An over-limit plan returns its peak instead of refusing: the caller is choosing
    # between plans, and a refusal belongs to the plan that is actually adopted.
    big = [res.PhaseEstimate("huge", [res.PlannedAllocation("x", 400.0)])]
    assert res.plan_peak_gb(big, budget=budget) == pytest.approx(400.0)
    assert res.preflight(phases, budget=budget) == pytest.approx(peak)


def test_preflight_refuses_and_says_which_phase(budget):
    phases = [res.PhaseEstimate("integrals", [res.PlannedAllocation("small", 0.1)]),
              res.PhaseEstimate("nevpt2", [res.PlannedAllocation("4-RDM", 400.0)],
                                advice=["reduce the active space"])]
    with pytest.raises(res.MemoryLimitError) as excinfo:
        res.preflight(phases, budget=budget)
    message = str(excinfo.value)
    assert "nevpt2" in message and "4-RDM" in message and "reduce the active space" in message


def test_preflight_records_its_estimate_for_the_summary(budget):
    res.preflight([res.PhaseEstimate("a", [res.PlannedAllocation("x", 0.4)])], budget=budget)
    assert budget.plan_peak_gb == pytest.approx(0.4)


def test_memory_plan_is_ordered_and_flags_the_ungoverned_scf():
    plan = br.memory_plan(nao=100, n_active=12, nevpt2=True)
    names = [p.name for p in plan]
    assert names[0] == "scalar X2C SCF"
    assert not plan[0].governed and plan[0].external_note
    assert all(p.governed for p in plan[1:])
    # The 4-RDM must be in the plan: it is the largest array in the program and it is
    # knowable from the active-space size alone.
    nevpt2 = [p for p in plan if p.name == "SC-NEVPT2"][0]
    assert max(a.gb for a in nevpt2.allocations) == pytest.approx(res.rdm_gb(12, 4))


def test_memory_plan_errs_high_on_the_cholesky_dimension():
    """The pre-flight's vectors-per-AO must not sit below what the decomposition produces.

    ⚠ Read from the committed cross-check data rather than written down here, because the
    count is a function of the **decomposition threshold** and the constant was once left
    behind when that threshold was tightened — an estimate silently 30% low, which is the
    direction that lets a calculation start and then fail. Tying the two together is what
    stops it happening again: tighten the threshold, regenerate, and this test fails.
    """
    records = json.loads((REPO / "tests/reference/tier2_kuiva.json").read_text())["records"]
    measured = {k: r["cholesky_naux"] / r["nao"] for k, r in records.items()}
    assert measured, "the cross-check data carries no Cholesky dimensions"
    worst = max(measured.values())
    assert br.CHOLESKY_VECTORS_PER_AO >= worst, \
        "the pre-flight estimates {} vectors per AO, below the {:.2f} measured for {}".format(
            br.CHOLESKY_VECTORS_PER_AO, worst,
            max(measured, key=lambda k: measured[k]))


def test_the_plan_budgets_the_block_the_pipeline_builds_not_the_square():
    """⚠ The regression this whole mechanism's credibility rests on.

    The pre-flight used to budget ``B^P_pq`` over every spinor against every spinor — an array
    no production path allocates — and refused calculations that would have run. A refusal on
    an array nobody builds teaches the user to raise the limit blindly, and then the next
    refusal, the real one, gets the same treatment.
    """
    from kuiva.mcscf.orbopt import cas_integrals_memory_gb

    nao, n_active, nelec = 100, 12, 60
    naux = int(br.CHOLESKY_VECTORS_PER_AO * nao)
    plan = br.memory_plan(nao=nao, n_active=n_active, nelec=nelec)
    mo = [p for p in plan if p.name == "spinor MO transform"][0]
    block = [a for a in mo.allocations if a.resident][0]
    assert block.gb == pytest.approx(cas_integrals_memory_gb(naux, 2 * nao, n_active))
    assert block.gb < 0.1 * tf.mo_block_memory_gb(naux, 2 * nao, 2 * nao)
    # The advice has to name the block that grew, or the user cannot act on the refusal.
    assert any("active space" in a for a in mo.advice)


def test_the_plan_falls_back_on_the_occupied_block_when_no_active_space_is_given():
    """Without an active space the largest block anything downstream asks for is B^P_p,occ."""
    nao, nelec = 100, 60
    naux = int(br.CHOLESKY_VECTORS_PER_AO * nao)
    mo = [p for p in br.memory_plan(nao=nao, nelec=nelec)
          if p.name == "spinor MO transform"][0]
    block = [a for a in mo.allocations if a.resident][0]
    assert block.gb == pytest.approx(tf.mo_block_memory_gb(naux, 2 * nao, nelec))
    assert "occupied" in block.note


def test_the_planned_nevpt2_blocks_bound_every_split_they_could_turn_out_to_be():
    """The inactive/virtual split is the one thing the pre-flight cannot know.

    ⚠ The failure mode of this correction is the *inverse* of the one it fixes: a plan that now
    under-estimates lets through a calculation the transform cannot allocate, and the whole
    design is refuse-before-allocate. So the assumed split is checked against every split the
    calculation could actually have, not merely against a plausible one.
    """
    from kuiva.pt.blocks import nevpt2_blocks_memory_gb

    nao, n_active, nelec = 40, 10, 24
    n, naux = 2 * nao, int(br.CHOLESKY_VECTORS_PER_AO * nao)
    phase = [p for p in br.memory_plan(nao=nao, n_active=n_active, nelec=nelec, nevpt2=True)
             if p.name == "SC-NEVPT2"][0]
    alloc = [a for a in phase.allocations if "B^P" in a.label][0]
    planned = alloc.gb

    # ⚠ The planner may not import the perturbation layer (nothing on the calculation path
    # may), so it sums the four blocks by a factorized identity instead of by asking for the
    # pair list. This is where the two are tied together: extend the pair list and this fails.
    n_i, n_a, n_v = br._nevpt2_block_split(n, nelec, n_active)
    assert planned == pytest.approx(nevpt2_blocks_memory_gb(naux, n_i, n_a, n_v))
    assert alloc.note == "{} inactive / {} active / {} virtual".format(n_i, n_a, n_v)
    # n_inactive = nelec - n_active_elec, so every admissible split is reachable this way.
    for n_active_elec in range(0, n_active + 1):
        n_i = nelec - n_active_elec
        assert planned >= nevpt2_blocks_memory_gb(naux, n_i, n_active, n - n_i - n_active)


# --- Scratch disk -------------------------------------------------------------------------


def test_scratch_limit_refuses_and_reports_both_bounds(tmp_path, monkeypatch):
    b = res.MemoryBudget(res.ResourceLimits(memory_gb=1.0, source="test", scratch_gb=0.001,
                                            scratch_dir=str(tmp_path)))
    with pytest.raises(res.ScratchLimitError) as excinfo:
        res.require_scratch("checkpoint", 5.0, budget=b, advice=["checkpoint less often"])
    message = str(excinfo.value)
    assert "scratch_gb limit" in message and "free on the filesystem" in message
    assert "checkpoint less often" in message


def test_scratch_defaults_to_tmpdir(monkeypatch, tmp_path):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    lims = res.ResourceLimits(memory_gb=1.0, source="test")
    assert lims.resolved_scratch_dir() == tmp_path.resolve()


def test_free_space_bounds_scratch_even_without_a_limit(tmp_path, monkeypatch):
    b = res.MemoryBudget(res.ResourceLimits(memory_gb=1.0, source="test",
                                            scratch_dir=str(tmp_path)))
    monkeypatch.setattr(res, "scratch_free_gb", lambda path=None: 0.5)
    with pytest.raises(res.ScratchLimitError):
        res.require_scratch("huge file", 10.0, budget=b)
    assert res.require_scratch("small file", 0.01, budget=b) == tmp_path


# --- The honesty check on the estimates ---------------------------------------------------


def test_summary_warns_when_the_plan_was_far_too_pessimistic(kuiva_caplog, monkeypatch):
    """The failure mode the hard limit creates: refusing runs that would have fitted."""
    b = res.MemoryBudget(res.ResourceLimits(memory_gb=100.0, source="test"))
    b.reserve("overestimated", 20.0)
    monkeypatch.setattr(res, "peak_rss_gb", lambda: 2.0)
    res.summary(budget=b)
    assert any("pessimistic" in r.getMessage() for r in kuiva_caplog.records)


def test_summary_warns_when_most_memory_is_unaccounted_for(kuiva_caplog, monkeypatch):
    b = res.MemoryBudget(res.ResourceLimits(memory_gb=100.0, source="test"))
    b.reserve("accounted", 1.0)
    monkeypatch.setattr(res, "peak_rss_gb", lambda: 40.0)
    res.summary(budget=b)
    assert any("not accounted for" in r.getMessage() for r in kuiva_caplog.records)


def test_summary_is_quiet_when_the_plan_matches_reality(kuiva_caplog, monkeypatch):
    b = res.MemoryBudget(res.ResourceLimits(memory_gb=100.0, source="test"))
    b.reserve("accounted", 4.0)
    monkeypatch.setattr(res, "peak_rss_gb", lambda: 5.0)
    res.summary(budget=b)
    assert not [r for r in kuiva_caplog.records if r.levelname == "WARNING"]


def test_peak_rss_is_readable_here():
    """If this ever returns None on Linux the honesty check above is silently disabled."""
    assert res.peak_rss_gb() is None or res.peak_rss_gb() > 0.0
