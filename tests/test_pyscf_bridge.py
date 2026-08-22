"""Tier-0/1 tests for the PySCF scalar-X2C front-end.

Tier-0: internal-consistency invariants (hermiticity, MO orthonormality, electron count).
Tier-1: cross-validation against a direct PySCF ``sfx2c1e`` run (correctness, not accuracy).
Tolerances: energies to 1e-8 Eh where an exact match is expected.
"""
import numpy as np
import pytest

from kuiva.interface import Molecule, scalar_x2c_reference, build_mole
from kuiva.interface.pyscf_bridge import run_scalar_x2c
from kuiva.util import resources as res

E_TOL = 1e-8 # Eh, meaningful energy tolerance


# --- fixtures --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def he_conv():
    mol = Molecule([("He", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c")
    return run_scalar_x2c(mol, fitting="conventional", screening="none")


@pytest.fixture(scope="module")
def ne_conv():
    mol = Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-TZVPall-2c")
    return run_scalar_x2c(mol, fitting="conventional", screening="none")


# --- Tier-0: internal consistency ------------------------------------------------------
def test_hermiticity(ne_conv):
    d = ne_conv
    assert np.allclose(d.s_ao, d.s_ao.T, atol=1e-12)
    assert np.allclose(d.h_x2c, d.h_x2c.T, atol=1e-10)


def test_overlap_is_spd(ne_conv):
    w = np.linalg.eigvalsh(ne_conv.s_ao)
    assert w.min() > 0.0                          # positive definite


def test_mo_orthonormality(ne_conv):
    d = ne_conv
    gram = d.mo_coeff.T @ d.s_ao @ d.mo_coeff     # C^T S C
    assert np.allclose(gram, np.eye(d.nmo), atol=1e-9)


def test_electron_count_from_density(ne_conv):
    d = ne_conv
    dm = (d.mo_coeff * d.mo_occ) @ d.mo_coeff.T    # sum_i occ_i c_i c_i^T
    assert np.isclose(np.einsum("ij,ji->", dm, d.s_ao), d.nelec_total, atol=1e-9)
    assert d.nelec_total == 10


def test_eri_packing_and_dims(he_conv):
    d = he_conv
    assert d.fit_route == "conventional" and d.df_cderi is None
    npair = d.nao * (d.nao + 1) // 2
    assert d.eri.shape == (npair * (npair + 1) // 2,)   # 8-fold packed


# --- Tier-1: cross-validation against direct PySCF -------------------------------------
def test_matches_direct_pyscf_sfx2c1e(he_conv):
    from pyscf import scf
    mol = build_mole(Molecule([("He", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c"))
    e_ref = scf.RHF(mol).sfx2c1e().run(conv_tol=1e-12).e_tot
    assert abs(he_conv.e_scf - e_ref) < E_TOL


def test_energy_decomposition(ne_conv):
    # E_scf should equal E_nuc + electronic; for an atom E_nuc = 0.
    assert abs(ne_conv.e_nuc) < 1e-12
    assert ne_conv.converged and ne_conv.e_scf < 0.0


def test_df_close_to_conventional():
    mol = Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-TZVPall-2c")
    e_df = run_scalar_x2c(mol, fitting="df", screening="none").e_scf
    e_cv = run_scalar_x2c(mol, fitting="conventional", screening="none").e_scf
    # x2c-JFIT is a Coulomb-only (RI-J) auxiliary, so DF-SCF exchange carries mEh-level
    # error — acceptable for a scalar *guess*; accurate 2e handling belongs to
    # kuiva.integrals.transform. Just confirm DF tracks the conventional result.
    assert abs(e_df - e_cv) < 2e-3


def test_default_route_is_cholesky_not_df():
    """Density fitting is **never** chosen automatically, even for a basis whose registry
    entry recommends an auxiliary.

    This test previously asserted the opposite. The routing was changed deliberately: the
    recommended auxiliaries are Coulomb-fitting sets, whose error in an individual transformed
    two-electron integral was measured at 1.7e-3 Eh — unbounded by any threshold, against a
    1e-8 Eh target. The Cholesky route's error *is* bounded, by a threshold the user sets, so
    it is the default in every case; DF happens only when the user asks for it.
    """
    mol = Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-TZVPall-2c")
    d = run_scalar_x2c(mol, screening="none")
    assert d.fit_route == "conventional"
    assert d.eri is not None and d.df_cderi is None and d.aux_name is None


def test_df_is_used_when_the_user_supplies_an_auxiliary():
    mol = Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-TZVPall-2c")
    d = run_scalar_x2c(mol, auxbasis="x2c-JFIT", screening="none")
    assert d.fit_route == "df" and d.aux_name == "x2c-JFIT"
    assert d.df_cderi is not None and d.eri is None


def test_auto_route_follows_the_memory_plan(monkeypatch):
    """``fitting=None`` (= ``"auto"``) resolves by the memory plan, not by a size constant.

    Stored integrals wherever their plan fits the configured limit; integral-direct where it
    does not. There is no fixed AO-count cutoff because no CPU crossover exists to place one
    at: the two routes were measured within a few per cent of each other from ~160 AOs up,
    so the ``O(nao^4/8)`` array only the stored route holds is the entire decision — and the
    one regime where the direct route is *more* efficient overall is the one where the
    stored plan is refused. The premise of each branch is asserted alongside its verdict, so
    a drift in the sizing functions fails this test loudly instead of hollowing it out.
    """
    from kuiva.interface.pyscf_bridge import _auto_fit_route, memory_plan
    nao, nelec = 105, 73                      # TiCl3 / x2c-SVPall-2c, no SCF needed

    roomy = res.MemoryBudget(res.ResourceLimits(memory_gb=64.0, source="test"))
    monkeypatch.setattr(res, "BUDGET", roomy)
    route, note = _auto_fit_route(nao, nelec=nelec)
    assert route == "conventional" and note == ""

    tight = res.MemoryBudget(res.ResourceLimits(memory_gb=0.5, source="test"))
    monkeypatch.setattr(res, "BUDGET", tight)
    conv_peak = res.plan_peak_gb(memory_plan(nao, conventional=True, nelec=nelec),
                                 budget=tight)
    dir_peak = res.plan_peak_gb(memory_plan(nao, conventional=False, direct=True,
                                            shell_ao_max=7, n_shells=13, nelec=nelec),
                                budget=tight)
    assert dir_peak <= 0.5 < conv_peak        # the premise the tight branch rests on
    route, note = _auto_fit_route(nao, nelec=nelec)
    assert route == "direct" and "chosen automatically" in note


@pytest.mark.slow
def test_auto_route_runs_direct_where_the_stored_route_would_be_refused():
    """End to end: at a limit the stored plan exceeds, the default route is the direct one,
    the calculation *runs* instead of being refused, and the result is the same reference
    (route differences are below 1e-11 Eh on the SCF energy, measured 9e-13 here)."""
    mol = Molecule([("Ti", (0.0, 0.0, 0.0)),
                    ("Cl", (2.2007128, 0.0, 0.0)),
                    ("Cl", (-1.1003564, 1.9058728, 0.0)),
                    ("Cl", (-1.1003564, -1.9058728, 0.0))],
                   basis="x2c-SVPall-2c", spin=1)
    d = run_scalar_x2c(mol, screening="none", memory_gb=0.5)
    assert d.fit_route == "direct"
    assert d.eri is None and d.factors is not None and d.factors.orbit_complete
    e_ref = run_scalar_x2c(mol, fitting="cholesky", screening="none", memory_gb=8.0).e_scf
    assert abs(d.e_scf - e_ref) < 1e-10


def test_the_direct_route_is_a_value_on_the_same_axis_and_carries_factors():
    """Three routes, one axis, one thing populated by each.

    ⚠ It is a *value* on ``fitting`` rather than a separate switch on purpose: a route that
    does not store the integrals and a route that fits them are alternatives, so pairing them
    is not a combination to refuse — it cannot be expressed.
    """
    mol = Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-TZVPall-2c")
    d = run_scalar_x2c(mol, fitting="cholesky-direct", screening="none")
    assert d.fit_route == "direct"
    assert d.eri is None and d.df_cderi is None
    assert d.factors is not None and d.factors.origin == "cholesky"
    assert d.factors.orbit_complete and d.factors.residual <= d.factors.tol
    with pytest.raises(ValueError, match="unknown fitting route"):
        run_scalar_x2c(mol, fitting="direct-cholesky", screening="none")


def test_open_shell_rohf_doublet():
    # Li atom, doublet (spin=1) -> ROHF path; 3 electrons.
    mol = Molecule([("Li", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=1)
    d = run_scalar_x2c(mol, fitting="conventional")
    assert d.converged and d.nelec == (2, 1) and d.nelec_total == 3
    gram = d.mo_coeff.T @ d.s_ao @ d.mo_coeff
    assert np.allclose(gram, np.eye(d.nmo), atol=1e-9)


def test_ingestion_is_pyscf_free(he_conv):
    # The container holds only numpy arrays / plain data (front-end boundary).
    for arr in (he_conv.s_ao, he_conv.h_x2c, he_conv.mo_coeff, he_conv.eri):
        assert isinstance(arr, np.ndarray)
    assert he_conv.mo_coeff.dtype == np.float64      # scalar guess is real


@pytest.mark.slow          # ~30 s: over the suite's "few seconds" timing guard
def test_mixed_basis_molecule():
    # Karlsruhe on a light atom + Peterson on U: both X2C, must pass consistency and run.
    mol = Molecule([("U", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.8))],
                   basis={"U": "cc-pVDZ-X2C", "O": "x2c-SVPall-2c"}, spin=2)
    d = run_scalar_x2c(mol, fitting="conventional")
    assert d.converged
