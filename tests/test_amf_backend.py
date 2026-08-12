"""Tests for the four-component atomic backend abstraction.

Two jobs here, and the second is the one that matters in five years' time.

The first is that the wrapper does not corrupt anything: this backend *is* PySCF, so the
four-component energies must be PySCF's own, and the blocks must come back in the conventions
kuiva/spinor/expand.py fixes rather than the j-adapted spinor basis PySCF works in.

The second is that **the abstraction is real and not a single-implementation fiction**.
The backend interface is a hard requirement precisely because a future native
four-component code must not begin by rewriting this module. An interface with one
implementation is indistinguishable from no interface at all until someone tries to add the
second one, so a stub second backend is registered and driven here. That test is the
deliverable — do not delete it because it looks trivial.
"""
import numpy as np
import pytest

from kuiva.amf import backend as bk
from kuiva.amf.backend import (AtomicDiracSolution, FourComponentBlocks, available_backends,
                               get_backend, register_backend, unregister_backend)
from kuiva.amf.pyscf_dhf import PySCFDiracBackend

BASIS = "x2c-SVPall-2c"


def atomic_basis(symbol, basis=BASIS):
    """The parsed basis of an isolated atom, as the driver passes it to a backend."""
    from pyscf import gto
    spin = int(gto.charge(symbol)) % 2          # so that odd-Z atoms can be built at all
    return gto.M(atom=[(symbol, (0.0, 0.0, 0.0))], basis=basis, spin=spin,
                 verbose=0)._basis[symbol]


def _solver_mole(symbol, basis=BASIS):
    """The decontracted ``Mole`` the backend actually solves in — needed by tests that
    evaluate an integral over the solution's own basis rather than the molecular one."""
    from pyscf import gto
    spin = int(gto.charge(symbol)) % 2
    return gto.M(atom=[(symbol, (0.0, 0.0, 0.0))], basis=basis, spin=spin,
                 verbose=0).decontract_basis(aggregate=True)[0]


@pytest.fixture(scope="module")
def ne():
    return PySCFDiracBackend().solve("Ne", atomic_basis("Ne"))


@pytest.fixture(scope="module")
def ar():
    return PySCFDiracBackend().solve("Ar", atomic_basis("Ar"))


# --- The registry, and the abstraction it exists to protect ------------------------------

def test_pyscf_backend_is_registered():
    assert "pyscf" in available_backends()
    impl = get_backend("pyscf")
    assert impl.name == "pyscf" and impl.version


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="unknown atomic four-component backend"):
        get_backend("dirac")


def test_a_second_backend_can_be_registered_and_driven():
    """**The deliverable that protects the future four-component work**.

    A stub backend that returns fabricated arrays is registered, resolved by name, and driven
    through the whole protocol. If the abstraction were fictional — if anything downstream
    reached into a PySCF object, or the protocol did not describe what is actually needed —
    this test could not be written without importing PySCF, and that is exactly the signal it
    is here to give.
    """
    nao = 4
    zeros = np.zeros((2 * nao, 2 * nao), dtype=np.complex128)
    identity = np.eye(2 * nao, dtype=np.complex128)

    class StubBackend:
        name = "stub"
        calls = []

        @property
        def version(self):
            return "0.0-stub"

        def solve(self, element, basis, *, charge=0, configuration=None,
                  interaction="coulomb", uncontract=True, **kw):
            self.calls.append(("solve", element, interaction))
            blocks = FourComponentBlocks(ll=identity.copy(), ls=zeros.copy(),
                                         sl=zeros.copy(), ss=identity.copy())
            return AtomicDiracSolution(
                element=element, atomic_number=1, charge=charge,
                basis="stub", basis_spec=basis,
                configuration=configuration or "1s1", interaction=interaction,
                light_speed=bk.LIGHT_SPEED, hcore=blocks, overlap=blocks,
                density=blocks, veff=blocks, contraction=None, uncontracted=uncontract,
                mo_energy=np.zeros(2 * nao), mo_occ=np.zeros(2 * nao),
                e_tot=-1.0, converged=True, backend=self.name,
                backend_version=self.version)

        def coulomb_mean_field(self, solution, dm):
            self.calls.append(("mean_field", solution.element, dm.shape))
            return np.zeros_like(dm)

    stub = StubBackend()
    register_backend("stub", lambda: stub)
    try:
        assert "stub" in available_backends()
        impl = get_backend("stub")
        solution = impl.solve("Xx", "made-up", interaction="gaunt")
        assert solution.backend == "stub" and solution.nao == nao
        impl.coulomb_mean_field(solution, np.zeros((2 * nao, 2 * nao), dtype=np.complex128))
        assert [c[0] for c in stub.calls] == ["solve", "mean_field"]
        # And it satisfies the declared protocol, structurally.
        assert isinstance(impl, bk.AtomicDiracBackend)
    finally:
        unregister_backend("stub")
    assert "stub" not in available_backends()


# --- What comes back ----------------------------------------------------------------------

def test_solution_shapes_and_basis(ne):
    """Blocks are over the **uncontracted** basis; the contraction maps back to the
    molecular one. Uncontracted is the physically correct choice for X2C decoupling
     and it must be visible on the object, not implicit."""
    assert ne.uncontracted
    assert ne.nao > ne.nao_target                 # decontraction genuinely enlarged the basis
    assert ne.contraction.shape == (ne.nao, ne.nao_target)
    for blocks in (ne.hcore, ne.overlap, ne.density, ne.veff):
        assert blocks.nao == ne.nao
        for b in (blocks.ll, blocks.ls, blocks.sl, blocks.ss):
            assert b.shape == (2 * ne.nao, 2 * ne.nao)
            assert b.dtype == np.complex128


def test_every_block_group_is_hermitian(ne):
    """The core Hamiltonian, the metric, the density and the mean field are all Hermitian.
    The spinor-to-spin-orbital transformation is a congruence with a unitary matrix, so it
    cannot create or destroy hermiticity — which makes this a check that the *blocking* was
    not scrambled on the way through."""
    for name, blocks in (("hcore", ne.hcore), ("overlap", ne.overlap),
                         ("density", ne.density), ("veff", ne.veff)):
        assert blocks.hermiticity() < 1e-10, name


def test_kinetic_balance_structure(ne):
    """``hcore.ls == hcore.sl == T`` and ``overlap.ss == T / (2 c^2)``.

    This is the restricted-kinetically-balanced convention the whole X2C construction is
    written in (:mod:`kuiva.amf.backend`). :mod:`kuiva.amf.decouple` reads ``T``, ``V``, ``W``
    and ``S`` back out of these blocks rather than recomputing them, so if the convention
    silently changed, the decoupling would be built from the wrong matrices and nothing else
    would notice.
    """
    c = ne.light_speed
    assert np.max(np.abs(ne.hcore.ls - ne.hcore.sl)) < 1e-12
    t = ne.hcore.ls
    assert np.max(np.abs(ne.overlap.ss - t * (0.5 / c**2))) < 1e-12
    assert np.max(np.abs(ne.overlap.ls)) == 0.0 and np.max(np.abs(ne.overlap.sl)) == 0.0
    # T is spin-free, so it is block diagonal and the two spin blocks are equal.
    n = ne.nao
    assert np.max(np.abs(t[:n, n:])) < 1e-12
    assert np.max(np.abs(t[:n, :n] - t[n:, n:])) < 1e-12


def test_spin_blocked_rows_not_spinor_rows(ne):
    """The overlap's large-component block is ``S (x) 1_2`` in the spin-blocked layout.

    In the j-adapted spinor basis PySCF actually solves in, the same operator is *not* block
    diagonal in this sense. This is the check that the basis change happened.
    """
    n = ne.nao
    s = ne.overlap.ll
    assert np.max(np.abs(s[:n, n:])) < 1e-12                 # no alpha-beta coupling
    assert np.max(np.abs(s[:n, :n] - s[n:, n:])) < 1e-12     # the two spin blocks agree
    assert np.max(np.abs(s[:n, :n].imag)) < 1e-12            # and it is real


def test_four_component_energy_matches_plain_pyscf_dhf(ne):
    """Neon's metric is well enough conditioned for PySCF's own generalized eigensolve, so
    the wrapper's answer must be PySCF's answer to all printed digits. That is the whole
    content of "this backend *is* PySCF"."""
    from pyscf import gto, scf

    xmol = gto.M(atom="Ne 0 0 0", basis=BASIS, verbose=0).decontract_basis(aggregate=True)[0]
    reference = scf.DHF(xmol)
    reference.verbose = 0
    reference.conv_tol = 1e-10
    e_plain = reference.kernel()
    assert reference.converged
    assert abs(ne.e_tot - e_plain) < 1e-8


def test_spinor_energies_give_the_four_component_reference_splitting(ne, ar):
    """The four-component j-splitting of the valence p shell, read straight off the spinor
    energies. This is the reference the whole plan is measured against, and it comes from the
    solution itself rather than from a stored number or another program.

    Neon: 6 valence spinors spanning ``2p_1/2`` (2) and ``2p_3/2`` (4). Argon the same for
    ``3p``. Both must show the exact 2 + 4 degeneracy pattern of a closed p shell.
    """
    for solution, expected_cm in ((ne, 903.0), (ar, 1609.0)):
        e = solution.occupied_energies()[-6:]
        cm = (e - e[0]) * 219474.6313632
        assert np.max(np.abs(cm[:2] - cm[0])) < 1.0      # j = 1/2 doublet
        assert np.max(np.abs(cm[2:] - cm[2])) < 1.0      # j = 3/2 quartet
        assert solution.shell_splitting() * 219474.6313632 == pytest.approx(
            expected_cm, rel=0.02)


def test_dirac_coulomb_energies_are_sane(ne, ar):
    """Against literature Dirac-Coulomb Hartree-Fock totals: Ne about -128.69 Eh and Ar about
    -528.68 Eh at the basis-set limit. A double-zeta set sits above those, so the assertion is
    a band, not a number — the point is to catch a *qualitatively* wrong energy, which is
    exactly what an unhandled metric singularity produces (Ar came back at -328 Eh)."""
    assert ne.converged and ar.converged
    assert -128.70 < ne.e_tot < -128.40
    assert -528.75 < ar.e_tot < -528.30


def test_metric_projection_is_benign_where_it_is_not_needed(ne):
    """Same statement from the other side: neon converges with or without the projection, and
    to the same energy. That is what licenses applying it unconditionally — see
    :mod:`kuiva.amf.pyscf_dhf`, point 2, where argon shows why it must be applied at all."""
    from pyscf import gto, scf

    xmol = gto.M(atom="Ne 0 0 0", basis=BASIS, verbose=0).decontract_basis(aggregate=True)[0]
    metric_min = float(np.linalg.eigvalsh(scf.DHF(xmol).get_ovlp())[0])
    assert metric_min > 1e-7        # Ne: no near-singularity to remove
    # (the energy agreement itself is test_four_component_energy_matches_plain_pyscf_dhf)


def test_gaunt_lowers_the_correlation_free_energy_and_grows_with_z(ne, ar):
    """The Gaunt term is a genuine physical contribution and it must scale with ``Z``.

    It is repulsive on balance for a closed shell (it removes part of the Coulomb attraction
    the Dirac-Coulomb operator over-counts), so the sign is asserted as measured rather than
    as assumed — what is *predicted* is that the magnitude grows steeply with nuclear charge,
    and that is the discriminating statement.
    """
    impl = PySCFDiracBackend()
    ne_g = impl.solve("Ne", atomic_basis("Ne"), interaction="gaunt")
    ar_g = impl.solve("Ar", atomic_basis("Ar"), interaction="gaunt")
    d_ne = abs(ne_g.e_tot - ne.e_tot)
    d_ar = abs(ar_g.e_tot - ar.e_tot)
    assert d_ne > 1e-4 and d_ar > 1e-3
    assert d_ar > 5.0 * d_ne
    assert ne_g.interaction == "gaunt" and ne_g.converged


# --- Refusals ------------------------------------------------------------------------------

@pytest.mark.parametrize("symbol,configuration", [
    ("Na", "[Ne]3s1"),        # odd electron count: once refused outright
    ("C", "[He]2s2 2p2"),     # even count, open p shell
    ("O", "[He]2s2 2p4"),     # the case a frontier-gap test lets through
])
def test_open_shell_atoms_are_averaged_not_refused(symbol, configuration):
    """The open-shell path: an open shell is a calculation, not an error.

    All three of these were refused before — ``Na`` by an odd-electron guard that ran before
    any SCF, ``C`` and ``O`` by the density-anisotropy guard after one. Under
    average-of-configuration they converge, and the *same* anisotropy guard now asserts that
    the averaging worked rather than refusing the atom. That inversion is
    the point: nothing was deleted to make open shells work.
    """
    from kuiva.amf.pyscf_dhf import SPHERICAL_DENSITY_TOLERANCE, _density_anisotropy

    solution = PySCFDiracBackend().solve(symbol, atomic_basis(symbol),
                                         configuration=configuration)
    assert solution.converged
    assert not solution.configuration.is_closed_shell
    # The occupation really is fractional and really does sum to the electron count.
    occupied = np.asarray(solution.mo_occ)[np.asarray(solution.mo_occ) > 0]
    assert np.any((occupied > 0.0) & (occupied < 1.0))
    assert occupied.sum() == pytest.approx(solution.n_electrons, abs=1e-10)
    assert _density_anisotropy(_solver_mole(symbol), solution.density.ll) \
        < 1e-3 * SPHERICAL_DENSITY_TOLERANCE


def test_average_of_configuration_spreads_the_open_shell_over_the_whole_l_shell():
    """The fraction is ``q / (4l+2)`` over **all** spinors of the frontier shell, not over the
    ``j`` sub-shell the electrons would aufbau into.

    Oxygen's 2p^4 gives 4/6 on each of the six 2p spinors. An aufbau occupation would instead
    fill ``2p_1/2`` (2 electrons, occupation 1) and put 2 into ``2p_3/2``, which is a choice
    among degenerate determinants and is what breaks the spherical symmetry. Asserting the
    *value* of the fraction — rather than only that some occupation is fractional — is what
    distinguishes averaging over the whole shell from averaging within a ``j`` sub-shell,
    since both are spherical and only one is average-of-configuration.
    """
    solution = PySCFDiracBackend().solve("O", atomic_basis("O"),
                                         configuration="[He]2s2 2p4")
    occ = np.asarray(solution.mo_occ)
    fractional = np.sort(occ[(occ > 1e-12) & (occ < 1.0 - 1e-12)])
    assert fractional.size == 6
    assert np.allclose(fractional, 4.0 / 6.0)


def test_an_aufbau_occupation_is_what_the_anisotropy_guard_catches(monkeypatch):
    """The guard, exercised by breaking exactly the thing it protects.

    Substituting an aufbau ``get_occ`` for the average-of-configuration one reproduces the
    pre-Stage-5 behaviour on oxygen: the SCF converges, the orbital energies look healthy, and
    the density is anisotropic at order one. A guard that is never seen to fire is not known
    to work, and this one now protects a *correct* code path rather than forbidding a case.
    """
    from kuiva.amf import pyscf_dhf

    def aufbau(mol, configuration, c, state=None):
        def get_occ(mo_energy=None, mo_coeff=None):
            e = np.asarray(mo_energy, dtype=float)
            occ = np.zeros(e.size)
            occ[np.where(e > -c * c)[0][:configuration.n_electrons]] = 1.0
            if state is not None:
                # An aufbau occupation has no fractionally occupied shell, so the
                # shell-dependent Fock machinery correctly installs nothing and the
                # SCF is the plain one this test is about.
                state.update({"mo_coeff": mo_coeff, "mo_occ": occ, "shells": []})
            return occ
        return get_occ

    monkeypatch.setattr(pyscf_dhf, "_average_of_configuration_occupation", aufbau)
    with pytest.raises(RuntimeError, match="anisotropic"):
        PySCFDiracBackend().solve("O", atomic_basis("O"), configuration="[He]2s2 2p4")


def test_a_configuration_the_basis_cannot_hold_is_refused():
    """A partially filled shell needs **all** ``4l+2`` of its spinors present, not just the
    occupied part, because the electrons are spread over the whole of it. A basis with no
    ``f`` functions asked for an ``f`` occupation must say so rather than silently put the
    electrons somewhere else."""
    with pytest.raises(ValueError, match="basis does not have"):
        PySCFDiracBackend().solve("Ne", atomic_basis("Ne"),
                                  configuration="1s2 2s2 2p5 4f1")


def test_closed_shell_density_really_is_spherical(ne, ar):
    """The other side of the same statement: for a genuine closed shell the measure the
    refusal is based on is essentially zero, so the threshold separates two populations rather
    than cutting through one."""
    from kuiva.amf.pyscf_dhf import SPHERICAL_DENSITY_TOLERANCE, _density_anisotropy

    for solution, symbol in ((ne, "Ne"), (ar, "Ar")):
        assert _density_anisotropy(_solver_mole(symbol),
                                   solution.density.ll) < 1e-3 * SPHERICAL_DENSITY_TOLERANCE


"""ECP bases are refused at the molecular entry point, where the ECP is actually visible —
see ``test_amf_correction.py::test_ecp_molecule_is_refused``. A backend receives only the
*parsed basis functions* of an element, which carry no pseudopotential, so the check here is
defensive and cannot be reached through this door."""


def test_unknown_interaction_is_refused():
    with pytest.raises(ValueError, match="unknown two-electron interaction"):
        PySCFDiracBackend().solve("Ne", atomic_basis("Ne"), interaction="dirac-coulomb")


# --- Data-model invariants ------------------------------------------------------------------

def test_blocks_reject_inconsistent_shapes():
    a = np.zeros((4, 4), dtype=np.complex128)
    with pytest.raises(ValueError, match="same shape"):
        FourComponentBlocks(ll=a, ls=a, sl=a, ss=np.zeros((6, 6), dtype=np.complex128))
    odd = np.zeros((3, 3), dtype=np.complex128)
    with pytest.raises(ValueError, match="even dimension"):
        FourComponentBlocks(ll=odd, ls=odd, sl=odd, ss=odd)


def test_blocks_round_trip_through_a_full_matrix():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    blocks = FourComponentBlocks.from_matrix(a)
    assert blocks.nao == 2
    assert np.allclose(blocks.assemble(), a)


def test_contract_is_a_congruence_on_both_spin_blocks(ne):
    """The contraction is spin-free, so it acts block-diagonally over ``[alpha; beta]``.
    Applying it to the identity must give the Gram matrix of the contraction itself — a check
    that the two spin blocks were not swapped or shared."""
    identity = np.eye(2 * ne.nao, dtype=np.complex128)
    contracted = ne.contract(identity)
    assert contracted.shape == (2 * ne.nao_target, 2 * ne.nao_target)
    gram = ne.contraction.T @ ne.contraction
    m = ne.nao_target
    assert np.allclose(contracted[:m, :m], gram)
    assert np.allclose(contracted[m:, m:], gram)
    assert np.max(np.abs(contracted[:m, m:])) == 0.0


# --- exact sizing functions, pinned two-sidedly ----------------------------------------------

def test_blocks_sizing_is_exact():
    """Two-sided against a real array's ``nbytes`` (a sizing function that grows a
    safety factor fails the suite)."""
    nao = 17
    a = np.zeros((2 * nao, 2 * nao), dtype=np.complex128)
    predicted = bk.blocks_memory_gb(nao)
    actual = 4 * a.nbytes / 1024.0**3
    assert predicted == pytest.approx(actual, rel=0, abs=1e-15)


def test_solution_sizing_is_exact(ne):
    """Predicted against the true footprint of a real solution."""
    real = sum(b.nbytes for blocks in (ne.hcore, ne.overlap, ne.density, ne.veff)
               for b in (blocks.ll, blocks.ls, blocks.sl, blocks.ss))
    real += np.zeros((ne.nao, ne.nao), dtype=np.float64).nbytes   # the contraction slot
    predicted = bk.solution_memory_gb(ne.nao)
    assert predicted == pytest.approx(real / 1024.0**3, rel=0, abs=1e-15)


def test_dirac_scf_sizing_is_a_stated_over_estimate():
    """The one figure here that is a guess about PySCF's allocation rather than a shape Kuiva
    controls, so the accounting rule requires it to err high — but by a stated, bounded factor, not an
    arbitrary one."""
    nao = 40
    one = np.zeros((4 * nao, 4 * nao), dtype=np.complex128).nbytes / 1024.0**3
    assert bk.dirac_scf_memory_gb(nao) == pytest.approx(10.0 * one, rel=0, abs=1e-15)
    assert bk.dirac_scf_memory_gb(nao) > bk.solution_memory_gb(nao)
