"""The hyperfine field operator: the AO blocks, the picture change, and what pins them.

Kuiva computes no A tensor — it writes the **operator**
``T_{K,u} = mu_N c alpha^2 [(r_K/r_K^3) x alpha]_u`` per nucleus, picture-changed, in Eh per
nuclear magneton, and the nuclear-spin algebra belongs to the external code. So what has to be
tested is the operator, and its failure mode is the worst kind: a Hermitian, time-reversal-odd
matrix of an entirely plausible magnitude that is wrong by a factor of two, a sign, or a
permutation of its Cartesian components.

⚠ **Every check here is chosen for what it can fail on**, and the four that carry the weight are:

* **The non-relativistic-limit identity** (Tier-0 item 1). ``W^LS + W^SL`` must equal the
  Breit-Pauli PSO + spin-dipole + Fermi-contact operator assembled from *different* integrals
  through a different code path in the library. It pins the spinor mapping, the component
  ordering, the sign, the factor of one half and the relative weight of the two mechanisms at
  once; it is a **refusal** in production, and it is verified here to *break* when the mapping is
  corrupted, because a guard that cannot fail proves nothing.
* **One-electron ions against four-component theory** (Tier-0 item 2). For one electron the X2C
  positive-energy states and picture-changed expectation values reproduce the four-component ones
  *exactly*. H through Hg(79+) agree to 1e-10 relative and better. This is the check that says
  the picture change is **right**, not merely applied.
* **The hydrogen 1s contact coupling** (Tier-0 item 3). The up-spin expectation value is
  ``(8 pi/3) |psi(0)|^2 <S_z>`` analytically, and 1420 MHz once the constants are in — a number
  no convention inside this program can influence.
* **Two-sided memory sizing** (Tier-0 item 6), against a real array's ``nbytes``.
"""
import numpy as np
import pytest
import scipy.linalg

from kuiva.interface.pyscf_bridge import (HYPERFINE_IDENTITY_TOL, four_component_one_electron,
                                          hyperfine_memory_gb, ingest_property_integrals,
                                          picture_changed_dipole, picture_changed_hyperfine,
                                          picture_changed_moment, property_transform,
                                          resolve_hyperfine_nuclei)
from kuiva.spinor.expand import (pauli_decompose, sigma_dot, spin_block_diagonal,
                                 decompose_two_component, two_component_operator)
from kuiva.util.nuclei import NuclearMoment
from kuiva.util.units import G_ELECTRON, HARTREE_TO_MHZ, NUCLEAR_MAGNETON
from kuiva.x2c.decouple import (FourComponentBlocks, canonical_orth, decoupling_matrices,
                                picture_change)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def hf_mol():
    from pyscf import gto
    return gto.M(atom="H 0 0 0; F 0 0 1.7", basis="unc-sto-3g", verbose=0, unit="Bohr")


def _even_tempered(l, first, ratio, n):
    return [[l, [first * ratio ** k, 1.0]] for k in range(n)]


def _hydrogenic(symbol, z, *, with_p=False):
    """A one-electron ion in an even-tempered basis tight enough to resolve the cusp."""
    from pyscf import gto
    shells = _even_tempered(0, 0.05, 4.0, 19)
    if with_p:
        shells += _even_tempered(1, 0.1, 4.0, 14)
    return gto.M(atom=[(symbol, (0.0, 0.0, 0.0))], basis={symbol: shells},
                 charge=z - 1, spin=1, verbose=0)


# --- Tier 0, item 1: the non-relativistic-limit identity ------------------------------------

def test_the_identity_holds_to_machine_precision(hf_mol):
    """Both nuclei, with the tolerance the production refusal uses."""
    hf = picture_changed_hyperfine(hf_mol, {"H": True, "F": True})
    assert [n.label for n in hf.nuclei] == ["H1", "F2"]
    for n in hf.nuclei:
        assert n.identity_residual < 1e-13, n.label
        assert n.identity_residual < HYPERFINE_IDENTITY_TOL


def test_the_identity_holds_with_a_gaussian_nucleus_too(hf_mol):
    """⚠ The measured answer to an open question: the delta function is smeared, and so is
    every other integral in the identity.

    All four integrals involved take their ``1/r`` through the same ``with_rinv_at_nucleus``
    mechanism, so the finite-nucleus exponent reaches both sides of the identity consistently
    and there is nothing to evaluate at a second nuclear setting. Had this not held, the check
    would have had to be run at the point-nucleus setting and would then not have tested the
    operator the calculation actually uses.
    """
    from pyscf import gto
    mol = gto.M(atom="H 0 0 0; F 0 0 1.7", basis="unc-sto-3g", verbose=0, unit="Bohr",
                nucmod="gaussian")
    hf = picture_changed_hyperfine(mol, {"F": True})
    assert hf.nuclei[0].identity_residual < 1e-13
    assert hf.nuclear_model == "gaussian"


def test_the_identity_check_is_load_bearing(hf_mol, monkeypatch):
    """Corrupt the spinor -> spin-blocked mapping and the build must **raise**.

    A swap of the alpha and beta row blocks leaves the operator Hermitian and of exactly the
    right magnitude. Nothing but this identity notices.
    """
    from kuiva.interface import pyscf_bridge

    good = pyscf_bridge._hyperfine_blocks

    def swapped(xmol, atom):
        blocks = good(xmol, atom)
        n = blocks.shape[-1] // 2
        out = blocks.copy()
        out[:, :n, :], out[:, n:, :] = blocks[:, n:, :], blocks[:, :n, :]
        return out

    monkeypatch.setattr(pyscf_bridge, "_hyperfine_blocks", swapped)
    with pytest.raises(RuntimeError, match="non-relativistic-limit identity"):
        picture_changed_hyperfine(hf_mol, {"F": True})


def test_the_identity_fails_on_a_missing_factor_of_one_half(hf_mol, monkeypatch):
    """⚠ The specific error the half exists to prevent, shown to be caught.

    ``int1e_cg_sa10sp`` carries its own one half and ``int1e_sa01sp`` does not; assuming the
    wrong one doubles every hyperfine coupling in the program, which is a magnitude no
    chemist's intuition rejects.
    """
    from kuiva.interface import pyscf_bridge

    good = pyscf_bridge._hyperfine_blocks
    monkeypatch.setattr(pyscf_bridge, "_hyperfine_blocks",
                        lambda xmol, atom: 2.0 * good(xmol, atom))
    with pytest.raises(RuntimeError, match="non-relativistic-limit identity"):
        picture_changed_hyperfine(hf_mol, {"F": True})


def test_the_identity_fails_on_permuted_cartesian_components(hf_mol, monkeypatch):
    """A cyclic permutation of ``u`` keeps every invariant of the *set* of operators and moves
    every element of each one. The identity is component-wise, so it sees it."""
    from kuiva.interface import pyscf_bridge

    good = pyscf_bridge._hyperfine_blocks
    monkeypatch.setattr(pyscf_bridge, "_hyperfine_blocks",
                        lambda xmol, atom: good(xmol, atom)[[1, 2, 0]])
    with pytest.raises(RuntimeError, match="non-relativistic-limit identity"):
        picture_changed_hyperfine(hf_mol, {"F": True})


def test_the_twelve_component_integral_reproduces_the_spinor_mapping(hf_mol):
    """An independent cross-check of the ``sph2spinor_coeff`` mapping.

    ``libcint`` returns ``int1e_sa01sp`` in the spherical basis as twelve components — three
    Cartesian components times a quaternion — and the production path uses the ``_spinor`` form
    mapped by the unitary instead. The two are related by

        raw_spinor_u = i sph[u,3] (x) 1 - sum_c sph[u,c] (x) sigma_c

    which is a second, unrelated route to the same matrix and pins the mapping and the
    quaternion component order together.
    """
    from pyscf.x2c import x2c

    from kuiva.interface.pyscf_bridge import _hyperfine_blocks

    helper = x2c.SpinOrbitalX2CHelper(hf_mol)
    helper.xuncontract = True
    xmol, _ = helper.get_xmol(hf_mol)
    nao = int(xmol.nao)
    ours = 2.0 * _hyperfine_blocks(xmol, 1)                  # undo the production one half
    with xmol.with_rinv_at_nucleus(1):
        sph = np.asarray(xmol.intor("int1e_sa01sp", comp=12)).reshape(3, 4, nao, nao)
    for u in range(3):
        theirs = (spin_block_diagonal(1j * sph[u, 3].astype(complex))
                  - sigma_dot(sph[u, :3].astype(complex)))
        assert np.abs(ours[u] - theirs).max() < 1e-10 * np.abs(ours[u]).max()


# --- Tier 0, item 2: against four-component theory for one electron -------------------------

@pytest.mark.parametrize("symbol,z,with_p", [("H", 1, False), ("Ne", 10, True),
                                             ("Hg", 80, True)])
def test_one_electron_ions_reproduce_the_four_component_expectation_values(symbol, z, with_p):
    """⚠ **The check that says the picture change is right rather than merely applied.**

    For a one-electron system the exact X2C decoupling is exact: the two-component
    eigenvalues are the four-component positive-energy ones, and a picture-changed operator's
    expectation values are the four-component ones. The four-component side here is an
    *independent assembly* — the raw ``W^LS``/``W^SL`` blocks in the full ``(4 nao)`` matrix,
    evaluated over eigenvectors of the four-component one-electron problem — sharing nothing
    with the transformation but the blocks themselves.

    Compared through the phase-invariant ``Tr_block(T_u T_v)`` over the lowest Kramers pair,
    because inside a degenerate pair the individual matrices depend on the eigensolver's
    arbitrary basis and nothing else does. Light (H) through heavy (Hg(79+)), which is where a
    picture-change error would first show: the effect grows as ``Z^3``-ish and the agreement
    does not degrade.
    """
    from pyscf.x2c import x2c

    from kuiva.interface.pyscf_bridge import _hyperfine_blocks

    mol = _hydrogenic(symbol, z, with_p=with_p)
    fc = four_component_one_electron(mol, uncontract=False)
    helper = x2c.SpinOrbitalX2CHelper(mol)
    helper.xuncontract = False
    xmol, _ = helper.get_xmol(mol)
    ls = _hyperfine_blocks(xmol, 0)
    zero = np.zeros_like(ls[0])
    blocks = [FourComponentBlocks(ll=zero, ls=ls[k], sl=ls[k].conj().T, ss=zero)
              for k in range(3)]

    # -- four component: solve in the canonically orthogonalized metric, as the decoupling does
    h4, m4 = fc.hcore.assemble(), fc.overlap.assemble()
    orth = canonical_orth(m4)
    e, vec = np.linalg.eigh(orth.conj().T @ h4 @ orth)
    vec = orth @ vec
    positive = e > -fc.light_speed ** 2
    e4, c4 = e[positive], vec[:, positive]
    c4 = c4[:, np.argsort(e4)]
    t4 = [b.assemble() for b in blocks]

    # -- two component
    x, r = decoupling_matrices(fc.hcore, fc.overlap, fc.light_speed)
    e2, c2 = scipy.linalg.eigh(picture_change(fc.hcore, x, r), fc.overlap.ll)
    t2 = [picture_change(b, x, r) for b in blocks]

    assert e2[0] == pytest.approx(np.sort(e4)[0], rel=1e-9)
    b4, b2 = c4[:, :2], c2[:, :2]

    def invariant(basis, ops):
        proj = [basis.conj().T @ o @ basis for o in ops]
        return np.array([[np.trace(proj[i] @ proj[j]).real for j in range(3)]
                         for i in range(3)])

    inv4, inv2 = invariant(b4, t4), invariant(b2, t2)
    scale = float(np.abs(inv4).max())
    assert scale > 0.0
    assert np.abs(inv4 - inv2).max() / scale < 1e-9
    # Isotropic for an s ground state, whatever the program: a check on the *set* of three.
    assert np.allclose(inv4, np.diag(np.diag(inv4)), atol=1e-6 * scale)
    assert np.diag(inv4) == pytest.approx([np.diag(inv4)[0]] * 3, rel=1e-8)


# --- Tier 0, item 3: hydrogen 1s against the analytic value --------------------------------

def test_hydrogen_1s_gives_the_fermi_contact_operator_exactly():
    """``<1s up| (W^LS + W^SL)_z |1s up> = (8 pi/3) |psi(0)|^2 <S_z> = (4 pi/3) |psi(0)|^2``.

    The textbook Fermi-contact operator, with the textbook coefficient. ⚠ The most valuable
    single number here, because the right-hand side is **analytic** and involves nothing from
    this program but the operator: the factor of one half in the blocks, the sign, the spin
    weight ``S = sigma/2`` and the spinor mapping all have to be right simultaneously for it to
    come out. A missing half here is a factor of two on every hyperfine coupling the program
    will ever print.
    """
    from kuiva.interface.pyscf_bridge import _hyperfine_blocks

    mol = _hydrogenic("H", 1)
    nao = int(mol.nao)
    s = mol.intor("int1e_ovlp")
    h = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    energy, orbitals = scipy.linalg.eigh(h, s)
    assert energy[0] == pytest.approx(-0.5, abs=1e-3)
    c1s = orbitals[:, 0]
    psi0_sq = float(mol.eval_gto("GTOval", np.zeros((1, 3)))[0] @ c1s) ** 2

    ls = _hyperfine_blocks(mol, 0)
    up = np.zeros(2 * nao, dtype=complex)
    up[:nao] = c1s                                     # alpha spin in the spin-blocked layout
    expect = np.array([float((up.conj() @ (ls[k] + ls[k].conj().T) @ up).real)
                       for k in range(3)])
    assert expect[:2] == pytest.approx([0.0, 0.0], abs=1e-10 * abs(expect[2]))
    assert expect[2] == pytest.approx(0.5 * 8.0 * np.pi / 3.0 * psi0_sq, rel=1e-12)


def test_hydrogen_1s_gives_1420_megahertz():
    """The same number as a coupling constant, against the measured 1420.4 MHz.

    ⚠ It is the **constants** this adds to the check above: ``mu_N``, ``alpha^2`` taken from the
    ``c`` the integrals were built at, ``g_e/2`` on the spin part, ``g(1H)``, and the Hartree-to-
    MHz factor. The remaining ~0.2% is basis incompleteness at the cusp plus the QED, nuclear-
    structure and relativistic corrections the non-relativistic limit does not contain.
    """
    mol = _hydrogenic("H", 1)
    hf = picture_changed_hyperfine(mol, {"H": 1})
    n = hf.nuclei[0]
    nao = int(mol.nao)
    s = mol.intor("int1e_ovlp")
    h = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    _e, orbitals = scipy.linalg.eigh(h, s)
    up = np.zeros(2 * nao, dtype=complex)
    up[:nao] = orbitals[:, 0]
    t_z = float((up.conj() @ n.operator[2] @ up).real)
    # H = g_N sum_u T_u I_u = a I.S with a = 2 g_N <T_z> over the up-spin state.
    a_mhz = 2.0 * n.moment.g * t_z * HARTREE_TO_MHZ
    assert a_mhz == pytest.approx(1420.4, rel=5e-3)


# --- the anomaly, the mechanisms, and the operator's symmetry -------------------------------

def test_the_anomaly_multiplies_the_spin_mechanism_and_nothing_else(hf_mol):
    """``g_e`` scales the spin (contact/dipolar) part; the orbital part is Dirac's ``g = 2``.

    Built twice, at ``g_e`` and at exactly 2, and the difference must be ``(g_e - 2)/2`` times
    the spin-dependent operator — which is also the statement that the stored spin part is the
    thing the anomaly is applied to.
    """
    with_anomaly = picture_changed_hyperfine(hf_mol, {"F": True}).nuclei[0]
    dirac = picture_changed_hyperfine(hf_mol, {"F": True}, g_electron=2.0).nuclei[0]
    ratio = (G_ELECTRON - 2.0) / G_ELECTRON
    expected = with_anomaly.operator - ratio * with_anomaly.spin_dependent
    assert np.abs(dirac.operator - expected).max() < 1e-14 * np.abs(dirac.operator).max()
    # ~0.1% of the spin part, and the spin part dominates a contact-like nucleus.
    delta = np.abs(with_anomaly.operator - dirac.operator).max()
    assert delta / np.abs(with_anomaly.operator).max() == pytest.approx(1.16e-3, rel=0.2)


def test_the_orbital_and_spin_mechanisms_add_up_to_the_operator(hf_mol):
    """The stored spin part is a *part*: the transformation is linear, so the complement is the
    orbital mechanism and the two sum to the total. A consumer that asks "is this contact or
    orbital?" gets an answer that adds up."""
    from kuiva.interface import pyscf_bridge
    from pyscf.x2c import x2c

    helper = x2c.SpinOrbitalX2CHelper(hf_mol)
    helper.xuncontract = True
    xmol, _ = helper.get_xmol(hf_mol)
    transform = property_transform(hf_mol, approx="1e", what="test")
    n = picture_changed_hyperfine(hf_mol, {"F": True}, transform=transform).nuclei[0]

    ls = pyscf_bridge._hyperfine_blocks(xmol, 1)
    scale = NUCLEAR_MAGNETON / transform.fc.light_speed ** 2
    zero = np.zeros_like(ls[0])
    orbital = []
    for k in range(3):
        sf, _sd = pauli_decompose(ls[k])
        w = spin_block_diagonal(sf)
        orbital.append(scale * transform.fc.contract(picture_change(
            FourComponentBlocks(ll=zero, ls=w, sl=w.conj().T, ss=zero),
            transform.x, transform.r)))
    total = np.stack(orbital) + n.spin_dependent
    assert np.abs(total - n.operator).max() < 1e-14 * np.abs(n.operator).max()
    # For 19F the contact mechanism dominates; the orbital part is not zero either.
    assert np.abs(np.stack(orbital)).max() > 0.0


def test_the_operator_is_hermitian_and_time_reversal_odd(hf_mol):
    """The hyperfine field is magnetic, hence time **odd** — so its time-*even* part, which is
    what :func:`decompose_two_component` keeps, must vanish. A transposed or swapped spin block
    shows up here and in no norm or hermiticity test."""
    for n in picture_changed_hyperfine(hf_mol, {"H": True, "F": True}).nuclei:
        for k in range(3):
            m = n.operator[k]
            assert np.abs(m - m.conj().T).max() < 1e-12 * np.abs(m).max()
            even = two_component_operator(*decompose_two_component(m))
            assert np.abs(even).max() < 1e-11 * np.abs(m).max()


def test_the_operator_grows_steeply_with_nuclear_charge(hf_mol):
    """A sanity check on the *magnitude*: the contact operator samples the density at the
    nucleus, so ``19``F's operator is orders of magnitude larger than ``1``H's in the same
    molecule. It fails on an operator accidentally centred on one atom for both nuclei — which
    the identity would not catch, since it is evaluated at whatever origin was set."""
    hf = picture_changed_hyperfine(hf_mol, {"H": True, "F": True})
    h_max = np.abs(hf.get("H1").operator).max()
    f_max = np.abs(hf.get("F2").operator).max()
    assert f_max > 50.0 * h_max


def test_a_finite_nucleus_lowers_the_operator_and_more_so_for_a_heavy_one():
    """⚠ Asserted rather than assumed: ``with_rinv_at_nucleus`` really does apply the nuclear
    exponent to this integral in the pinned version, and the effect grows with ``Z``.

    The finite nucleus smears ``1/r`` with the nuclear *charge* exponent, i.e. a Gaussian
    magnetization equal to the charge distribution. A version in which the zeta did not reach
    this integral would silently produce point-nucleus hyperfine couplings inside a
    finite-nucleus calculation, with the header stating the opposite.
    """
    from pyscf import gto

    def contact(symbol, nucmod):
        mol = gto.M(atom=[(symbol, (0.0, 0.0, 0.0))],
                    basis={symbol: [[0, [3.0e7, 1.0]], [0, [30.0, 1.0]]]},
                    spin=None, charge=0, verbose=0, nucmod=nucmod)
        with mol.with_rinv_at_nucleus(0):
            return float(np.asarray(mol.intor("int1e_sa01sp", comp=12))
                         .reshape(3, 4, mol.nao, mol.nao)[2, 2, 0, 0])

    shifts = {}
    for symbol in ("H", "Ar", "Rn"):
        point, gauss = contact(symbol, None), contact(symbol, "gaussian")
        assert point != gauss, symbol
        shifts[symbol] = abs(gauss / point - 1.0)
    assert shifts["H"] < shifts["Ar"] < shifts["Rn"]


# --- the shared transform: bitwise -----------------------------------------------------------

def test_sharing_the_transform_leaves_the_moment_and_dipole_bitwise(hf_mol):
    """⚠ The load-bearing assertion of the refactor that made the hyperfine stage affordable.

    Building one four-component problem and one decoupling for every operator instead of one
    apiece may not move a bit of ``mu`` or ``d`` — every committed property reference depends on
    it. Both are deterministic functions of the molecule and the decoupling, so this holds; the
    point is that it is asserted, not argued.
    """
    shared = property_transform(hf_mol, approx="1e", what="test")
    for own, reused in ((picture_changed_moment(hf_mol, "mass").moment,
                         picture_changed_moment(hf_mol, "mass", transform=shared).moment),
                        (picture_changed_dipole(hf_mol, "mass").position,
                         picture_changed_dipole(hf_mol, "mass", transform=shared).position)):
        assert np.array_equal(own, reused)


def test_a_transform_built_for_another_decoupling_is_refused(hf_mol):
    """One file may not mix two decouplings; passing a mismatched transform is a programming
    error and is refused rather than silently honoured."""
    shared = property_transform(hf_mol, approx="1e", what="test")
    for build in (picture_changed_moment, picture_changed_dipole):
        with pytest.raises(ValueError, match="two decouplings"):
            build(hf_mol, "mass", approx="1e-dlu", transform=shared)
    with pytest.raises(ValueError, match="two decouplings"):
        picture_changed_hyperfine(hf_mol, {"F": True}, approx="1e-dlu", transform=shared)


def test_the_local_decoupling_is_accepted_and_recorded(hf_mol):
    """DLU is the bottom rung of the cost ladder and is reachable here, because the hyperfine
    operator is the most core-local property there is. ⚠ Whether a DLU-transformed hyperfine
    operator is accurate is a separate, measured question; what is asserted here is only that
    the decoupling it was built with is recorded, so a number from it can never be mistaken for
    an exact-decoupling one."""
    hf = picture_changed_hyperfine(hf_mol, {"F": True}, approx="1e-dlu")
    assert hf.decoupling == "1e-dlu"
    assert "1e-dlu" in hf.label()
    assert hf.nuclei[0].identity_residual < 1e-13      # the identity is pre-transformation


@pytest.mark.parametrize("approx", ["atom1e", "2e", "4c"])
def test_an_unsupported_decoupling_is_refused(hf_mol, approx):
    with pytest.raises(NotImplementedError, match="must be the one the Hamiltonian uses"):
        picture_changed_hyperfine(hf_mol, {"F": True}, approx=approx)


# --- Tier 0, item 6: the sizing function ----------------------------------------------------

@pytest.mark.parametrize("nao,n_nuclei", [(7, 1), (31, 3), (120, 2)])
def test_the_sizing_function_is_exact_on_both_sides(nao, n_nuclei):
    """⚠ Bounded on **both** sides against real arrays, so a sizing function that grows a
    safety factor fails: a resource estimate in this program is exact and never pads."""
    real = sum(np.empty((3, 2 * nao, 2 * nao), dtype=np.complex128).nbytes
               for _ in range(2 * n_nuclei)) / float(2 ** 30)
    assert hyperfine_memory_gb(nao, n_nuclei) == pytest.approx(real, rel=0, abs=0)


def test_the_stored_operators_match_their_declared_size(hf_mol):
    """The sizing function against what was actually allocated, which is the pairing that makes
    it a *plan* rather than a formula."""
    hf = picture_changed_hyperfine(hf_mol, {"H": True, "F": True})
    actual = sum(n.operator.nbytes + n.spin_dependent.nbytes for n in hf.nuclei)
    assert hyperfine_memory_gb(hf.nao, len(hf.nuclei)) == pytest.approx(
        actual / float(2 ** 30), rel=0, abs=0)


# --- selection: the addressing and every refusal ---------------------------------------------

@pytest.fixture(scope="module")
def two_metals():
    from pyscf import gto
    return gto.M(atom="Cu 0 0 0; Cu 0 0 4.0; O 0 0 2.0; H 2.0 0 2.0",
                 basis="sto-3g", verbose=0, unit="Bohr", spin=None, charge=0)


def test_every_addressing_form_selects_the_same_nucleus(two_metals):
    """Element, atom label and 1-based number — the addressing of
    :mod:`kuiva.basis.atommap`, shared with per-atom bases and reference configurations, so
    there is one way to name an atom in this program."""
    by_label = resolve_hyperfine_nuclei(two_metals, {"Cu2": True})
    by_number = resolve_hyperfine_nuclei(two_metals, {2: True})
    assert [n[0] for n in by_label] == [1]
    assert [n[0] for n in by_number] == [1]
    assert by_label[0][3] is by_number[0][3]
    by_element = resolve_hyperfine_nuclei(two_metals, {"Cu": 65})
    assert [n[0] for n in by_element] == [0, 1]
    assert {n[3].label for n in by_element} == {"65Cu"}


def test_isotopes_may_differ_between_two_atoms_of_one_element(two_metals):
    """Most-specific-first, exactly as a per-atom basis resolves."""
    got = resolve_hyperfine_nuclei(two_metals, {"Cu": 63, "Cu2": 65})
    assert [(n[0], n[3].label) for n in got] == [(0, "63Cu"), (1, "65Cu")]


def test_a_user_supplied_moment_reaches_the_operator(two_metals):
    """The escape hatch for an isomer, a revised moment, or an element the table lacks — and
    the operator must not depend on it beyond the size model, which is what makes the factored
    file layout work."""
    custom = NuclearMoment(spin=3.5, g=0.25, quadrupole_barn=-0.5)
    got = resolve_hyperfine_nuclei(two_metals, {"O": custom})
    assert got[0][3] is custom and got[0][0] == 2


def test_a_bare_true_is_refused_on_a_molecule(two_metals):
    """⚠ "All magnetic nuclei" is not a selection: the cost is one stored transformed operator
    per nucleus and a complex has dozens of ``1``H."""
    with pytest.raises(ValueError, match="not a selection"):
        resolve_hyperfine_nuclei(two_metals, True)
    with pytest.raises(ValueError, match="no 'default' key"):
        resolve_hyperfine_nuclei(two_metals, {"default": True})


def test_a_bare_true_is_accepted_for_a_single_atom():
    """The free ions the Tier-2 validation rests on, where it is unambiguous."""
    from pyscf import gto
    atom = gto.M(atom="F 0 0 0", basis="sto-3g", verbose=0, spin=1)
    got = resolve_hyperfine_nuclei(atom, True)
    assert [(n[0], n[3].label) for n in got] == [(0, "19F")]


def test_a_request_that_selects_nothing_is_refused(two_metals):
    """A silently empty request would give a property file with no hyperfine matrices and no
    statement of why."""
    with pytest.raises(ValueError, match="selected no atom"):
        resolve_hyperfine_nuclei(two_metals, {})


def test_a_ghost_atom_is_refused():
    """A ghost carries basis functions and no nucleus, and every consumer that needs an
    *element* skips it by an explicit test rather than by an accident of naming -- a ghost's
    own ``atom_pure_symbol`` is the ghost label, not an element symbol."""
    from pyscf import gto
    mol = gto.M(atom="F 0 0 0; ghost-F 0 0 2.0", basis="sto-3g", verbose=0, unit="Bohr",
                spin=1)
    with pytest.raises(ValueError, match="is a ghost"):
        resolve_hyperfine_nuclei(mol, {"ghost-F": True})


def test_a_key_naming_no_atom_is_refused(two_metals):
    with pytest.raises(ValueError, match="names no atom"):
        resolve_hyperfine_nuclei(two_metals, {"Fe": True})


def test_an_isotope_with_no_moment_is_refused_by_name(two_metals):
    with pytest.raises(ValueError, match="has I = 0"):
        resolve_hyperfine_nuclei(two_metals, {"O": 16})


# --- the container, the provenance and the warning ------------------------------------------

def test_the_selection_warns_about_core_polarization(hf_mol, kuiva_caplog):
    """⚠ Behaviour, not decoration: the contact part of a valence-CAS hyperfine coupling is
    qualitatively wrong for s/d spin density and for ligand nuclei, and nothing downstream can
    repair it — so the run says so at the point of selection."""
    picture_changed_hyperfine(hf_mol, {"F": True})
    text = kuiva_caplog.text
    assert "core-s spin polarization" in text and "F2 (19F)" in text
    assert "qualitatively wrong" in text


def test_the_output_stream_stays_ascii(hf_mol, kuiva_caplog):
    """The output stream is ASCII only: the file is tailed, diffed and parsed over ssh on
    machines with unknown locales. Unicode belongs in docstrings."""
    picture_changed_hyperfine(hf_mol, {"F": True})
    for record in kuiva_caplog.records:
        record.getMessage().encode("ascii")


def test_the_provenance_states_every_approximation(hf_mol):
    """⚠ A stored property file that does not say how its operators were made is not
    interpretable, and the three approximations here are not visible in any number: the
    unperturbed transformation, the missing two-electron picture change, and the nuclear model.
    """
    hf = picture_changed_hyperfine(hf_mol, {"F": True})
    p = hf.provenance()
    assert p["unit"] == "Eh per nuclear magneton"
    assert p["decoupling"] == "1e" and p["nuclear_model"] == "point"
    assert p["anomaly_included"] is True
    assert "unperturbed" in p["x2c_response"]
    assert "not applied" in p["two_electron_picture_change"]
    assert p["nuclear_magneton_au"] == NUCLEAR_MAGNETON
    assert p["nuclei"][0]["label"] == "19F" and p["nuclei"][0]["atom"] == 2
    assert p["nuclei"][0]["atom_label"] == "F2"
    assert p["nuclei"][0]["twice_spin"] == 1
    assert "Stone" in p["nuclei"][0]["source"]
    import json
    json.dumps(p)                                       # it goes into a text header verbatim


def test_the_product_dimension_is_reported_and_never_refused(hf_mol):
    """⚠ Kuiva never forms the electron-nuclear product space — that is the whole point of the
    factored layout — so its dimension is stated because the consumer's allocation is, not
    Kuiva's."""
    hf = picture_changed_hyperfine(hf_mol, {"H": True, "F": True})
    assert hf.product_dimension(6) == 6 * 2 * 2
    assert hf.product_dimension(1) == 4


def test_the_container_refuses_rather_than_returning_zeros(hf_mol):
    props = ingest_property_integrals(hf_mol)
    assert not props.has_hyperfine
    assert "hyperfine" not in props.provenance()
    with pytest.raises(ValueError, match="no hyperfine operators"):
        props.hyperfine_integrals()
    with pytest.raises(KeyError, match="no hyperfine nucleus"):
        picture_changed_hyperfine(hf_mol, {"F": True}).get("Cl")


def test_ingestion_carries_the_operators_without_the_property_picture_change(hf_mol):
    """⚠ **The recorded exception**: ``property_picture_change`` governs ``mu`` and ``d``; the
    hyperfine operators are picture-changed regardless, because the bare operator is wrong by a
    factor of 4-10 wherever s character carries spin density. The header states the treatment of
    each family separately, which is how the reason behind "one flag governs both" is still met.
    """
    props = ingest_property_integrals(hf_mol, hyperfine={"F": True})
    assert props.has_hyperfine
    assert props.picture_change is None and props.dipole_picture_change is None
    assert "always applied" in props.provenance()["hyperfine"]["picture_change"]
    assert "none (bare" in props.provenance()["picture_change"]


def test_ingestion_shares_one_transform_across_every_operator(hf_mol, monkeypatch):
    """⚠ Asserted by counting, because the cost is the reason the class exists: with the moment,
    the dipole and two nuclei all asking, there is still **one** four-component problem and one
    decoupling."""
    from kuiva.interface import pyscf_bridge

    calls = []
    real = pyscf_bridge.four_component_one_electron
    monkeypatch.setattr(pyscf_bridge, "four_component_one_electron",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    props = ingest_property_integrals(hf_mol, picture_change=True,
                                      hyperfine={"H": True, "F": True})
    assert props.has_hyperfine and props.picture_change is not None
    assert len(calls) == 1
