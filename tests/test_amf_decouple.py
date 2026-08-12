"""Tests for the X2C decoupling of the two-electron mean field.

This is where the physics is established, and the tests are ordered by how much they prove.
Read the ordering as a claim about what each one can and cannot catch — this project's own
ranking turned out to need correcting, and the correction is recorded here rather than
discovered again.

1. **The X2C decoupling is the same one the one-electron Hamiltonian uses.** Our ``X`` and
   ``R``, applied to the four-component core Hamiltonian, must reproduce PySCF's
   ``SpinOrbitalX2CHelper.get_hcore()`` for the same atom. A mismatch here would be silent and
   fatal: a mismatch
   here is silent and fatal; it is checked, not assumed.

2. **A vanishing mean field gives an exactly vanishing correction.** The structural statement
   that the subtraction is a difference of two things that are the same thing when there is
   nothing to correct.

3. **The non-relativistic limit.** The correction vanishes as ``c -> inf``, and — the sharper
   statement — it vanishes *as ``1 / c^2``*, measured across three decades. A term subtracted
   with the wrong operator would leave an ``O(1)`` or ``O(1/c)`` residue.

   ⚠ **But it does not test the density convention, contrary to the subtraction.** Every plausible
   variant of the density transformation (``R^-1 D R^-dag``, ``R^dag D R``, ``D_LL``,
   ``R D R^dag``) reduces to ``D_LL`` as ``R -> 1``, so all four pass the ``c -> inf`` test,
   and at the physical ``c`` all four give a correction of the same order with **spin-orbit
   splittings agreeing to 0.2%**. This was measured, not reasoned about. The original claim
   that "nothing else about the implementation can be wrong and still pass it" is false, and
   test 4 is what replaces it.

4. **The X2CAMF energy functional reproduces four-component Dirac-Coulomb**, in the same
   basis, to sub-microhartree accuracy — and it is the one observable that separates the
   density conventions, by five to six orders of magnitude:

   =====================  ============  ============  =============
   density                Ne error      Ar error      Kr error
   =====================  ============  ============  =============
   ``R^-1 D R^-dag``      **3.4e-07**   **1.3e-05**   **1.7e-04**
   ``R^dag D R``          5.7e-02       1.6e-01       -1.2e+02
   ``D_LL``               2.8e-02       2.8e-01       4.4e+00
   ``R D R^dag``          5.6e-02       5.7e-01       8.7e+00
   =====================  ============  ============  =============

   See :func:`test_energy_functional_reproduces_four_component` for the functional, and note
   the factor of one half in it: ``dG`` is a **two-electron** mean field, so it enters the
   energy with the ``1/2`` that any mean field does, while entering the Fock operator whole.

5. Physical magnitude: atomic j-splittings against the four-component reference, and their
   growth with ``Z``.
"""
import numpy as np
import pytest

from kuiva.amf import decouple
from kuiva.amf.atomic import atomic_solution, clear_cache
from kuiva.amf.backend import LIGHT_SPEED, FourComponentBlocks, get_backend
from kuiva.amf.correction import mean_field_double_counting
from kuiva.spinor.expand import (decompose_two_component, is_time_reversal_even,
                                 spin_block_diagonal, two_component_operator)

BASIS = "x2c-SVPall-2c"
HARTREE_CM = 219474.6313632


def build_atom(symbol, basis=BASIS, charge=0):
    from pyscf import gto
    return gto.M(atom=[(symbol, (0.0, 0.0, 0.0))], basis=basis, charge=charge, verbose=0)


def solve(symbol, *, uncontract=True, light_speed=None, interaction="coulomb"):
    mol = build_atom(symbol)
    return mol, atomic_solution(symbol, mol._basis[symbol], uncontract=uncontract,
                                light_speed=light_speed, interaction=interaction)


def correction_matrix(solution):
    """The raw two-component ``dG`` in the solution's target basis."""
    impl = get_backend("pyscf")
    amf = decouple.amf_atomic_correction(
        solution, lambda dm: impl.coulomb_mean_field(solution, dm))
    return amf, two_component_operator(amf.h_sf, amf.w)


@pytest.fixture(scope="module")
def ne_contracted():
    """Neon solved in its own (contracted) basis, so that every matrix in a test lives in one
    basis and a comparison against PySCF or against the four-component energy has no basis
    confound in it."""
    clear_cache()
    return solve("Ne", uncontract=False)


# --- 1. The decoupling is the one the one-electron Hamiltonian uses -----------------------

def test_our_x_and_r_reproduce_pyscf_one_electron_x2c(ne_contracted):
    """The silent-and-fatal decoupling check, made loud.

    ``X`` and ``R`` are built here from the blocks the backend returned, and are then applied
    to the *core* Hamiltonian. The result must be PySCF's own two-component X2C Hamiltonian
    for the same atom in the same basis. If it were not, the two-electron picture change would
    be done with a different decoupling from the one-electron part, and every downstream check
    would still pass.
    """
    from pyscf.x2c import x2c

    mol, solution = ne_contracted
    helper = x2c.SpinOrbitalX2CHelper(mol)
    helper.xuncontract = False              # keep PySCF in the same basis as the solution
    reference = np.asarray(helper.get_hcore())
    ours = decouple.x2c_one_electron(solution)
    assert ours.shape == reference.shape
    scale = float(np.max(np.abs(reference)))
    assert np.max(np.abs(ours - reference)) < 1e-9 * scale


def test_renormalization_satisfies_its_defining_equation(ne_contracted):
    """``R^dag S~ R = S``. Everything about the density transformation follows from this
    identity, so it is checked directly rather than inferred from a result."""
    _, solution = ne_contracted
    t, v, w, s = decouple.one_electron_integrals(solution)
    x, r = decouple.x2c_decoupling(solution)
    s_nesc = s + x.conj().T @ t @ x * (0.5 / solution.light_speed**2)
    assert np.max(np.abs(r.conj().T @ s_nesc @ r - s)) < 1e-10 * float(np.max(np.abs(s)))


def test_one_electron_integrals_are_recovered_from_the_blocks(ne_contracted):
    """``(t, v, w, s)`` read back out of the four-component blocks must be the integrals
    themselves — the property that lets this module be free of any integral library."""
    mol, solution = ne_contracted
    from pyscf.x2c import x2c

    t, v, w, s = decouple.one_electron_integrals(solution)
    assert np.max(np.abs(t - x2c._block_diag(mol.intor_symmetric("int1e_kin")))) < 1e-10
    assert np.max(np.abs(v - x2c._block_diag(mol.intor_symmetric("int1e_nuc")))) < 1e-10
    assert np.max(np.abs(s - x2c._block_diag(mol.intor_symmetric("int1e_ovlp")))) < 1e-10
    w_ref = x2c._sigma_dot(mol.intor("int1e_spnucsp"))
    assert np.max(np.abs(w - w_ref)) < 1e-8 * float(np.max(np.abs(w_ref)))


# --- 2. A vanishing mean field gives an exactly vanishing correction ----------------------

def test_zero_mean_field_gives_exactly_zero_correction(ne_contracted):
    """The structural content of the subtraction, isolated from every physical scale.

    With no two-electron mean field and no density there is nothing to picture-change and
    nothing to subtract, and the answer must be **exactly** zero — not small. This is what
    catches a term that fails to vanish with the density, which is the class of error the
    ``c -> inf`` test was supposed to catch and (see the module docstring) does not.
    """
    _, solution = ne_contracted
    import dataclasses

    zeros = FourComponentBlocks(
        ll=np.zeros_like(solution.veff.ll), ls=np.zeros_like(solution.veff.ls),
        sl=np.zeros_like(solution.veff.sl), ss=np.zeros_like(solution.veff.ss))
    stripped = dataclasses.replace(solution, veff=zeros, density=zeros)
    impl = get_backend("pyscf")
    amf = decouple.amf_atomic_correction(
        stripped, lambda dm: impl.coulomb_mean_field(stripped, dm))
    assert not amf.h_sf.any()
    assert not amf.w.any()
    # ⚠ And the compensating one-electron term with it. The decoupling now uses the
    # ``X`` of the converged Fock and pays for it with ``h1e(X_2e) - h1e(X_1e)``; with no mean
    # field the Fock *is* ``h``, so the two decouplings coincide and that term must be exactly
    # zero. It is asserted separately from the total because a compensation that failed to
    # vanish here would be cancelling against the rest rather than being absent.
    assert amf.compensation_scale == 0.0


def test_the_decoupling_source_is_a_real_choice_and_the_default_is_the_fock(ne_contracted):
    """The two conventions of :func:`kuiva.amf.decouple.x2c_decoupling`, and that they differ.

    ⚠ Both halves matter. If the two ``X`` were *equal*, the compensating term would be zero
    for a physical atom too and the decoupling-convention change would be a no-op dressed up as a
    decision —
    so the test asserts the difference is real. And the default is asserted explicitly, because
    everything the two implementations now agree to eight digits about depends on it.
    """
    _, solution = ne_contracted
    x_fock, r_fock = decouple.x2c_decoupling(solution, source="fock")
    x_1e, r_1e = decouple.x2c_decoupling(solution, source="one-electron")

    assert np.array_equal(x_fock, decouple.x2c_decoupling(solution)[0])   # the default
    scale = float(np.max(np.abs(x_1e)))
    assert float(np.max(np.abs(x_fock - x_1e))) > 1e-6 * scale, \
        "the Fock and one-electron decouplings coincide; the mean field is not entering X"
    # ...and they are both healthy decouplings, not one of them noise (a recorded rule:
    # max|X| of order 10 is healthy, 1e+3 is a null space).
    for x in (x_fock, x_1e):
        assert float(np.max(np.abs(x))) < 1e2

    with pytest.raises(ValueError, match="unknown decoupling source"):
        decouple.x2c_decoupling(solution, source="fock-but-different")


# --- 3. The non-relativistic limit --------------------------------------------------------

def test_correction_scales_as_one_over_c_squared():
    """``dG`` is a picture-change effect, so it is ``O(1/c^2)`` and nothing else.

    Measured across three decades in ``c``. This is a stronger statement than "it goes to
    zero": a wrongly subtracted *operator* would leave a residue of a different order, which a
    single large-``c`` point could not distinguish from slow convergence. The tolerance is
    loose (2%) because it is testing an exponent, not a coefficient.
    """
    clear_cache()
    scaled = {}
    compensation = {}
    for factor in (1.0, 10.0, 100.0):
        c = None if factor == 1.0 else LIGHT_SPEED * factor
        mol, solution = solve("Ne", light_speed=c)
        assert solution.converged
        amf, _ = correction_matrix(solution)
        scaled[factor] = amf.spin_orbit_scale * factor**2
        compensation[factor] = amf.compensation_scale
    reference = scaled[1.0]
    for factor, value in scaled.items():
        assert value == pytest.approx(reference, rel=0.02), factor
    # ...and the unscaled correction really does collapse: by 1e4 over two decades in c, which
    # is the same statement read the other way round.
    assert scaled[100.0] / 100.0**2 < 1e-3 * reference

    # ⚠ The compensating one-electron term of the Fock-based decoupling must die with it. Both
    # ``X`` go to zero as ``c -> inf``, so ``h1e(X_2e)`` and ``h1e(X_1e)`` converge on the same
    # matrix and their difference vanishes. Asserted as a *collapse* rather than a scaling law,
    # because unlike ``dw`` this term is a difference of two quantities that are each already
    # small, so its leading order is not something to pin an exponent on.
    assert compensation[1.0] > 0.0                              # real at the physical c
    assert compensation[100.0] < 1e-4 * compensation[1.0], compensation


"""⚠ **There is deliberately no test at an absurd speed of light, and the reason is a result.**

An earlier version asserted that the correction reaches machine zero at ``c = 1e8`` — it does,
measured at 6e-15 Eh. But at that ``c`` the four-component SCF is numerically meaningless:
PySCF's ``(SS|SS)`` and ``(SS|LL)`` contributions carry ``1/c^4`` and ``1/c^2`` prefactors that
vanish into rounding, the SCF stops converging above roughly ``100 x c``, and the density it
returns is **not spherical** (anisotropy 7e-02 for neon). The closed-shell guard of
:mod:`kuiva.amf.pyscf_dhf` now refuses that solution, which is exactly what it is for.

So the measurement stands and the test does not: it would have been asserting a structural
cancellation over an unconverged, symmetry-broken atomic solution, and calling the result a
non-relativistic limit. The ``1/c^2`` scaling above, over three decades where the SCF *does*
converge, is both the stronger statement and an honest one.
"""


# --- 4. The energy functional: the test that pins the density convention -------------------

def x2camf_energy(mol, dg_2c):
    """``E = Tr(D h_X2C) + 1/2 Tr(D dG) + 1/2 Tr(D G_nr[D]) + E_nuc``, at the X2CAMF density.

    Implemented as a two-component (GHF) SCF whose core Hamiltonian is ``h_X2C + dG`` — the
    Fock operator gets the **whole** correction, because that is what a mean field
    contributes to an operator — followed by removing ``1/2 Tr(D dG)`` from the total, because
    that is what a mean field contributes to an *energy*. Getting only one of the two right is
    the standard trap with a frozen mean-field correction, and the two are not the same number.

    ⚠ The subtraction goes through :func:`kuiva.amf.correction.mean_field_double_counting`
    rather than being written out here: the accounting belongs in the library
    so that anyone who ever wants an absolute total from a corrected Hamiltonian finds it
    instead of rediscovering the factor of two.
    """
    import scipy.linalg
    from pyscf import scf
    from pyscf.x2c import x2c

    helper = x2c.SpinOrbitalX2CHelper(mol)
    helper.xuncontract = False
    h1 = two_component_operator(*decompose_two_component(np.asarray(helper.get_hcore())))
    s = mol.intor_symmetric("int1e_ovlp")

    mf = scf.GHF(mol)
    mf.verbose = 0
    mf.conv_tol = 1e-12
    mf.max_cycle = 300
    mf.get_hcore = lambda *a, **k: h1 + dg_2c
    mf.get_ovlp = lambda *a, **k: scipy.linalg.block_diag(s, s)
    mf.kernel()
    assert mf.converged
    return float(mf.e_tot - mean_field_double_counting(dg_2c, mf.make_rdm1()))


@pytest.mark.parametrize("symbol,tolerance", [("Ne", 5e-6), ("Ar", 1e-4)])
def test_energy_functional_reproduces_four_component(symbol, tolerance):
    """**The strongest check in this tier, and it needs no external program.**

    X2CAMF is an approximation to four-component Dirac-Coulomb theory, so the statement that
    it *is* one is that its energy reproduces the four-component energy — in the same basis,
    with the same interaction, at the same speed of light. Measured: 3.4e-07 Eh for Ne and
    1.3e-05 Eh for Ar, against total energies of 128 and 528 Eh, i.e. nine significant figures.

    The tolerances above are an order of magnitude looser than the measured values, so this is
    a regression guard and not a fit. The *physically* meaningful tolerance is the 1e-8 Eh of
    the suite asks of a total energy, which X2CAMF does not reach and is not expected to: it
    is an approximation, and what this test pins is the size of the approximation.

    See the module docstring for what the three wrong density conventions give here (5.7e-02
    to 1.2e+02 Eh) — that spread is why this test exists.
    """
    clear_cache()
    mol, solution = solve(symbol, uncontract=False)
    assert solution.converged
    _, dg = correction_matrix(solution)
    assert abs(x2camf_energy(mol, dg) - solution.e_tot) < tolerance


def test_the_uncorrected_hamiltonian_is_measurably_worse(ne_contracted):
    """The same comparison with no correction at all, so the number above means something.

    Plain X2C-1e misses the four-component energy by 1.2e-03 Eh for Ne — three and a half
    orders of magnitude more than X2CAMF. Asserting the *ratio* rather than either value is
    what makes this a statement about the correction rather than about the basis.
    """
    mol, solution = ne_contracted
    _, dg = correction_matrix(solution)
    zero = np.zeros_like(dg)
    with_amf = abs(x2camf_energy(mol, dg) - solution.e_tot)
    without = abs(x2camf_energy(mol, zero) - solution.e_tot)
    assert without > 100.0 * with_amf


# --- 5. Structure and physical magnitude --------------------------------------------------

def test_correction_is_structurally_sound(ne_contracted):
    """Structural invariants **on the correction itself**, not only on the total.

    A correction that broke time-reversal symmetry would be invisible in the sum: the
    one-electron part is four orders of magnitude larger and its own residual is projected out
    at ingestion, so the corrupted Kramers splitting would surface much later, in the CI, as a
    physical-looking near-degeneracy with nothing pointing back to here.
    """
    _, solution = ne_contracted
    amf, dg = correction_matrix(solution)
    assert np.isrealobj(amf.h_sf) and np.isrealobj(amf.w)
    assert np.max(np.abs(amf.h_sf - amf.h_sf.T)) < 1e-14
    assert np.max(np.abs(amf.w + np.transpose(amf.w, (0, 2, 1)))) < 1e-14
    assert np.max(np.abs(dg - dg.conj().T)) < 1e-12
    assert is_time_reversal_even(dg, tol=1e-12)


def test_correction_does_not_move_the_shell_barycentre(ne_contracted):
    """``sigma . W`` is traceless, so the spin-orbit *part* of the correction cannot shift the
    centre of gravity of a shell — exactly as for the one-electron operator
    (``tests/test_soc_ingestion.py``). The spin-free part may and does; the two are separated
    here, which is the point of storing them separately."""
    from kuiva.integrals.transform import transform_1e
    from kuiva.orth.canonical import canonical_orthogonalization
    from kuiva.spinor.expand import expand_scalar_mos
    from pyscf import scf

    mol, solution = ne_contracted
    amf, _ = correction_matrix(solution)
    mf = scf.RHF(mol).sfx2c1e()
    mf.verbose = 0
    mf.kernel()
    ob = canonical_orthogonalization(mol.intor("int1e_ovlp"))
    sb = expand_scalar_mos(ob.to_working(mf.mo_coeff)).transform_scalar_basis(ob.x, "ao")
    nocc = int(np.sum(np.asarray(mf.mo_occ) > 0))
    cols = np.array([[2 * p, 2 * p + 1] for p in range(nocc - 3, nocc)]).ravel()
    c = sb.take(cols)

    full = np.linalg.eigvalsh(transform_1e(two_component_operator(amf.h_sf, amf.w), c))
    spin_free = np.linalg.eigvalsh(transform_1e(spin_block_diagonal(amf.h_sf), c))
    shift_cm = abs(full.mean() - spin_free.mean()) * HARTREE_CM
    assert shift_cm < 1e-6


def test_the_correction_has_a_substantial_spin_free_part(ne_contracted):
    """The two-electron **scalar** picture change is not a small side effect of this method,
    and it is something Breit-Pauli AMFI and SNSO screening factors do not describe at all. That
    is why the two parts are stored and reported separately.

    ⚠ How much larger it is depends on the basis it is measured in, so the assertion is only
    that it is the bigger of the two. Measured on Ne: **8.0e-04 Eh** spin-free against
    **5.3e-04 Eh** spin-orbit in the contracted basis, but **6.3e-03** against **5.3e-04** when
    the correction is built in the primitive basis and contracted back — a factor of eight in
    the ratio, because ``max |A_ij|`` is a matrix-element statement and contraction mixes
    primitives. Neither number is a property of the physics on its own.
    """
    _, solution = ne_contracted
    amf, _ = correction_matrix(solution)
    assert amf.spin_free_scale > amf.spin_orbit_scale


def test_correction_grows_with_nuclear_charge():
    """A picture-change effect scales steeply with ``Z``; a bug in the scaling would not."""
    clear_cache()
    scales = {}
    for symbol in ("Ne", "Ar"):
        _, solution = solve(symbol, uncontract=False)
        amf, _ = correction_matrix(solution)
        scales[symbol] = amf.spin_orbit_scale
    assert scales["Ar"] > 5.0 * scales["Ne"]


def test_gaunt_increases_the_screening():
    """Adding the Gaunt interaction to the atomic reference makes the correction **larger**.

    ⚠ Note what ``spin_orbit_scale`` measures: the magnitude of ``dw``, which is the size of
    the *screening correction*, not of the spin-orbit coupling. The Gaunt term supplies the
    spin-other-orbit interaction, a second screening channel that Dirac-Coulomb omits
    entirely, so a bigger correction is more screening and less residual coupling. The
    direction is easy to state backwards and this test is written to make that impossible.

    Measured on Ar: a factor of **1.29**.
    """
    clear_cache()
    _, coulomb = solve("Ar", uncontract=False)
    _, gaunt = solve("Ar", uncontract=False, interaction="gaunt")
    amf_c, _ = correction_matrix(coulomb)
    amf_g, _ = correction_matrix(gaunt)
    assert gaunt.interaction == "gaunt"
    ratio = amf_g.spin_orbit_scale / amf_c.spin_orbit_scale
    assert 1.0 < ratio < 2.0


def test_cancellation_is_reported_and_bounded(ne_contracted):
    """The correction is a difference of two large mean fields, so how much cancellation it
    rests on decides whether it is meaningful at the precision claimed.

    Measured on Ne: ``max |G~| = 17.2 Eh`` against ``max |dG| = 8.0e-04 Eh``, i.e. **2e4**, or
    a little over four decimal digits of cancellation. Double precision carries sixteen, so
    eleven digits survive — which is why the sub-microhartree agreement with four-component
    theory above is believable rather than luck. The bound asserted is three orders of
    magnitude looser than the measurement, because what would matter is a *qualitative* change
    (a correction that had become rounding error), not a factor of two.
    """
    _, solution = ne_contracted
    amf, _ = correction_matrix(solution)
    assert 1.0 <= amf.cancellation < 1.0e7
