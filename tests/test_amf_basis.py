"""Basis generality of the atomic mean-field correction.

X2C decoupling belongs in the **primitive** basis, so every correction goes
decontract -> solve -> transform -> contract back. This file is about that round trip holding
for all four contraction families declared supported, and about the two failure modes it
has:

* a **size** mismatch, which is loud and already refused;
* an **AO ordering or normalization** mismatch, which is **silent** — it produces a Hermitian
  correction of an entirely plausible magnitude that is simply expressed over the wrong
  functions, and would be read as physics. That is what these tests exist for.

⚠ Which comparison is meaningful, and why the obvious one is not
----------------------------------------------------------------
The tempting per-family check is "does the corrected two-component splitting reproduce the
four-component one?". Done naively it does **not**, and the reason is an already-recorded trap
reappearing one level up: the two-component calculation lives in
the **contracted** molecular basis while the four-component reference lives in the
**decontracted** basis the solver used, and those are different spaces. Measured that way the
residual is +0.27% for Ne in ``x2c-SVPall-2c`` and **-16%** for Ca in ``cc-pVDZ-X2C`` — almost
entirely the basis change, not the correction.

The family-independent statement is the like-for-like one, and it comes in two useful forms:

======================================== ==========================================
comparison                               what it measures
======================================== ==========================================
uncontracted molecule vs uncontracted 4c the correction itself
contracted molecule vs **contracted** 4c what decoupling in a contracted basis costs
======================================== ==========================================

Both are asserted below. The raw cross-family spread of the splitting *is* also reported, so
the basis-set error stays visible rather than being hidden inside a tolerance — Ne's 2p
splitting is 903, 984 and 998 cm^-1 in ``x2c-SVPall-2c``, ``dyallv2z`` and ANO-RCC
respectively, and no picture-change correction can or should recover a basis truncation.

Cost
----
The per-family checks that need no SCF (classification, the contraction round trip, and the
decoupling against PySCF's own X2C) are built from one-electron integrals and run in
milliseconds, so they cover every family including the expensive ones. The checks that need a
four-component solve are on Ne (sub-second) in the default suite; Ca, ANO-RCC and the
triple-zeta mixed sets are marked ``slow`` (6-50 s each).
"""
import numpy as np
import pytest
from pyscf import gto
from pyscf.x2c import x2c

from kuiva.amf import amf_correction
from kuiva.amf.atomic import atomic_solution, clear_cache
from kuiva.amf.backend import AtomicDiracSolution, FourComponentBlocks
from kuiva.amf.decouple import x2c_one_electron
from kuiva.amf.pyscf_dhf import CONTRACTION_TOL, PySCFDiracBackend, _validate_contraction
from kuiva.basis.registry import Contraction, classify_contraction, resolve_for_pyscf

HARTREE_CM = 219474.6313632
LIGHT_SPEED = 137.035999084

#: One representative of each contraction family, with the verdict measured from the parsed
#: basis data. See :func:`kuiva.basis.registry.classify_contraction` for why two of these
#: disagree with the family-level label in the registry, and why that is informative.
FAMILIES = [
    ("x2c-SVPall-2c", "Ne", Contraction.SEGMENTED),
    ("x2c-SVPall-2c", "Ti", Contraction.SEGMENTED),
    ("x2c-TZVPPall-2c", "Ne", Contraction.MIXED),        # segmented s/p, uncontracted d
    ("dyallv2z", "Ne", Contraction.UNCONTRACTED),
    ("dyallv2z", "Kr", Contraction.UNCONTRACTED),
    ("ANO-RCC", "Ne", Contraction.GENERAL),
    ("ANO-RCC", "Ti", Contraction.GENERAL),
    ("cc-pVDZ-X2C", "Ca", Contraction.GENERAL),
    ("cc-pVTZ-X2C", "Ce", Contraction.MIXED),            # general through f, uncontracted g
    ("cc-pVTZ-X2C", "U", Contraction.MIXED),
    ("cc-pwCVTZ-X2C", "Ca", Contraction.MIXED),
]


def build(name, element):
    """The molecular ``Mole`` for one element in one registry family."""
    return gto.M(atom=[(element, (0.0, 0.0, 0.0))],
                 basis=resolve_for_pyscf(name, [element]),
                 spin=int(gto.charge(element)) % 2, verbose=0)


def one_electron_solution(mol, uncontract=True):
    """An :class:`AtomicDiracSolution` carrying **only** the one-electron blocks.

    Built straight from integrals, with the density and mean field zero and no SCF run at all.
    That is legitimate here because the X2C decoupling is a property of the one-electron
    problem alone: ``X`` and ``R`` come from ``(t, v, w, s)``, which is exactly what the
    backend's block convention encodes. It makes the per-family sweep below cost milliseconds
    instead of minutes, so it can cover ``cc-pVTZ-X2C`` uranium — 266 primitives, whose
    four-component SCF is well over the ten-minute line — rather than stopping
    at the atoms that happen to be affordable.
    """
    element = mol.atom_pure_symbol(0)
    xmol, contraction = PySCFDiracBackend._build_mole(
        element, mol._basis[mol.atom_symbol(0)], 0, uncontract)
    t = x2c._block_diag(xmol.intor_symmetric("int1e_kin"))
    v = x2c._block_diag(xmol.intor_symmetric("int1e_nuc"))
    s = x2c._block_diag(xmol.intor_symmetric("int1e_ovlp"))
    w = x2c._sigma_dot(xmol.intor("int1e_spnucsp"))
    zero = np.zeros_like(t)
    return AtomicDiracSolution(
        element=element, atomic_number=int(xmol.atom_charge(0)), charge=0,
        basis=name_of(mol), basis_spec=mol._basis[mol.atom_symbol(0)],
        configuration=None, interaction="coulomb", light_speed=LIGHT_SPEED,
        hcore=FourComponentBlocks(ll=v, ls=t, sl=t, ss=w * (0.25 / LIGHT_SPEED**2) - t),
        overlap=FourComponentBlocks(ll=s, ls=zero, sl=zero,
                                    ss=t * (0.5 / LIGHT_SPEED**2)),
        density=FourComponentBlocks(ll=zero, ls=zero, sl=zero, ss=zero),
        veff=FourComponentBlocks(ll=zero, ls=zero, sl=zero, ss=zero),
        contraction=contraction, uncontracted=uncontract,
        mo_energy=np.zeros(1), mo_occ=np.zeros(1), e_tot=0.0, converged=True,
        backend="one-electron probe", backend_version="0")


def name_of(mol):
    return "probe"


# --- The measured contraction type ---------------------------------------------------------

@pytest.mark.parametrize("name,element,expected", FAMILIES)
def test_contraction_type_is_measured_from_the_data(name, element, expected):
    """The classifier is what lets a correction record *what it was actually given*.

    ⚠ The discriminator is primitive **sharing**, not the shape of the stored block: Basis Set
    Exchange emits a segmented set as one block with a block-diagonal coefficient matrix, so
    Karlsruhe neon arrives as "one s shell, 7 primitives, 3 contractions" and looks generally
    contracted. Counting shapes rather than sharing classified every Karlsruhe set as general,
    which is what this parametrization pins.
    """
    mol = build(name, element)
    assert classify_contraction(mol._basis[mol.atom_symbol(0)]) is expected


def test_every_declared_contraction_family_is_represented():
    """Four contraction families are declared supported. A test set that quietly
    lost one — the mixed case in particular, the one expected to break
    first — would look exactly like a passing suite."""
    covered = {expected for _, _, expected in FAMILIES}
    assert covered == set(Contraction)


# --- The contract-back round trip ------------------------------------------------------------

@pytest.mark.parametrize("name,element,_expected", FAMILIES)
def test_decontraction_round_trips_for_every_family(name, element, _expected):
    """``C^T A_primitive C == A_molecular``, for two different one-electron operators.

    This is the claim the whole contract-back path rests on, checked rather than assumed. Two
    operators are used deliberately: the overlap is what a normalization error breaks, and the
    nuclear attraction has a different radial weighting, so a permutation that happened to
    preserve one could not also preserve the other. Measured residual across all families:
    1e-16 to 1e-14, worst for ANO-RCC.
    """
    mol = build(name, element)
    xmol, contraction = PySCFDiracBackend._build_mole(
        mol.atom_pure_symbol(0), mol._basis[mol.atom_symbol(0)], 0, True)
    assert contraction.shape == (xmol.nao, mol.nao)
    assert _validate_contraction(mol, xmol, contraction) < CONTRACTION_TOL


def test_a_scrambled_contraction_is_caught_not_absorbed():
    """The silent failure mode, induced on purpose.

    Permuting the rows of the contraction matrix leaves its shape intact, leaves the resulting
    correction Hermitian, and leaves its magnitude entirely plausible — nothing downstream
    would notice. The round-trip check is the only thing standing between that and a wrong
    answer attributed to the physics, so it has to be seen to fire.
    """
    mol = build("x2c-SVPall-2c", "Ne")
    xmol, contraction = PySCFDiracBackend._build_mole(
        "Ne", mol._basis["Ne"], 0, True)
    scrambled = contraction[::-1].copy()
    with pytest.raises(NotImplementedError, match="does not round-trip"):
        _validate_contraction(mol, xmol, scrambled)


def test_the_contraction_matrix_is_taken_from_the_basis_not_reconstructed():
    """⚠ **The basis where a naive implementation fails**, and the reason the requirement says
    the coefficients must come from the basis rather than be rebuilt.

    ``decontract_basis(aggregate=True)`` **merges primitives shared between different shells of
    the same angular momentum**. A per-shell block-diagonal reconstruction — the obvious way to
    build a contraction matrix, and one with exactly the right shape for most bases — is then
    wrong, because two shells map onto overlapping columns of the primitive set rather than
    onto disjoint ones.

    The basis here reuses one exponent in two ``s`` shells and one in two ``p`` shells, which
    is not contrived: it is what a diffuse-augmented set looks like when the augmenting
    function duplicates a primitive already present. Aggregation collapses 12 naive primitives
    to 8, and the naive matrix does not even have the right shape — while the matrix taken from
    the basis round-trips to machine precision.
    """
    custom = [[0, [1.0, 1.0]],
              [0, [1.0, 1.0], [0.3, 1.0]],
              [1, [0.8, 1.0]],
              [1, [0.8, 0.6], [0.2, 0.4]]]
    mol = gto.M(atom="He 0 0 0", basis={"He": custom}, verbose=0)
    xmol, contraction = PySCFDiracBackend._build_mole("He", custom, 0, True)

    naive_primitives = sum(len([r for r in sh[1:] if isinstance(r, (list, tuple))])
                           * (2 * sh[0] + 1) for sh in custom)
    assert naive_primitives == 12                 # what a per-shell reconstruction would give
    assert xmol.nao == 8                          # what aggregation actually gives
    assert contraction.shape == (8, mol.nao)
    assert _validate_contraction(mol, xmol, contraction) < CONTRACTION_TOL


# --- Check (i): the decoupling reproduces PySCF's own X2C, in every family --------------------

@pytest.mark.parametrize("name,element,_expected", FAMILIES)
@pytest.mark.parametrize("uncontract", [True, False])
def test_the_decoupling_reproduces_pyscf_x2c_in_every_family(name, element, _expected,
                                                             uncontract):
    """**The substantive per-family check.**

    ``X`` and ``R`` are built from the blocks the backend's conventions define, and applied to
    the core Hamiltonian they must give PySCF's own two-component X2C Hamiltonian for the same
    atom in the same basis. If they did not, the two-electron picture change would be done
    with a different decoupling from the one-electron part and every downstream check would
    still pass.

    This is also what settles the earlier worry, the worry that
    ``pyscf_dhf.METRIC_LINDEP_THRESHOLD`` and ``decouple._canonical_orth``'s 1e-14 were tuned
    on decontracted Karlsruhe sets and might not transfer. They do: the worst relative
    residual over every family here, contracted and decontracted, is **6e-9** — on uranium in
    ``cc-pVTZ-X2C``, whose 266-primitive decontracted metric is the most nearly singular case
    in the set. Dyall, already primitive, is 3e-9.
    """
    mol = build(name, element)
    solution = one_electron_solution(mol, uncontract)
    helper = x2c.SpinOrbitalX2CHelper(mol)
    helper.xuncontract = uncontract
    reference = np.asarray(helper.get_hcore())
    ours = x2c_one_electron(solution)
    assert ours.shape == reference.shape
    scale = float(np.max(np.abs(reference)))
    assert np.max(np.abs(ours - reference)) < 1e-7 * scale


# --- The conditioning of the decoupling, which check (i) cannot see ---------------------------

#: ``max |X|`` for a healthy atom. ``X`` relates small to large components, so its elements are
#: of order ``p / 2c`` — a few for a heavy core, 1.0 for neon, 7.7 for titanium, and 2.9-10 for
#: the lanthanides once the metric is projected consistently. Three orders above that is not a
#: heavy atom, it is noise in the null space of a singular basis.
MAX_DECOUPLING_SCALE = 1.0e2


@pytest.mark.parametrize("element", ["Ne", "Ar", "Ti", "Ce", "Dy", "Yb", "Bi"])
def test_the_decoupling_is_well_conditioned_in_a_decontracted_basis(element):
    """⚠ **The check that the per-family sweep was missing, and it is not optional.**

    A decontracted Karlsruhe *lanthanide* set is numerically singular — Ce's four-component
    metric has a smallest eigenvalue of **6.1e-14** over 1004 directions, four to six orders
    below titanium's 5.0e-08. Until the decoupling shared its linear-dependence threshold with
    the four-component solve, those directions survived and filled ``X`` with noise:

    ==== ============= ============= ==================
    atom max abs X      TR residual   one-electron check
    ==== ============= ============= ==================
    Ti       7.7          1.7e-09       5.2e-11
    Ce   **5.5e+03**  **1.0e-03**       9.2e-08
    ==== ============= ============= ==================

    The two-electron correction built from that came out **96% time-reversal odd** — and the
    third column is why this test exists: reproducing PySCF's one-electron X2C, the substantive
    per-family check of ``test_the_decoupling_reproduces_pyscf_x2c_in_every_family``, reads
    9e-08 for Ce at **every** threshold including the broken one. ``h`` is dominated by
    well-conditioned directions; the two-electron mean field is not. A check that cannot fail
    on a broken input is not a check, and this is the one that can.

    Costs no SCF — it is a property of the one-electron problem.
    """
    from kuiva.amf.decouple import picture_change, x2c_decoupling
    from kuiva.spinor.expand import time_reversal_residual

    solution = one_electron_solution(build("x2c-SVPall-2c", element), uncontract=True)
    x, r = x2c_decoupling(solution)
    assert float(np.max(np.abs(x))) < MAX_DECOUPLING_SCALE

    # ...and the picture change of a Hermitian, time-reversal-even operator stays even. Same
    # quantity the correction is refused on, measured where it costs no SCF.
    residual, _ = time_reversal_residual(picture_change(solution.hcore, x, r))
    assert residual < 1e-4


def test_the_density_back_transformation_survives_a_singular_renormalization():
    """⚠ **`R` is exactly singular whenever the basis is linearly dependent, and an LU solve
    against it divides by zero — quietly.**

    :func:`kuiva.amf.decouple.renormalization` builds ``R`` from the overlap eigenvectors it
    keeps, so ``R`` is identically zero on every direction the metric projection dropped.
    Measured on neutral Ce in a decontracted Karlsruhe basis: rank **496 of 502**, condition
    **8.7e+17**. The two-component density then came back at ``max abs D~ = 2.4e+16`` against a
    four-component ``D_LL`` of 4.5, the correction was 1.1e-02 time-reversal odd, and
    ``max abs dG`` was 1.6e+04 Eh where the physical value is 8.5.

    This test builds the pathology directly — a deliberately rank-deficient ``R`` — so it costs
    no lanthanide SCF, and asserts both halves of the fix: the pseudo-inverse recovers the
    density on the range of ``R``, **and** it agrees with a plain inverse to machine precision
    when ``R`` is full rank, which is why Ne and Ti(3+) were unaffected.
    """
    import scipy.linalg

    from kuiva.amf.decouple import two_component_density

    rng = np.random.default_rng(3)
    n = 12
    d = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    d = d + d.conj().T

    # (a) full rank: the pseudo-inverse must reproduce the plain solve exactly.
    full = np.eye(n) + 0.05 * rng.normal(size=(n, n))
    lu = scipy.linalg.lu_factor(full)
    expected = scipy.linalg.lu_solve(
        lu, scipy.linalg.lu_solve(lu, d).conj().T).conj().T
    assert np.allclose(two_component_density(d, full), expected, rtol=0, atol=1e-10)

    # (b) rank deficient: exactly what renormalization returns for a dependent basis.
    u, s, vh = np.linalg.svd(full)
    s[-3:] = 0.0
    singular = (u * s) @ vh
    assert np.linalg.matrix_rank(singular) == n - 3
    got = two_component_density(d, singular)
    assert np.all(np.isfinite(got))
    # bounded by the density it came from, rather than 1/eps times it
    assert np.max(np.abs(got)) < 1e3 * np.max(np.abs(d))


def test_a_singular_metric_is_refused_rather_than_projected_and_reported():
    """The refusal itself, driven by a deliberately broken decoupling.

    ⚠ This path once *warned* and returned the correction anyway, which is how a
    35-minute lanthanide run produced a 96%-odd result that would have been recorded as
    physics. Projecting the odd part out does not rescue it: the even half is contaminated by
    the same amount and is the part that gets used.
    """
    import dataclasses

    from kuiva.amf.backend import FourComponentBlocks, get_backend
    from kuiva.amf.decouple import TIME_REVERSAL_LIMIT, amf_atomic_correction

    solution = one_electron_solution(build("x2c-SVPall-2c", "Ne"), uncontract=False)
    n = 2 * solution.nao
    rng = np.random.default_rng(0)
    odd = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    odd = odd + odd.conj().T                        # Hermitian, but not time-reversal even
    blocks = FourComponentBlocks(ll=odd, ls=np.zeros_like(odd), sl=np.zeros_like(odd),
                                 ss=np.zeros_like(odd))
    broken = dataclasses.replace(solution, veff=blocks)
    impl = get_backend("pyscf")
    with pytest.raises(RuntimeError, match="time-reversal odd"):
        amf_atomic_correction(broken, lambda dm: impl.coulomb_mean_field(broken, dm))
    assert TIME_REVERSAL_LIMIT == 1e-3


# --- Structures that cannot be handled are refused, not approximated -------------------------

def test_a_cartesian_basis_is_refused_with_a_reason():
    """⚠ Structural, and not a limitation more code could lift.

    PySCF's four-component solver works in the j-adapted 2-spinor basis, which is built on
    *spherical* harmonics: ``Mole.nao_2c()`` of a Cartesian molecule is twice its **spherical**
    ``nao``, not its Cartesian one. The correction would be computed over a different set of
    functions from the Hamiltonian it has to be added to.

    The refusal is explicit rather than left to the shape check further down, because for
    ``l <= 1`` the two bases coincide — so a molecule of s and p functions only would pass the
    shape check and be silently wrong, and one with d functions would fail with a message
    about basis-function counts that names nothing.
    """
    mol = gto.M(atom="Ne 0 0 0", basis="cc-pvdz", cart=True, verbose=0)
    assert mol.nao != gto.M(atom="Ne 0 0 0", basis="cc-pvdz", verbose=0).nao
    with pytest.raises(NotImplementedError, match="spherical"):
        amf_correction(mol, method="x2camf")
    # ...and the zero path stays usable, which is what keeps the seam safe.
    assert amf_correction(mol, method="none").is_zero


def test_an_ecp_basis_is_still_refused():
    """Unchanged and asserted here too, because the basis handling was rewritten
    handling around it: X2C has no meaning with a pseudopotential core."""
    mol = gto.M(atom="I 0 0 0", basis="def2-svp", ecp="def2-svp", spin=1, verbose=0)
    with pytest.raises(NotImplementedError, match="all-electron"):
        amf_correction(mol, method="x2camf")


def test_a_decontracted_mole_is_refused_rather_than_silently_mis_basised():
    """⚠ ``Mole.decontract_basis()`` returns a molecule whose ``_bas`` is primitive but whose
    ``_basis`` **still holds the contracted definition it started from**.

    Measured on cc-pVDZ neon: ``xmol.nao`` is 26 while ``xmol._basis["Ne"]`` rebuilds to 14.
    Anything reading ``_basis`` there — which is how the correction learns what functions an
    element carries — would compute the atomic mean field over a different set of functions
    from the Hamiltonian it corrects. It is silent, it is easy to hit while measuring exactly
    the contraction questions these tests are about, and it is now an error.
    """
    from kuiva.amf.atomic import elements_and_bases

    mol = gto.M(atom="Ne 0 0 0", basis="cc-pvdz", verbose=0)
    xmol = mol.decontract_basis(aggregate=True)[0]
    assert xmol.nao == 26 and gto.M(atom="Ne 0 0 0", basis={"Ne": xmol._basis["Ne"]},
                                    verbose=0).nao == 14
    with pytest.raises(ValueError, match="disagree"):
        elements_and_bases(xmol)
    # The properly built primitive molecule — what a caller actually wants — is fine.
    primitive = gto.M(atom="Ne 0 0 0",
                      basis={"Ne": gto.uncontract(mol._basis["Ne"])}, verbose=0)
    assert primitive.nao == 26
    assert len(elements_and_bases(primitive)) == 1


# --- The physics, per family: which comparison is meaningful ---------------------------------

#: Row 3 of the table in the module docstring, measured: a like-for-like comparison of the
#: corrected two-component splitting against four-component theory, **both in the primitive
#: basis**. Observed -0.010% to +0.003% across all four contraction families, so the band is
#: ~50x the worst case and is not tuned onto it. The physically meaningful figure for a
#: picture-change treatment is a fraction of a percent, not the 15% cross-code band.
LIKE_FOR_LIKE_TOL = 5e-4

#: Row 2: decoupling in a **contracted** basis instead, with the reference contracted to match.
#: This is what the "decouple in the primitive basis" advice is worth in numbers,
#: and it is the cost of ``uncontract=False``. Observed -0.174% (Ne, x2c-SVPall-2c), -0.121%
#: (Ne, ANO-RCC), -0.275% (Ne, x2c-TZVPPall-2c), and -0.5% to -1.7% for Ca — it grows with Z
#: and with how hard the set is contracted, so this is a ceiling for the light cases here and
#: not a universal figure.
CONTRACTED_DECOUPLING_TOL = 2e-2


def splitting_from(mol, sl, correction=None):
    """Valence j-splitting [cm^-1] from a **self-consistent two-component** SCF.

    Imported from ``test_amf_correction`` rather than rewritten: which construction produced a
    splitting is part of what the number means (the frozen-orbital
    construction disagrees by 30% on the same operator), so there is exactly one of it.
    """
    from test_amf_correction import self_consistent_spectrum

    e = self_consistent_spectrum(mol, correction)
    return float((e[sl][-1] - e[sl][0]) * HARTREE_CM)


def four_component_splitting(element, basis, sl, uncontract):
    e = atomic_solution(element, basis, uncontract=uncontract).occupied_energies()
    return float((e[sl][-1] - e[sl][0]) * HARTREE_CM)


#: ``(family, element, slice of the ascending occupied spectrum, shell label)``. The slice is
#: stated rather than taken as "the last six", because for calcium the frontier is 4s and the
#: shell under test is the 3p below it.
PHYSICS = [
    ("x2c-SVPall-2c", "Ne", slice(4, 10), "2p"),
    ("dyallv2z", "Ne", slice(4, 10), "2p"),
    ("x2c-TZVPPall-2c", "Ne", slice(4, 10), "2p"),
    pytest.param("ANO-RCC", "Ne", slice(4, 10), "2p", marks=pytest.mark.slow),
    pytest.param("cc-pVDZ-X2C", "Ca", slice(12, 18), "3p", marks=pytest.mark.slow),
    pytest.param("x2c-SVPall-2c", "Ca", slice(12, 18), "3p", marks=pytest.mark.slow),
]


@pytest.mark.parametrize("name,element,sl,label", PHYSICS)
def test_the_correction_is_family_independent_like_for_like(name, element, sl, label):
    """**The basis-generality criterion, stated the only way it can be true.**

    With the molecule *and* the four-component reference both in the primitive basis, the
    corrected splitting reproduces four-component theory to **-0.010% to +0.003%** — across
    segmented, general, uncontracted and mixed contractions, and across two elements whose
    splittings differ by a factor of three. The correction is therefore family-independent;
    what differs between families is the basis-set description of the atom, which is common to
    both sides here and cancels.
    """
    clear_cache()
    mol = build(name, element)
    parsed = mol._basis[mol.atom_symbol(0)]
    primitive = gto.M(atom=[(element, (0.0, 0.0, 0.0))],
                      basis={element: gto.uncontract(parsed)}, verbose=0)
    reference = four_component_splitting(element, parsed, sl, True)
    corrected = splitting_from(primitive, sl,
                               amf_correction(primitive, method="x2camf"))
    assert abs(corrected - reference) / reference < LIKE_FOR_LIKE_TOL


@pytest.mark.parametrize("name,element,sl,label", PHYSICS)
def test_decoupling_in_a_contracted_basis_costs_a_stated_amount(name, element, sl, label):
    """Row 2: what ``uncontract=False`` buys and what it costs.

    The decoupling belongs in the primitive basis; this is that advice in
    numbers, with the reference contracted to match so the comparison stays like-for-like. It
    is asserted as a **band**, not a target: the point is that the penalty is small and bounded
    (a few tenths of a percent at Ne, under 2% at Ca) rather than that it takes any particular
    value, and that it is an order of magnitude larger than the primitive-basis residual above.
    """
    clear_cache()
    mol = build(name, element)
    parsed = mol._basis[mol.atom_symbol(0)]
    reference = four_component_splitting(element, parsed, sl, False)
    corrected = splitting_from(mol, sl, amf_correction(mol, method="x2camf",
                                                       uncontract=False))
    assert abs(corrected - reference) / reference < CONTRACTED_DECOUPLING_TOL


def test_comparing_across_a_basis_change_is_the_trap_and_is_not_the_correction():
    """⚠ Row 1: the comparison that looks obvious and means almost nothing.

    A two-component calculation in the **contracted** molecular basis against a four-component
    reference in the **decontracted** basis the solver used is a comparison between two
    different spaces. It gives +0.272% for neon in ``x2c-SVPall-2c`` — a number once
    originally recorded as the accuracy of the correction, and which turned out to be a
    basis mismatch — and **-16.2%** for calcium in ``cc-pVDZ-X2C``, where the contracted
    valence description is far more truncated.

    This is asserted so the failure mode is pinned rather than merely warned about: the
    like-for-like residual is smaller by two orders of magnitude on the very same atom and
    code path. If someone "fixes" the correction until row 1 is small, they will have broken it.
    """
    clear_cache()
    mol = build("cc-pVDZ-X2C", "Ca")
    parsed = mol._basis["Ca"]
    sl = slice(12, 18)
    corrected = splitting_from(mol, sl, amf_correction(mol, method="x2camf"))

    mismatched = four_component_splitting("Ca", parsed, sl, True)     # primitive reference
    matched = four_component_splitting("Ca", parsed, sl, False)       # contracted reference
    assert abs(corrected - mismatched) / mismatched > 0.10            # the trap: -16%
    assert abs(corrected - matched) / matched < 0.05                  # like-for-like: -1.3%
    # and the reference itself moves by ~17% between the two bases, which is the whole effect
    assert abs(mismatched - matched) / matched > 0.10


def test_the_raw_splitting_varies_between_families_and_that_is_basis_set_error():
    """The cross-family spread, reported rather than hidden inside a tolerance.

    Neon's 2p splitting is **903 cm^-1** in ``x2c-SVPall-2c`` and **984** in ``dyallv2z`` — 9%
    apart, in four-component theory itself, before any picture change is discussed. No
    correction can or should recover that, and a test that demanded the two agree would be
    demanding the wrong thing. What must agree is each family's residual against *its own*
    four-component reference, which is the test above.
    """
    clear_cache()
    small = build("x2c-SVPall-2c", "Ne")
    large = build("dyallv2z", "Ne")
    a = four_component_splitting("Ne", small._basis["Ne"], slice(4, 10), True)
    b = four_component_splitting("Ne", large._basis["Ne"], slice(4, 10), True)
    assert abs(b - a) / a > 0.05          # a real basis-set difference, not noise
    assert 850.0 < a < 1050.0 and 850.0 < b < 1050.0
