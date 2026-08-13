"""The scalar average-of-configuration SCF (``run_scalar_aoc``).

What can actually go wrong here is not that the SCF fails — it converges cleanly either way —
but that it converges to **a different functional than the one it claims**. The occupations
are imposed, so they always look right; the energy is a plausible number whatever coupling
coefficient went into it; and the orbitals of a "shell" are a shell only if the angular-
momentum assignment picked the orbitals the configuration meant. The tests are ordered by
what they can catch.

1. **Two limits where the answer is known independently.** A closed shell must reproduce
   ordinary RHF (the averaging engine is inert there), and a one-electron open shell must
   reproduce ROHF — that is the ``alpha = 0`` end of the coupling, where a true average gives
   the single electron no self-repulsion and a fractional density still charges it ``1/n^2``
   of its own.

2. ⚠ **The scalar result against the two-component one, on the same one-electron Hamiltonian
   and the same configuration.** This is the decisive check of the whole change: the
   configuration-average machinery was validated in the GHF convention (``G = J - K`` on a
   spin-orbital density), and the front-end runs it in the RHF convention (``J - K/2`` on a
   spin-restricted total density) with occupations that go up to 2 instead of 1. Every part of
   that could be off by a factor of two and still converge to a clean-looking number. The two
   sides share the filling rule and the energy functional but not the representation, so
   agreement to SCF tolerance is a statement about the conventions and nothing else.

3. **The converged density is spherical**, which under average of configuration is the
   assertion that the averaging *worked* — with an aufbau ROHF density on the same atom as
   the negative control, because a check that nothing can fail proves nothing.

4. **What is refused**, and it is refused before the SCF: a configuration that is not a bound
   state of the element, and one the basis has no functions for.

Cost: light atoms in a split-valence basis, all sub-second. Spin-orbit ingestion is off in
every test but one (a carbon four-component solve, computed fresh, no warm cache needed).
"""
import numpy as np
import pytest
from pyscf import gto, scf

from kuiva.amf.configuration import AtomicConfiguration
from kuiva.amf.pyscf_dhf import SPHERICAL_DENSITY_TOLERANCE, density_anisotropy
from kuiva.extras.shells import ShellConfiguration
from kuiva.interface.pyscf_bridge import run_scalar_aoc, run_scalar_x2c

from test_amf_open_shell import average_of_configuration_ghf

BASIS = "x2c-SVPall-2c"
MEMORY_GB = 8.0

#: SCF agreement. Both sides are converged to 1e-10 on the energy, so anything above this is a
#: different functional rather than a different path to the same one.
SCF_TOL = 1e-9


class _Atom:
    """A ``Molecule``-shaped object for the ordinary bridge entry point."""
    unit = "Bohr"

    def __init__(self, symbol, charge=0, spin=0, basis=BASIS):
        self.atoms = [(symbol, (0.0, 0.0, 0.0))]
        self.charge = charge
        self.spin = spin
        self.basis = basis


def aoc(element, configuration, **kwargs):
    kwargs.setdefault("with_soc", False)          # SOC changes no scalar quantity, and costs
    kwargs.setdefault("memory_gb", MEMORY_GB)     # a four-component solve per element
    return run_scalar_aoc(element, configuration, basis=BASIS, **kwargs)


def mole(symbol, charge=0):
    z = int(gto.charge(symbol))
    return gto.M(atom=[(symbol, (0.0, 0.0, 0.0))], basis=BASIS, charge=charge,
                 spin=(z - charge) % 2, verbose=0)


# --- 1. The two limits with an independent answer -------------------------------------------

def test_a_closed_shell_average_is_ordinary_rhf():
    """With no open shell there is nothing to average: the coupling engine is not installed
    at all and the occupations are the integers RHF would use. The two SCFs still take
    *different paths* — this one fills each ``l`` channel separately, RHF fills the lowest
    orbitals — so they are compared at the fixed point, not step by step."""
    data = aoc("Ne", "[Ne]")
    plain = run_scalar_x2c(_Atom("Ne"), with_soc=False, memory_gb=MEMORY_GB)

    assert data.converged and plain.converged
    assert data.e_scf == pytest.approx(plain.e_scf, abs=SCF_TOL)
    assert set(np.unique(data.mo_occ)) == {0.0, 2.0}
    assert data.reference == "aoc"
    assert data.nelec == (5, 5) and data.nelec_total == 10


def test_a_one_electron_shell_is_rohf():
    """⚠ ``alpha = n(q-1)/(q(n-1))`` is **zero** at ``q = 1``, where a true configuration
    average gives the open shell no two-electron energy of its own — and that is exactly what
    ROHF does with a single electron in a nondegenerate orbital. A fractional-occupation SCF
    without the coupling coefficient would instead charge the electron ``1/n^2`` of its own
    repulsion and land above this by a visible margin.
    """
    data = aoc("Li", "1s2 2s1")
    ro = scf.ROHF(mole("Li")).sfx2c1e()
    ro.conv_tol, ro.verbose = 1e-10, 0
    ro.kernel()

    assert data.converged and ro.converged
    assert data.e_scf == pytest.approx(float(ro.e_tot), abs=SCF_TOL)
    assert data.mo_occ[1] == pytest.approx(1.0)      # 2s: one electron in one orbital


# --- 2. The decisive convention check -------------------------------------------------------

@pytest.mark.parametrize("symbol,configuration", [
    ("C", "[He]2s2 2p2"),                            # alpha = 0.6
    ("O", "[He]2s2 2p4"),                            # alpha = 0.9
    ("Ti", "[Ar]3d2 4s2"),                           # two channels, one of them open
])
def test_the_scalar_average_reproduces_the_two_component_one(symbol, configuration):
    """⚠ **The check the whole scalar path rests on.**

    The same configuration, the same spin-free one-electron Hamiltonian and the same
    average-of-configuration functional, solved once in the spin-blocked two-component basis
    (occupations in ``[0, 1]``, ``G = J - K``) and once in the spin-restricted scalar one
    (occupations in ``[0, 2]``, ``G = J - K/2``). The energy functional is representation
    independent, so the two must agree to SCF tolerance.

    Three things could be off by a factor of two between them and none of them would announce
    itself: the frontier occupation, the open-shell density entering ``Tr[D_s G[D_s]]``, and
    the occupation used in the effective-Fock block rule. The two-component side is the one
    validated against four-component DHF, which is what makes it the reference here.

    ⚠ **Only the scalar side is constrained to spherical solutions**, and the comparison
    survives that because every configuration here has a single open shell, which converges
    spherically on its own. Adding a two-open-shell case would make the two sides converge to
    *different states* — the constrained one to the spherical solution, the unconstrained
    helper to a broken one below it — and the disagreement would look like a convention error
    in the scalar path rather than what it is.
    """
    import scipy.linalg

    config = AtomicConfiguration.parse(configuration)
    mol = mole(symbol)
    h_scalar = np.asarray(scf.RHF(mol).sfx2c1e().get_hcore())
    ghf = average_of_configuration_ghf(mol, config,
                                       scipy.linalg.block_diag(h_scalar, h_scalar))
    data = aoc(symbol, configuration)

    assert data.converged and ghf.converged
    assert data.e_scf == pytest.approx(float(ghf.e_tot), abs=SCF_TOL)


# --- 3. Sphericity, with its negative control -----------------------------------------------

def test_the_converged_density_is_spherical_and_an_aufbau_one_is_not():
    """An atom is spherically symmetric, so this is not a closed-shell restriction but the
    assertion that the averaging did what it claims. The ROHF comparison is the control: it
    occupies ``2p_x^2 2p_y^1 2p_z^1``, converges perfectly happily, and is anisotropic at
    order one — a wrong answer with nothing wrong-looking about it.
    """
    data = aoc("O", "[He]2s2 2p4")
    mol = mole("O")
    dm = (data.mo_coeff * data.mo_occ) @ data.mo_coeff.T
    assert density_anisotropy(mol, dm) < SPHERICAL_DENSITY_TOLERANCE

    ro = scf.ROHF(mol).sfx2c1e()
    ro.conv_tol, ro.verbose = 1e-10, 0
    ro.kernel()
    assert ro.converged
    spin_traced = np.asarray(ro.make_rdm1()).sum(axis=0)      # ROHF hands back (alpha, beta)
    assert density_anisotropy(mol, spin_traced) > 0.1


def test_the_spherical_constraint_is_what_keeps_a_two_open_shell_atom_spherical():
    """⚠ **The observable half of a fixed defect, with the control that shows it was one.**

    Fractional occupation of a whole ``l`` shell makes the density spherical *given* spherical
    orbitals. It does not make the spherical solution **stable**: a fractionally occupied
    Hartree-Fock functional has symmetry-broken solutions below it, so the iteration slides
    into one, and the anisotropy grows from roundoff by about an order of magnitude per cycle.
    Ti(+1) ``[Ar] 3d2 4s1`` — two open shells, the shape every Ln(I) reference has — is where
    it first showed: measured **4.1e-6** without the constraint against **2.4e-13** with it,
    on the same converged energy to 1e-10 Eh.

    The projection of the Fock onto its rank-zero part is therefore not a convergence aid but
    a statement of which state is being solved, and it is on by default. Turning it off warns,
    and this is the test that the warning is not decoration.
    """
    with_it = aoc("Ti", "[Ar] 3d2 4s1", conv_tol=1e-11)
    without = aoc("Ti", "[Ar] 3d2 4s1", conv_tol=1e-11, spherical=False)
    mol = mole("Ti", charge=1)

    def anisotropy(data):
        return density_anisotropy(mol, (data.mo_coeff * data.mo_occ) @ data.mo_coeff.T)

    assert with_it.converged and without.converged
    assert anisotropy(with_it) < 1e-9
    assert anisotropy(without) > SPHERICAL_DENSITY_TOLERANCE
    assert anisotropy(without) > 1e6 * anisotropy(with_it)
    # ⚠ The same state, not a different one: the constraint changes the trajectory, and at a
    # spherical fixed point it is the identity. A change here would mean stored numbers move.
    assert with_it.e_scf == pytest.approx(without.e_scf, abs=1e-9)


def test_the_spherical_constraint_does_not_move_a_solution_that_was_already_clean():
    """A single open shell converges spherically without help, so the projection must be
    inert there — bitwise is too strong (it changes the trajectory), but the converged energy
    is the same state to well inside SCF convergence."""
    for element, configuration in (("Ne", "[He]2s2 2p6"), ("O", "[He]2s2 2p4")):
        with_it = aoc(element, configuration, conv_tol=1e-11)
        without = aoc(element, configuration, conv_tol=1e-11, spherical=False)
        assert with_it.e_scf == pytest.approx(without.e_scf, abs=1e-10)


def test_turning_the_constraint_off_says_so(kuiva_caplog):
    aoc("O", "[He]2s2 2p4", spherical=False)
    assert any("WITHOUT the spherical constraint" in record.getMessage()
               for record in kuiva_caplog.records)


def test_the_occupations_are_the_configuration_and_nothing_else():
    """The frontier shell carries ``q / (2l+1)`` on **every** one of its orbitals, and the
    shells below it are full. Asserting the value is what separates an average over the whole
    ``2p`` shell from any other spherical-looking arrangement of four electrons."""
    data = aoc("O", "[He]2s2 2p4")
    occ = np.sort(data.mo_occ)[::-1]

    assert float(occ.sum()) == pytest.approx(8.0, abs=1e-10)
    assert occ[0] == pytest.approx(2.0) and occ[1] == pytest.approx(2.0)   # 1s, 2s
    assert np.allclose(occ[2:5], 4.0 / 3.0)                                # 2p^4 over three
    assert np.allclose(occ[5:], 0.0)


def test_an_ion_takes_its_charge_from_the_configuration():
    """The configuration is the single source of truth for how many electrons are solved for,
    as it is for the atomic mean field: Ti(3+) is named by its configuration and the charge
    follows, so the two can never disagree."""
    data = aoc("Ti", "[Ar]3d1")
    assert data.nelec_total == 19
    assert data.nelec == (10, 9)                   # formal split; see ScalarX2CData
    assert data.converged
    # one d electron shared over five d orbitals, and nothing else fractional
    fractional = np.sort(data.mo_occ[(data.mo_occ > 1e-12) & (data.mo_occ < 2.0 - 1e-12)])
    assert fractional.size == 5 and np.allclose(fractional, 0.2)


def test_a_shell_resolved_configuration_is_accepted_without_an_import():
    """⚠ The seam the Slater-Condon feature uses: anything with ``to_atomic()`` is accepted,
    duck-typed. The bridge may not import the package that class lives in — the dependency
    runs the other way — and it does not need to, because the per-``l`` form is all an SCF is
    defined by."""
    shell = ShellConfiguration.parse("[He] 2s2 2p4")
    assert aoc("O", shell).e_scf == pytest.approx(aoc("O", "[He]2s2 2p4").e_scf, abs=1e-12)


# --- 4. Refusals, before the SCF ------------------------------------------------------------

def test_a_configuration_that_is_not_a_bound_state_is_refused():
    with pytest.raises(ValueError, match="not a bound state"):
        aoc("O", "[Ne]")                             # ten electrons on Z = 8


def test_a_basis_without_the_functions_the_configuration_needs_is_refused():
    """The refusal names the missing channel, and it happens before any SCF cycle: a
    split-valence basis for neon has no ``f`` functions to put an f electron in."""
    with pytest.raises(ValueError, match="needs f functions"):
        aoc("Ne", "1s2 2s2 2p5 4f1")


# --- 5. The mean field is taken over the same state -----------------------------------------

def test_the_atomic_mean_field_defaults_to_this_configuration():
    """⚠ A parameter extracted from an average-of-configuration reference and screened by the
    mean field of a *different* ion would be a mixture of two states, with nothing in the
    provenance to say so. The screening therefore defaults to the configuration being solved
    rather than to the element's default reference — which for carbon differs (the default is
    the neutral ground configuration; here it is whatever was asked for).

    Carbon's four-component solve is sub-second and is computed, not replayed.
    """
    data = aoc("C", "[He]2s2 2p2", with_soc=True)
    record = data.soc.provenance()["screening"]
    assert record["method"] == "x2camf"
    assert record["configurations"] == {"C": "s4 p2"}

    explicit = run_scalar_aoc("C", "[He]2s2 2p2", basis=BASIS, memory_gb=MEMORY_GB,
                              screening_options={"configuration": "[He]2s2 2p1"})
    assert explicit.soc.provenance()["screening"]["configurations"] == {"C": "s4 p1"}
