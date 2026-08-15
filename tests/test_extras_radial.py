"""Shell identification and radial extraction from a converged atomic solution.

An SCF hands back each degenerate shell in an **arbitrary rotation** of its ``2l+1`` orbitals.
Nothing in the coefficients says which rotation, so a radial function read off an MO is a
radial function times an unknown mixing of ``m`` values — and every angular coefficient
applied to it afterwards is then wrong by that mixing, silently and plausibly. The extraction
sidesteps it by reading the radial function off the **Fock operator**, whose blocks are
``(l, m)``-diagonal and m-independent by symmetry, and *building* the orbitals from it.

So the tests are about the properties that construction is supposed to have, and about the
diagnostics that say when the solution underneath it does not deserve them:

1. **The orbitals are pure ``m`` and orthonormal**, and every one of a shell shares one radial
   function — asserted on the coefficients, since that is exactly what the alignment claim is.
2. **The extraction agrees with the SCF** on the shells' energies and on how many electrons
   each holds, computed the other way round (a projection of the converged density, which
   does not care how the eigensolver oriented a degenerate manifold).
3. ⚠ **The diagnostics fail on a solution that deserves them to.** A symmetry-broken ROHF
   oxygen is the control: it converges cleanly, its shells look ordinary, and it is not
   spherical. If the anisotropy did not separate it from an average of configuration by orders
   of magnitude, none of the numbers above would mean anything.

Cost: two light atoms in a split-valence basis, no spin-orbit ingestion, ~2 s.
"""
from dataclasses import replace

import numpy as np
import pytest

from kuiva.extras.shells import (SHELL_ANISOTROPY_TOLERANCE, ShellConfiguration,
                                 _channel_blocks, extract_shells, parse_shell_label,
                                 radial_phase)
from kuiva.interface.pyscf_bridge import run_scalar_aoc, run_scalar_x2c

BASIS = "x2c-SVPall-2c"
MEMORY_GB = 8.0


class _Atom:
    unit = "Bohr"

    def __init__(self, symbol, spin=0, charge=0, basis=BASIS):
        self.atoms = [(symbol, (0.0, 0.0, 0.0))]
        self.charge = charge
        self.spin = spin
        self.basis = basis


@pytest.fixture(scope="module")
def oxygen():
    """O 2p^4, the cheapest system with a partly filled degenerate shell."""
    config = ShellConfiguration.parse("[He] 2s2 2p4")
    data = run_scalar_aoc("O", config, basis=BASIS, with_soc=False, memory_gb=MEMORY_GB)
    assert data.converged
    return data, config


@pytest.fixture(scope="module")
def titanium():
    """Ti(3+) 3d^1: a d shell, and a configuration with an empty valence shell above it."""
    config = ShellConfiguration.parse("[Ar] 3d1")
    data = run_scalar_aoc("Ti", config, basis=BASIS, with_soc=False, memory_gb=MEMORY_GB)
    assert data.converged
    return data, config


# --- 1. What the construction claims --------------------------------------------------------

def test_the_orbitals_of_a_shell_are_one_radial_function_in_pure_m(oxygen):
    """⚠ The alignment claim, asserted on the coefficients rather than on a consequence.

    Every column of a shell must be the *same* radial function placed in a different ``m``
    channel: nonzero only on AOs of that ``(l, m)``, and carrying identical coefficients over
    the radial functions. An MO of the same shell satisfies neither — it is a mixture over
    ``m`` with no way to tell from the numbers.
    """
    data, config = oxygen
    layout = data.ao_layout
    shell = extract_shells(data, config, shells=("2p",))["2p"]

    assert shell.size == 3 and shell.m_values == (-1, 0, 1)
    assert shell.coefficients.shape == (data.nao, 3)
    for column, m in enumerate(shell.m_values):
        channel = (layout.ao_l == 1) & (layout.ao_m == m)
        coefficients = shell.coefficients[:, column]
        assert np.allclose(coefficients[~channel], 0.0)          # pure m
        order = np.argsort(layout.ao_shell[channel])
        assert np.allclose(coefficients[channel][order], shell.radial)   # one radial function


def test_the_shell_orbitals_are_orthonormal(oxygen):
    """They are built, not solved for, so orthonormality is a statement about the m-averaged
    generalized eigenproblem being the right one — and it is what every integral transform
    over these orbitals silently assumes."""
    data, config = oxygen
    shells = extract_shells(data, config, shells=("1s", "2s", "2p"))
    columns = np.hstack([shell.coefficients for shell in shells])
    overlap = columns.T @ data.s_ao @ columns
    assert np.allclose(overlap, np.eye(columns.shape[1]), atol=1e-10)


def test_the_shells_span_the_same_space_the_scf_orbitals_do(oxygen):
    """The construction may not *move* anything: a shell's ``2l+1`` orbitals must span exactly
    the manifold the SCF's own degenerate orbitals span, only in a different basis of it. The
    projector is basis independent, so comparing projectors is the invariant form of that."""
    data, config = oxygen
    shell = extract_shells(data, config, shells=("2p",))["2p"]
    sc = data.s_ao @ data.mo_coeff
    weights = np.sum((shell.coefficients.T @ sc) ** 2, axis=0)
    members = np.where(weights > 0.5)[0]

    assert members.size == 3
    assert np.allclose(weights[members], 1.0, atol=1e-9)          # whole MOs, nothing partial
    assert np.isclose(weights.sum(), 3.0, atol=1e-9)              # ...and no leakage elsewhere
    mine = shell.coefficients @ shell.coefficients.T
    theirs = data.mo_coeff[:, members] @ data.mo_coeff[:, members].T
    assert np.allclose(mine, theirs, atol=1e-9)                   # the same projector


# --- 2. Agreement with the solution it came from --------------------------------------------

def test_the_shell_energies_and_occupations_are_the_scf_s_own(oxygen):
    """Two routes to the same numbers: the m-averaged Fock block, and the SCF's spectrum with
    its occupations. The shell energies are orbital energies and the occupations are the
    configuration's, projected out of the converged density."""
    data, config = oxygen
    shells = extract_shells(data, config)

    assert shells.labels == ("1s", "2s", "2p")
    assert [s.occupation for s in shells] == pytest.approx([2.0, 2.0, 4.0], abs=1e-9)
    assert [s.degenerate_block for s in shells] == [1, 1, 3]
    assert max(s.energy_spread for s in shells) < 1e-10
    assert shells.anisotropy < SHELL_ANISOTROPY_TOLERANCE
    # the orbital energies rise with n and the 2p sits highest, as an occupied spectrum must
    assert shells["1s"].energy < shells["2s"].energy < shells["2p"].energy < 0.0


def test_an_empty_shell_can_be_extracted_and_says_so(titanium):
    """A shell the configuration does not occupy is still a well-defined radial function of
    the converged Fock, and the Slater-Condon use needs exactly that — ``F^k(4f, 5d)`` for a
    5d that carries no electrons is a legitimate question. What it must not do is pretend to
    an occupation: the projection returns zero and the configuration agrees."""
    data, config = titanium
    shells = extract_shells(data, config, shells=("3d", "4s"))

    assert shells["3d"].occupation == pytest.approx(1.0, abs=1e-9)
    assert shells["4s"].occupation == pytest.approx(0.0, abs=1e-9)
    assert shells["3d"].degenerate_block == 5
    assert shells.anisotropy < SHELL_ANISOTROPY_TOLERANCE


def test_a_shell_is_addressed_by_name(oxygen):
    data, config = oxygen
    shells = extract_shells(data, config, shells=("2s", "2p"))
    assert shells["2p"] is shells[(2, 1)] is shells[1]
    with pytest.raises(KeyError, match="3d was not extracted"):
        shells["3d"]
    assert parse_shell_label("4f") == (4, 3) and parse_shell_label("6s") == (6, 0)
    with pytest.raises(ValueError, match="cannot read"):
        parse_shell_label("4f9")                  # an occupation is not part of a shell name
    with pytest.raises(ValueError, match="no 2d"):
        parse_shell_label("2d")


# --- 3. The diagnostics, against a solution that must fail them -----------------------------

def test_a_symmetry_broken_solution_is_caught(oxygen):
    """⚠ **The control that gives every other number here its meaning.**

    ROHF on oxygen occupies ``2p_x^2 2p_y^1 2p_z^1``: it converges cleanly, reports a healthy
    gap, and is not spherically symmetric. Read as if it had shells it gives a plausible 2p
    radial function that is an average over orbitals which are *not* degenerate — the failure
    this whole construction exists to make visible.

    Every diagnostic fires on it, and by margins that are not delicate: the Fock varies across
    the ``m`` channels by order 0.1 where an average of configuration gives 1e-11, the 2p
    orbitals no longer form a degenerate group of three, and even the occupation — the
    loosest of the three, and deliberately a check on the *configuration* rather than on
    sphericity — lands outside its bound.
    """
    aoc_data, config = oxygen
    broken = run_scalar_x2c(_Atom("O", spin=2), reference="rohf", with_soc=False,
                            memory_gb=MEMORY_GB)
    assert broken.converged

    with pytest.raises(RuntimeError, match="not the one this configuration describes"):
        extract_shells(broken, config, shells=("2p",))

    loose = extract_shells(broken, config, shells=("2p",), occupation_tol=1.0,
                           anisotropy_tol=1e9)
    reference = extract_shells(aoc_data, config, shells=("2p",))
    assert loose.anisotropy > 0.01
    assert loose.anisotropy > 1e6 * reference.anisotropy
    assert loose["2p"].degenerate_block != 3


def test_the_anisotropy_warns_rather_than_refusing(oxygen, kuiva_caplog):
    """Advisory, like every sphericity diagnostic in the project: the run may still be the one
    the user wanted, and the number is reported beside the parameters either way."""
    data, config = oxygen
    shells = extract_shells(data, config, shells=("2p",), anisotropy_tol=1e-30)
    assert shells["2p"].occupation == pytest.approx(4.0, abs=1e-9)
    assert any("m channels" in record.getMessage() for record in kuiva_caplog.records)


def test_what_extraction_refuses_outright(oxygen):
    """The three cases where there is no shell to speak of, refused with the reason rather
    than returning something shaped like an answer."""
    data, config = oxygen

    with pytest.raises(ValueError, match="radial function of the s channel"):
        extract_shells(data, config, shells=("9s",))          # past what the basis offers

    unrestricted = run_scalar_x2c(_Atom("O", spin=2), reference="uhf", with_soc=False,
                                  memory_gb=MEMORY_GB)
    with pytest.raises(ValueError, match="spin-restricted"):
        extract_shells(unrestricted, config, shells=("2p",))

    with pytest.raises(ValueError, match="AO layout"):
        extract_shells(replace(data, ao_layout=None), config, shells=("1s",))

    # "the 4f shell" of a molecule is not a well-defined object, and the refusal says so
    two_atoms = replace(data.ao_layout, atom_symbols=("O", "O"))
    with pytest.raises(ValueError, match="single atom"):
        extract_shells(replace(data, ao_layout=two_atoms), config, shells=("1s",))


# --- 6. The phase of a radial function ------------------------------------------------------

def test_the_radial_functions_come_out_positive_in_their_outer_region(oxygen, titanium):
    """⚠ **The convention that gives a cross parameter's sign a meaning.**

    A radial function is an eigenvector, so its overall sign is whatever LAPACK returned:
    reproducible for one matrix and unrelated between two. ``F^k``, ``G^k`` and ``zeta`` are
    quadratic in every radial function they involve and cannot see it, but a genuine
    ``R^k(ab;cd)`` is linear in two of them and flips with either — so an unfixed phase makes
    the sign of every cross parameter incomparable between two ions, silently.

    :func:`kuiva.extras.shells.radial_phase` fixes it to ``P_nl(r) > 0`` as ``r -> infinity``.
    After extraction the convention must therefore be a *fixed point*: asking for the phase of
    what came out returns ``+1``.
    """
    for data, config in (oxygen, titanium):
        shells = extract_shells(data, config)
        layout = data.ao_layout
        for shell in shells:
            blocks = _channel_blocks(layout, shell.l)
            assert radial_phase(layout, blocks, shell.radial) == pytest.approx(1.0)


def test_the_phase_convention_detects_a_flipped_radial_function(oxygen):
    """The other half: the function must actually *see* a sign, not return +1 regardless."""
    data, config = oxygen
    shells = extract_shells(data, config)
    layout = data.ao_layout
    for shell in shells:
        blocks = _channel_blocks(layout, shell.l)
        assert radial_phase(layout, blocks, -np.asarray(shell.radial)) == pytest.approx(-1.0)


def test_a_parameter_follows_the_phase_of_the_shells_it_names_an_odd_number_of_times(oxygen):
    """⚠ **Which parameters the convention protects, and which never needed it.**

    ``R^k(ab;cd)`` carries the product of the phases of ``a``, ``b``, ``c`` and ``d``, so
    flipping one shell's radial function flips exactly those parameters that name it an **odd**
    number of times. That rule covers the special cases without needing them stated:
    ``F^k(a,b) = R^k(ab;ab)`` and ``G^k(a,b) = R^k(ab;ba)`` each name both shells twice and are
    therefore phase-independent, which is why the defect this convention fixes was invisible in
    everything except the genuine cross parameters.

    This is the mechanism rather than the observable, and it is what makes an unfixed phase a
    defect rather than a cosmetic detail.
    """
    from kuiva.extras.shells import AtomicShells
    from kuiva.extras.slater_condon import extract_parameters

    data, config = oxygen
    shells = extract_shells(data, config)
    parameters = list(extract_parameters(shells, data))
    reference = {p.label: p.value for p in parameters}

    # ⚠ F and G store two labels, not four: F^k(a,b) is R^k(ab;ab) and G^k(a,b) is
    # R^k(ab;ba), so each names both shells twice. Expanding to the four labels the phase
    # actually multiplies is what makes one rule cover all three kinds.
    def four(parameter):
        a, b = parameter.shells[:2]
        return {"F": (a, b, a, b), "G": (a, b, b, a)}.get(parameter.kind, parameter.shells)

    sensitive = 0
    for target in shells.labels:
        flipped = AtomicShells(
            shells=tuple(replace(s, coefficients=-s.coefficients, radial=-s.radial)
                         if s.label == target else s for s in shells),
            anisotropy=shells.anisotropy, configuration=shells.configuration)
        after = {p.label: p.value for p in extract_parameters(flipped, data)}
        assert set(after) == set(reference)
        for parameter in parameters:
            odd = four(parameter).count(target) % 2
            sensitive += odd
            expected = -reference[parameter.label] if odd else reference[parameter.label]
            assert after[parameter.label] == pytest.approx(
                expected, abs=1e-12, rel=1e-10), (parameter.label, target)

    assert sensitive, "no parameter here depends on a phase, so this proves nothing"
    assert sensitive < len(parameters) * len(shells.labels), "and none is independent of one"
