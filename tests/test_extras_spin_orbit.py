"""One-electron spin-orbit constants ``zeta_nl`` from a converged atomic solution.

Symmetry leaves a single number free: inside one shell of a spherical atom the spin-dependent
part of a one-electron operator can only be ``zeta l . s``. So the fit cannot be wrong by a
little — it is either the right number or the whole construction (the ``m`` ordering, the
spin-blocked row layout, the Pauli assembly, the shell orbitals) is wrong, and then it is
wrong by a factor. Three authorities are used, and each fails on something the others cannot:

1. **The hydrogenic closed form** ``zeta_{nl} = Z^4 alpha^2 / [2 n^3 l (l + 1/2) (l + 1)]``,
   which involves no other program at all and pins the magnitude, the sign, the ``l``
   dependence and the ``n`` dependence at once. ⚠ Its accuracy here is limited by the basis
   set, not by the extraction — see the test.
2. **A second route through the two-component machinery**: the shell's spinors built by
   :mod:`kuiva.spinor.expand`, the assembled 2c Hamiltonian transformed by
   :func:`kuiva.integrals.transform.transform_1e` and diagonalized. It must produce the
   ``2l`` + ``2l+2`` level pattern with the separation ``(2l+1) zeta / 2``. Different code,
   different conventions (interleaved Kramers columns rather than spin-blocked ones), same
   number.
3. **The two-electron screening**, which is the largest methodological choice in any
   spin-orbit constant and moves this one by 43%.

Plus the negative controls, and ⚠ **what they establish is a boundary rather than a
capability**: shell orbitals whose ``m`` channels are permuted, rotated or unequally scaled
send the residual to 0.4-1.0, while a whole other shell mixed into one channel leaves it
untouched at 6e-8 — the atomic spin-orbit operator is block diagonal in ``l``, so that
contamination genuinely is invisible here (the radial-parameter residual is what catches it).
Both are asserted, so a future change cannot quietly claim the diagnostic covers more than it
does.

Cost: three light atoms in split-valence and quadruple-zeta bases, one four-component atomic
solve for boron (sub-second), about six seconds.
"""
from dataclasses import replace

import numpy as np
import pytest

from kuiva.extras.shells import AtomicShells, ShellConfiguration, extract_shells
from kuiva.extras.spin_orbit import (ZETA_RESIDUAL_TOLERANCE, extract_spin_orbit,
                                     shell_spin_orbit_block)
from kuiva.interface.pyscf_bridge import run_scalar_aoc, run_scalar_x2c

BASIS = "x2c-SVPall-2c"
MEMORY_GB = 8.0

#: The fine-structure constant squared, CODATA 2018. Only the hydrogenic closed form uses it.
ALPHA_SQUARED = (1.0 / 137.035999084) ** 2


def hydrogenic_zeta(z, n, l):
    """``Z^4 alpha^2 / [2 n^3 l (l + 1/2) (l + 1)]`` [Eh] — the exact one-electron constant.

    The Pauli-limit spin-orbit constant of a hydrogenic ion, i.e. ``alpha^2/2`` times
    ``<r^-3>_{nl} = Z^3 / [n^3 l (l+1/2) (l+1)]`` for a nuclear charge ``Z``. Bethe & Salpeter,
    *Quantum Mechanics of One- and Two-Electron Atoms* (1957), Section 12.
    """
    return z ** 4 * ALPHA_SQUARED / (2.0 * n ** 3 * l * (l + 0.5) * (l + 1))


@pytest.fixture(scope="module")
def boron():
    """B 1s^2 2s^2 2p^1: two ``s`` shells that must give nothing, one ``p`` shell that must."""
    config = ShellConfiguration.parse("1s2 2s2 2p1")
    data = run_scalar_aoc("B", config, basis=BASIS, screening="none", memory_gb=MEMORY_GB,
                          conv_tol=1e-12)
    assert data.converged
    return data, extract_shells(data, config)


# --- 1. The closed form -------------------------------------------------------------------

@pytest.mark.parametrize("configuration,shell,n,l", [("2p1", "2p", 2, 1),
                                                     ("3d1", "3d", 3, 2)])
def test_the_constant_is_the_hydrogenic_one(configuration, shell, n, l):
    """``zeta`` of a one-electron ion, against the analytic answer.

    Li(2+) has one electron, so there is no two-electron screening to argue about and the
    constant is a closed-form function of ``Z``, ``n`` and ``l`` alone. It is the only check
    in this file that involves no other program, and it fails on a wrong factor, a wrong sign,
    a wrong ``l`` dependence and a wrong ``n`` dependence alike — the deviations it is asked
    to distinguish are factors of two and more.

    ⚠ **The tolerance is set by the basis set, not by the extraction, and the test says so
    with a number**: the same basis misses the hydrogenic *orbital energy* by 0.2-0.5%, and
    ``zeta`` follows ``<r^-3>``, which is more sensitive to the shape of the orbital near the
    nucleus than the energy is. The measured deviations are -0.85% (2p) and +0.42% (3d)
    against a relativistic correction of order ``(Z alpha)^2 = 4.8e-4``, so the basis
    dominates by more than an order of magnitude. A tighter bound would be a test of the
    Karlsruhe set's polarization functions.
    """
    config = ShellConfiguration.parse(configuration)
    data = run_scalar_aoc("Li", config, basis="x2c-QZVPall-2c", screening="none",
                          memory_gb=MEMORY_GB, conv_tol=1e-12)
    assert data.converged
    constant = extract_spin_orbit(extract_shells(data, config, shells=(shell,)), data)[shell]

    assert constant.zeta == pytest.approx(hydrogenic_zeta(3, n, l), rel=2e-2)
    assert constant.zeta > 0.0                       # j = l - 1/2 lies lowest
    # The orbital energy's own basis error, quoted so the bound above is not a free parameter.
    assert data.e_scf == pytest.approx(-0.5 * 9.0 / n ** 2, rel=1e-2)


# --- 2. A second route through the two-component machinery ---------------------------------

def test_the_spinor_diagonalization_gives_the_same_splitting(boron):
    """The level pattern and the splitting, computed through the pipeline's own machinery.

    Nothing of the fit is reused: the shell's ``2l+1`` orbitals are expanded into Kramers
    pairs by :mod:`kuiva.spinor.expand` (interleaved columns, not the spin-blocked layout the
    fit works in), the full two-component Hamiltonian is assembled and transformed by
    :func:`kuiva.integrals.transform.transform_1e`, and the six eigenvalues are read off. They
    must fall into a two-fold and a four-fold level separated by ``(2l+1) zeta / 2``.

    ⚠ This is a **frozen-orbital** splitting of one fixed average-of-configuration reference,
    and it is not the self-consistent two-component splitting of the same operator; the two
    differ by tens of per cent and are different physical statements.
    """
    from kuiva.integrals.transform import transform_1e
    from kuiva.spinor.expand import expand_scalar_mos

    data, shells = boron
    shell = shells["2p"]
    constant = extract_spin_orbit(shells, data)["2p"]

    spinors = expand_scalar_mos(shell.coefficients, basis="ao")
    h = transform_1e(data.soc.hamiltonian(), spinors.c)
    energies = np.linalg.eigvalsh(h)

    # ⚠ The bound on each level's spread is 1e-10 Eh, not machine precision, and the floor is
    # the X2C decoupling's own rounding rather than anything here: the discarded
    # time-reversal-odd part of this Hamiltonian is 1.8e-11 Eh and the measured spread is
    # 7.6e-12. In wavenumbers that is 2e-5 cm^-1, four orders below the 0.1 cm^-1 at which a
    # free atom's degeneracy would be a different physical answer.
    assert energies.size == 6
    assert energies[1] - energies[0] < 1e-10         # j = 1/2, two states
    assert energies[5] - energies[2] < 1e-10         # j = 3/2, four states
    splitting = float(energies[2] - energies[0])
    assert splitting == pytest.approx(constant.splitting, rel=1e-6)
    assert constant.splitting_cm == pytest.approx(constant.splitting * 219474.6, rel=1e-6)


# --- 3. What must vanish ------------------------------------------------------------------

def test_an_s_shell_has_no_constant_at_all(boron):
    """``l . s`` is identically zero for ``l = 0``, so an ``s`` shell yields no constant.

    ⚠ It is **absent, not reported as zero**: "the fit returned 0" and "there is nothing to
    fit" are different statements, and only the second is true here.

    ⚠ **And there is deliberately no check that its operator block vanishes**, which is the
    obvious thing to add and would be worthless: an ``l = 0`` shell is a single real orbital
    and the spin-orbit factors are real *antisymmetric*, so ``C^T w C = 0`` is an algebraic
    identity for any vector at all. It is asserted here on a deliberately contaminated ``2s``
    — an orbital that is 23% ``2p`` — precisely to record that the identity holds there too
    and that a check built on it could never fail.
    """
    data, shells = boron
    constants = extract_spin_orbit(shells, data)

    assert [c.shell for c in constants] == ["2p"]
    assert set(constants.as_dict()) == {"2p"}
    with pytest.raises(KeyError, match="s shell has none"):
        constants["1s"]

    contaminated = shells["2s"].coefficients + 0.3 * shells["2p"].coefficients[:, :1]
    for orbital in (shells["1s"].coefficients, shells["2s"].coefficients, contaminated):
        block = shell_spin_orbit_block(replace(shells["1s"], coefficients=orbital), data.soc)
        assert np.max(np.abs(block)) < 1e-25         # measured: 1e-31, i.e. exactly zero


# --- 4. The block is Hermitian and the fit is a projection ---------------------------------

def test_the_operator_block_is_hermitian_and_time_reversal_even(boron):
    """Two structural properties of the block the fit is applied to, both cheap and both
    violated by a sign or a transpose in the Pauli assembly. Time-reversal evenness is what
    makes ``zeta`` real: the Kramers partners of a shell must stay degenerate."""
    data, shells = boron
    block = shell_spin_orbit_block(shells["2p"], data.soc)

    assert np.max(np.abs(block - block.conj().T)) < 1e-16
    eigenvalues = np.linalg.eigvalsh(block)
    assert np.max(np.abs(eigenvalues[0::2] - eigenvalues[1::2])) < 1e-16


def test_the_residual_is_far_below_the_bound_on_a_clean_solution(boron):
    """The operator of a converged spherical atom *is* ``zeta l . s``, to the rounding of the
    X2C decoupling. Measured: 3e-12 Eh absolute on boron, against a decoupling floor of
    2e-11 Eh — so the relative number (6e-8 of a small block) is below the noise, and the
    absolute gate is what keeps this from warning."""
    data, shells = boron
    constants = extract_spin_orbit(shells, data)

    assert constants["2p"].max_residual < 1e-10
    assert constants["2p"].rms_residual <= constants["2p"].max_residual
    assert constants.decoupling_floor == pytest.approx(data.soc.tr_residual)
    assert constants["2p"].operator_scale > 100 * constants["2p"].max_residual


def _fit(shells, shell, coefficients, data):
    """The constant and residual of one shell whose orbitals have been tampered with."""
    broken = AtomicShells(shells=(replace(shell, coefficients=coefficients),),
                          anisotropy=shells.anisotropy, configuration=shells.configuration)
    return extract_spin_orbit(broken, data)[shell.label]


def test_the_residual_catches_orbitals_whose_m_channels_are_wrong(boron, kuiva_caplog):
    """⚠ **The negative control.**

    Three ways the ``2l+1`` orbitals of a shell can stop being what the model assumes, all of
    which have a plausible-looking route into the code: two ``m`` channels **permuted** (the
    integral library stores a p shell as ``px, py, pz``, i.e. ``m = +1, -1, 0``, so an
    extraction that forgot to reorder produces exactly this), two channels **rotated** into
    each other, and one channel **scaled**. Measured: relative residuals of 1.0, 0.39 and
    0.15 against 6e-8 for the clean shell — and the permutation additionally flips the sign
    of ``zeta``, which is a two-fold way of noticing the trap that costs the most elsewhere in
    this project.
    """
    data, shells = boron
    p = shells["2p"]
    clean = extract_spin_orbit(shells, data)["2p"]

    c, s = np.cos(0.4), np.sin(0.4)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    scaled = p.coefficients.copy()
    scaled[:, 1] *= 1.3

    permuted = _fit(shells, p, p.coefficients[:, [1, 0, 2]], data)
    assert permuted.relative_residual > 0.5
    assert permuted.zeta < 0.0 < clean.zeta
    for coefficients in (p.coefficients @ rotation, scaled):
        spoiled = _fit(shells, p, coefficients, data)
        assert spoiled.relative_residual > 1e-2
        assert spoiled.relative_residual > 1e5 * clean.relative_residual
    assert any("reproduced by zeta * l.s only to a relative" in record.getMessage()
               for record in kuiva_caplog.records)


def test_the_residual_does_not_see_an_l_contaminated_channel(boron):
    """⚠ **The boundary of the diagnostic, asserted so it cannot be overstated later.**

    Mix a fifth of the ``2s`` orbital into one ``m`` channel of the ``2p`` shell and this
    residual does not move: the atomic spin-orbit operator is ``f(r) l``, block diagonal in
    ``l``, so an ``s`` admixture contributes no matrix element at all. The radial-parameter
    residual catches exactly this case (the Coulomb interaction couples different ``l``
    freely), which is why the two are reported side by side rather than one standing in for
    the other.
    """
    data, shells = boron
    clean = extract_spin_orbit(shells, data)["2p"]
    mixed = shells["2p"].coefficients.copy()
    mixed[:, 2] += 0.2 * shells["2s"].coefficients[:, 0]

    spoiled = _fit(shells, shells["2p"], mixed, data)
    assert spoiled.relative_residual == pytest.approx(clean.relative_residual, rel=0.5)
    assert spoiled.zeta == pytest.approx(clean.zeta, rel=1e-12)


def test_the_residual_does_not_see_a_non_spherical_solution(boron):
    """⚠ **The other boundary, and the one that would be assumed away.**

    A symmetry-broken ROHF oxygen is the control the shell extraction uses: it converges
    cleanly and its Fock operator is anisotropic by 0.27, nine orders above a converged
    average of configuration. Its spin-orbit residual is **the same** as the clean solution's,
    to within a factor of two — because the one-electron operator is spherical whatever the
    SCF did, and the shell orbitals are built pure-``m`` by construction. What *does* move is
    the constant itself, by 1.7%.

    So the sphericity of the solution is measured by ``AtomicShells.anisotropy`` and by
    nothing else, exactly as it is for the radial parameters.
    """
    config = ShellConfiguration.parse("[He] 2s2 2p4")
    broken = run_scalar_x2c(_Oxygen(), reference="rohf", screening="none",
                            memory_gb=MEMORY_GB)
    clean = run_scalar_aoc("O", config, basis=BASIS, screening="none", memory_gb=MEMORY_GB,
                           conv_tol=1e-12)
    # occupation_tol: an ROHF 2p^4 does not hold 4 electrons in a spherical 2p shell, which
    # is the point of the control and is what the extraction would otherwise refuse.
    broken_shells = extract_shells(broken, config, occupation_tol=1.0)
    clean_shells = extract_shells(clean, config)

    assert broken_shells.anisotropy > 1e8 * clean_shells.anisotropy
    a = extract_spin_orbit(broken_shells, broken)["2p"]
    b = extract_spin_orbit(clean_shells, clean)["2p"]
    assert a.relative_residual == pytest.approx(b.relative_residual, rel=0.5)
    assert a.zeta != pytest.approx(b.zeta, rel=1e-3)


class _Oxygen:
    """The minimal molecule container the bridge accepts, stated inline."""

    unit = "Bohr"
    atoms = [("O", (0.0, 0.0, 0.0))]
    charge = 0
    spin = 2
    basis = BASIS


# --- 5. The screening, and what the constants say about themselves -------------------------

def test_the_two_electron_screening_lowers_the_constant():
    """⚠ **The single largest methodological choice in a spin-orbit constant.**

    The one-electron X2C operator misses the screening of the nucleus by the other electrons,
    which makes spin-orbit splittings 5-30% too large; the atomic mean field supplies it. The
    direction is not negotiable and the size is large — measured on boron's 2p: 21.8 cm^-1
    unscreened against 12.5 cm^-1 screened, a reduction of 43%.

    What is asserted is the direction and the fact that the record says which of the two a
    given constant is. The absolute values are not asserted: they are frozen-orbital constants
    of one average-of-configuration reference in a split-valence basis, and quoting either as
    an accuracy would be quoting a construction and a basis.
    """
    config = ShellConfiguration.parse("1s2 2s2 2p1")
    zetas, provenance = {}, {}
    for screening in ("none", "x2camf"):
        data = run_scalar_aoc("B", config, basis=BASIS, screening=screening,
                              memory_gb=MEMORY_GB, conv_tol=1e-12)
        constants = extract_spin_orbit(extract_shells(data, config), data)
        zetas[screening] = constants["2p"].zeta
        provenance[screening] = constants.provenance

    assert 0.0 < zetas["x2camf"] < zetas["none"]
    assert zetas["x2camf"] / zetas["none"] < 0.9
    assert provenance["none"]["screening"]["method"] == "none"
    assert provenance["x2camf"]["screening"]["method"] == "x2camf"


def test_it_refuses_a_solution_that_carries_no_spin_orbit_operator():
    """``with_soc=False`` is the ordinary way to run this feature when only the radial
    parameters are wanted — it saves a four-component atomic solve — so asking for constants
    afterwards must be a clear refusal rather than an attribute error two frames down."""
    config = ShellConfiguration.parse("1s2 2s2 2p1")
    data = run_scalar_aoc("B", config, basis=BASIS, with_soc=False, memory_gb=MEMORY_GB)
    shells = extract_shells(data, config)

    with pytest.raises(ValueError, match="with_soc=False"):
        extract_spin_orbit(shells, data)


def test_the_constants_can_be_reported_and_looked_up(boron, kuiva_caplog):
    """The table goes through the output grammar, and the lookup is by shell label."""
    import logging

    data, shells = boron
    constants = extract_spin_orbit(shells, data)
    with kuiva_caplog.at_level(logging.INFO):
        constants.report()

    assert any("H_SO = zeta l.s" in record.getMessage() for record in kuiva_caplog.records)
    assert constants["2p"] is constants[0]
    assert constants.max_relative_residual < ZETA_RESIDUAL_TOLERANCE * 1e3
    assert "2p" in repr(constants)
