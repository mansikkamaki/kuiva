"""The PySCF four-component atomic backend.

This is the *only* implementation of :class:`~kuiva.amf.backend.AtomicDiracBackend` today, and
it is deliberately thin: run ``pyscf.scf.dhf`` on a single atom, convert what comes back into
the conventions of kuiva/spinor/expand.py, and hand over plain arrays. Everything specific to PySCF —
the ``Mole`` object, the j-adapted spinor basis, the SCF driver — stops here.

Three things this module has to get right, and each of them was measured rather than assumed
-------------------------------------------------------------------------------------------
**1. The basis change from j-adapted spinors to spin-blocked spin-orbitals.** PySCF's
four-component code works in the j-adapted 2-spinor basis (``nao_2c = 2 * nao``), while
kuiva/spinor/expand.py fixes the spin-blocked ``[alpha; beta]`` spin-orbital basis for everything in
Kuiva. The transformation between them is ``U = [C_alpha ; C_beta]`` from
``Mole.sph2spinor_coeff()``, which is **unitary** — verified numerically here
(``max |U U^dag - 1| = 2e-16``), not taken on trust — so an operator or a density transforms
the same way::

    A_spin_orbital = U A_spinor U^dag

The direction matters and is easy to get backwards; the inverse map ``U^dag A_so U`` goes the
other way, and it is what the round-trip test in ``tests/test_amf_backend.py`` uses to check
this against PySCF's own spin-orbital X2C helper. Because ``U`` is unitary, operators and
densities transform identically at this step — that is a property of ``U``, **not** a general
licence, and the conjugation trap still applies everywhere else (it bites in
:mod:`kuiva.amf.decouple`, where ``R`` is not unitary).

**2. Near-linear dependence of the four-component metric, which is not optional to handle.**
The X2C decoupling is done in the **uncontracted** basis, and the 4c
metric's small-component block is ``T / (2 c^2)``, which pushes its smallest eigenvalues
several orders of magnitude below the large-component block's. Measured on the decontracted
``x2c-SVPall-2c`` sets:

===== ================== ====================== =========================
atom  4c metric min eig  plain generalized eigh  canonical orthogonalization
===== ================== ====================== =========================
Ne    7.5e-06            -128.58991792, conv     -128.58991792, conv
Ar    3.8e-09            **-328.836, NOT conv**  **-528.52205540, conv**
===== ================== ====================== =========================

Argon's plain solve stalls after 100 cycles at a **200 Eh error** and reports nothing but
``converged = False``; with canonical orthogonalization of the metric it converges in 2.5 s.
Neon, whose metric is better conditioned, converges either way and to the *same* energy to all
printed digits — which is the evidence that the projection is benign rather than merely
convenient, and is asserted as a test.

**3. Open shells are averaged, never aufbau-occupied.** An open-shell atom run through an
aufbau ``get_occ`` picks one arbitrary determinant out of a degenerate manifold, breaks the
spherical symmetry of the atom, and produces a mean field with a spurious spatial orientation
baked into it — a wrong answer that converges cleanly and looks entirely reasonable.
:func:`_average_of_configuration_occupation` instead spreads the open-shell electrons equally
over the whole frontier ``l`` shell, which is spherical by construction, and
:func:`density_anisotropy` then *asserts* that the converged density really is spherical.
The reference configuration is an explicit, canonical object
(:mod:`kuiva.amf.configuration`), because for an ion it is a genuine choice that changes the
answer and has to be part of the cache key.

⚠ **And the occupations are not enough, which was measured the hard way.** "Spherical by
construction" holds for the density *given* spherical orbitals; it says nothing about whether
the iteration stays there, and it does not. The symmetric solution is an **unstable fixed
point** — a fractionally occupied Hartree-Fock functional has broken-symmetry solutions below
it — so the anisotropy grows about an order of magnitude per cycle from roundoff until the
guard above refuses the result, or, worse, until the SCF stops just under it. It showed on
Ti(+1) ``s7 p12 d2`` and the shape that produced it (two partly filled channels in an ion) is
the shape every Ln(I) reference has. The symmetry is therefore **imposed**: the Fock is
projected onto its rank-zero part every cycle
(:func:`kuiva.amf.configuration.spherical_projector` over
:func:`spinor_symmetry_groups`), which is exact by the Wigner-Eckart theorem and is the
identity at a spherical fixed point, so nothing that was already clean moves.

**4. Basis contraction is handled here and nowhere else.** X2C decoupling belongs in the
primitive basis, so the sequence is decontract, solve, contract back — the
first and last in :meth:`PySCFDiracBackend._build_mole` and
:meth:`kuiva.amf.backend.AtomicDiracSolution.contract`, with :mod:`kuiva.amf.decouple` kept
basis-agnostic and PySCF-free. The contraction matrix is **taken from the basis**
(``Mole.decontract_basis(aggregate=True)``) and never reconstructed from the shell structure,
because aggregation merges primitives shared between different shells of the same angular
momentum and a per-shell block-diagonal reconstruction is then wrong while still having the
right shape. :func:`_validate_contraction` checks the resulting matrix against two
one-electron operators rather than trusting it; see its docstring for which failure mode is
silent and why that is the one worth two integral evaluations.

References
----------
* PySCF: Q. Sun et al., J. Chem. Phys. 153, 024109 (2020), doi:10.1063/5.0006074.
* Dirac-Hartree-Fock as implemented there: following I. P. Grant, "Relativistic Quantum Theory
  of Atoms and Molecules", Springer (2007), and the Dirac-Coulomb / Gaunt / Breit operators of
  K. G. Dyall, K. Faegri, "Introduction to Relativistic Quantum Chemistry", Oxford University
  Press (2007), ch. 4 and 11.
* Canonical orthogonalization: P.-O. Loewdin, Adv. Phys. 5, 1 (1956),
  doi:10.1080/00018735600101155 — the scheme :mod:`kuiva.orth.canonical` implements for the
  molecular working basis, applied here to the four-component metric.
* The j-adapted spinor basis and its relation to the spin-orbital basis: K. G. Dyall,
  K. Faegri, "Introduction to Relativistic Quantum Chemistry", Oxford University Press (2007),
  ch. 6.
"""
from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..basis.registry import classify_contraction
from ..util import resources as res
from ..util.logging import get_logger
from ..util.timing import timer
from .backend import (AtomicDiracSolution, FourComponentBlocks, INTERACTIONS,
                      METRIC_LINDEP_THRESHOLD as _METRIC_LINDEP_THRESHOLD,
                      dirac_scf_memory_gb, metric_keep_mask as _metric_keep_mask,
                      solution_memory_gb)
from .configuration import (SHELL_LETTERS, AtomicConfiguration, OpenShell,
                            average_occupations, install_configuration_average,
                            spherical_projector)

log = get_logger(__name__)

#: Eigenvalues of the (normalized) four-component metric dropped before the generalized
#: eigenproblem is solved. See point 2 of the module docstring: this is not a tuning parameter
#: for accuracy, it is what makes the uncontracted 4c SCF converge at all.
#:
#: ⚠ **Imported, not defined here.** Every operation on this metric must project it the same
#: way — the X2C decoupling of :mod:`kuiva.x2c.decouple`, the molecular one-electron path, and
#: this SCF. See the constant's docstring in :mod:`kuiva.x2c.decouple` for the 96%
#: time-reversal-odd correction that resulted when two of them disagreed.
METRIC_LINDEP_THRESHOLD = _METRIC_LINDEP_THRESHOLD

#: Maximum dimensionless anisotropy of the converged atomic density for the mean field to be
#: usable — see :func:`density_anisotropy`. An atom is spherical, so this is not a
#: closed-shell restriction: it is the check that the *solution* is spherical too. A closed
#: shell is (measured: 1e-12 for Ne and Ar); an aufbau-occupied open shell is anisotropic at
#: order unity; and an average-of-configuration solution is spherical again (measured: 1e-13
#: for O, 8e-13 for Ti(3+)). Nothing lives in between, so the threshold is not delicate.
#:
#: ⚠ **Under average-of-configuration this check changes role rather than going away.**
#: Fractional occupation over a whole ``l`` shell restores the symmetry an aufbau occupation
#: broke, so a solution that still comes back anisotropic has *not* averaged — which is the
#: silent failure mode of the open-shell path, and this is what catches it.
SPHERICAL_DENSITY_TOLERANCE = 1e-6

#: Kept as the former name of :data:`SPHERICAL_DENSITY_TOLERANCE`, whose meaning widened when
#: average-of-configuration made "closed shell" the wrong word for what it measures.
CLOSED_SHELL_ANISOTROPY = SPHERICAL_DENSITY_TOLERANCE


@contextlib.contextmanager
def light_speed(c: Optional[float]):
    """Temporarily set PySCF's speed of light (``lib.param.LIGHT_SPEED``).

    ⚠ **This is a process-global.** It is a context manager rather than an argument because
    PySCF reads the global from inside integral evaluation and from the X2C helpers, so there
    is no way to pass it locally. Every consumer must therefore keep it set for the *whole* of
    a correction — the four-component SCF, the decoupling and the subtracted mean field — or
    the two halves of the X2CAMF subtraction are computed at different speeds of light and their
    difference
    is meaningless rather than zero. :func:`kuiva.amf.atomic.atomic_correction` is where that
    sequencing is enforced.

    ``c=None`` is a no-op, so the physical case pays nothing and touches nothing.
    """
    if c is None:
        yield
        return
    from pyscf import lib
    previous = lib.param.LIGHT_SPEED
    lib.param.LIGHT_SPEED = float(c)
    log.debug("speed of light set to %.6e a.u. (was %.6e)", c, previous)
    try:
        yield
    finally:
        lib.param.LIGHT_SPEED = previous


def current_light_speed() -> float:
    """PySCF's speed of light [a.u.] as it stands right now."""
    from pyscf import lib
    return float(lib.param.LIGHT_SPEED)


# --- Helpers ------------------------------------------------------------------------------

def _canonical_orthogonalization(s: np.ndarray, threshold: float) -> np.ndarray:
    """``X`` with ``X^dag S X = 1``, dropping directions below ``threshold`` (canonical orthogonalization with linear-dependence removal).

    The metric is normalized to unit diagonal first, so that ``threshold`` is a statement
    about redundancy rather than about the units of whatever the small-component block was
    scaled by — a raw threshold on the 4c metric would be dominated by the ``1/(2c^2)`` factor
    and would drop physical functions.

    ⚠ **The cut goes through :func:`kuiva.x2c.decouple.metric_keep_mask`**, so whole degenerate
    groups are dropped and this projection and
    :func:`kuiva.x2c.decouple.canonical_orth` remain the *same* operation on the *same* metric
    — which the single-definition rule of ``kuiva/x2c`` requires, and which two copies of one line would eventually stop being.
    """
    d = np.real(np.diag(s))
    norm = 1.0 / np.sqrt(np.where(d > 0.0, d, 1.0))
    sn = norm[:, None] * s * norm[None, :]
    val, vec = np.linalg.eigh(sn)
    keep = _metric_keep_mask(val, threshold)
    x = vec[:, keep] / np.sqrt(val[keep])
    return norm[:, None] * x


def eigh_canonical(threshold: float):
    """An ``_eigh`` for a PySCF SCF object that is safe on a near-singular metric.

    Replaces the plain generalized eigensolve. PySCF calls ``_eigh(h, s, *args, **kwargs)``,
    so the extra arguments are accepted and ignored.
    """
    def eigh(h, s, *args, **kwargs):
        x = _canonical_orthogonalization(np.asarray(s), threshold)
        e, c = np.linalg.eigh(x.conj().T @ np.asarray(h) @ x)
        return e, x @ c
    return eigh


def _angular_momentum_map(mol) -> np.ndarray:
    """``l`` of every function of the j-adapted 2-spinor basis, as an ``(nao_2c,)`` array.

    Each shell of angular momentum ``l`` contributes ``nctr * 2 * (2l+1)`` spinor functions —
    the ``j = l - 1/2`` and ``j = l + 1/2`` blocks together — and they are contiguous, so the
    map is exact rather than a heuristic. It is derived from the shell data rather than by
    parsing ``Mole.spinor_labels()``, because a label is a display string and this is used to
    decide which electrons go where.
    """
    ls = []
    for ib in range(mol.nbas):
        l = int(mol.bas_angular(ib))
        ls.extend([l] * (int(mol.bas_nctr(ib)) * 2 * (2 * l + 1)))
    ls = np.asarray(ls, dtype=int)
    if ls.size != int(mol.nao_2c()):
        raise RuntimeError(
            "the angular-momentum map has {} entries for a {}-function spinor basis; the "
            "shell layout assumed here does not match this Mole".format(
                ls.size, mol.nao_2c()))
    return ls


def spinor_symmetry_groups(mol) -> "List[np.ndarray]":
    """The ``(l, j)`` symmetry classes of the j-adapted 2-spinor basis, as index arrays.

    One ``(n_radial, 2j+1)`` array per class, columns in ascending ``m_j`` — the input
    :func:`kuiva.amf.configuration.spherical_projector` needs to project an atomic operator
    onto its spherical part.

    Derived from the shell data rather than from :meth:`Mole.spinor_labels`, for the reason
    :func:`_angular_momentum_map` is: a label is a display string, and this decides which
    matrix elements survive. Each shell of angular momentum ``l`` lays out, per contraction,
    the ``j = l - 1/2`` block (``2l`` functions) and then the ``j = l + 1/2`` block
    (``2l + 2``), each in ascending ``m_j``, and they are contiguous.
    """
    classes: Dict[Tuple[int, int], List[np.ndarray]] = {}
    position = 0
    for ib in range(mol.nbas):
        l = int(mol.bas_angular(ib))
        for _ in range(int(mol.bas_nctr(ib))):
            for two_j in ((2 * l - 1, 2 * l + 1) if l else (1,)):
                classes.setdefault((l, two_j), []).append(
                    np.arange(position, position + two_j + 1))
                position += two_j + 1
    if position != int(mol.nao_2c()):
        raise RuntimeError(
            "the spinor symmetry classes cover {} functions of a {}-function spinor basis; "
            "the shell layout assumed here does not match this Mole".format(
                position, mol.nao_2c()))
    return [np.stack(blocks) for blocks in classes.values()]


def _average_of_configuration_occupation(mol, configuration, c: float, state=None):
    """A ``get_occ`` implementing **average-of-configuration** occupation.

    Two things are decided here, and they are separable on purpose.

    **1. Which branch is electronic — by energy, never by index.** ⚠ PySCF's own ``get_occ``
    locates the positive-energy branch by index arithmetic (``n2c = len(mo_energy) // 2``,
    then occupy ``[n2c : n2c + nelectron]``), which presumes exactly half the eigenvectors are
    positronic. Canonical orthogonalization of the metric (point 2 of the module docstring)
    drops a handful of directions, all from the small component, so that presumption fails by
    however many were dropped and the occupied set silently slides up the spectrum — a
    converged SCF for the wrong state with no warning anywhere. Selecting by energy is correct
    however many vectors survive: the electronic states sit near zero (the deepest is about
    ``-Z^2/2``, i.e. -5000 Eh even at Z = 100) and the positronic branch below ``-2 c^2`` =
    -37558 Eh, so ``-c^2`` separates them with four orders of magnitude to spare, and it keeps
    doing so at the modified ``c`` of the non-relativistic-limit test.

    **2. Which electronic spinors are occupied, and by how much — by angular momentum.** Each
    spinor is assigned to an ``l`` channel by the Mulliken weight of its **large component**
    (an atomic four-component spinor's large component has definite ``l``, so this is a clean
    assignment and not a partition of something genuinely mixed). Within a channel, spinors
    are filled in energy order: ``N_l // (4l+2)`` shells fully, and the remaining
    ``q = N_l mod (4l+2)`` electrons spread **equally over all ``4l+2`` spinors of the frontier
    shell** at occupation ``q / (4l+2)``.

    That last step is the whole of average-of-configuration, and what it buys is worth stating.
    An aufbau occupation of an open shell picks one arbitrary determinant out of a degenerate
    manifold, spontaneously breaks the spherical symmetry of the atom, and bakes an arbitrary
    spatial orientation into the mean field — a wrong answer that converges cleanly and looks
    entirely reasonable (see :func:`density_anisotropy`). The fractional occupation is
    spherical by construction, and it is spherical whichever ``j`` sub-shell the frontier
    electrons land in, so no choice between ``p_1/2`` and ``p_3/2`` is ever made.

    ⚠ **Assignment by ``l`` rather than by index is what makes a cold start reproducible.**
    Filling "the next ``4l+2`` spinors after the core" presumes the frontier shell is the next
    one in energy, which is exactly what is *not* reliable for a lanthanide: 4f, 5d and 6s lie
    within an eV of each other and their order changes during the SCF. Counting per channel is
    invariant to that, because within one ``l`` the ordering is never in doubt.

    The filling rule itself lives in :func:`kuiva.amf.configuration.average_occupations`, and
    is shared with the two-component validation SCF that checks the resulting correction
    against four-component theory: the two must occupy *the same configuration the same way*,
    or that comparison is between two different states.
    """
    lmap = _angular_momentum_map(mol)
    n2c = int(mol.nao_2c())
    channels = sorted(set(int(l) for l in lmap))
    overlap = np.asarray(mol.intor("int1e_ovlp_spinor"))
    # Refuse a basis with no functions of an occupied l up front, where the message can name
    # the basis rather than describing a shortfall found halfway through an SCF.
    missing = [l for l, n in enumerate(configuration.occupations) if n and l not in channels]
    if missing:
        raise ValueError(
            "the configuration {} needs {} functions, which this basis does not have".format(
                configuration.canonical,
                "/".join(SHELL_LETTERS[l] for l in missing)))

    def get_occ(mo_energy=None, mo_coeff=None):
        if mo_coeff is None:
            raise RuntimeError(
                "average-of-configuration occupation needs the orbital coefficients to "
                "resolve the angular momentum of each spinor; get_occ was called with "
                "energies only")
        e = np.asarray(mo_energy, dtype=float)
        mo = np.asarray(mo_coeff)
        occ = np.zeros(e.size)
        electronic = np.where(e > -c * c)[0]
        if electronic.size < configuration.n_electrons:
            raise RuntimeError(
                "only {} positive-energy spinors survive for {} electrons; the metric "
                "projection has removed too much".format(
                    electronic.size, configuration.n_electrons))

        # Mulliken population of each electronic spinor on the large component, per l.
        large = mo[:n2c]
        population = np.real(np.conj(large) * (overlap @ large))
        weights = np.stack([population[lmap == l].sum(axis=0) for l in channels])
        assigned = np.asarray(channels)[np.argmax(weights, axis=0)]
        # Positronic solutions are excluded by being assigned an l no configuration occupies,
        # rather than by slicing — so the indices the filling returns are indices into the
        # full spectrum and nothing has to be mapped back.
        assigned[np.setdiff1d(np.arange(e.size), electronic)] = -1
        occ[:] = average_occupations(configuration, e, assigned)
        if state is not None:
            # ⚠ **The orbitals are captured here because there is nowhere else to get them.**
            # PySCF's SCF loop keeps ``mo_coeff`` in a local and only assigns
            # ``mf.mo_coeff`` *after* convergence, so a shell-dependent Fock that read the
            # attribute would see ``None`` on every cycle and silently fall back to the plain
            # one — the SCF converges, to the wrong functional, with nothing to show for it.
            # ``get_occ`` is the one hook the loop calls with the current coefficients.
            state["mo_coeff"], state["mo_occ"] = mo, occ
            shells = state.setdefault("shells", [])
            # ⚠ Recorded here and nowhere else: the shell-dependent Fock needs to know which
            # spinors form each open shell, and this is the only place that is resolved (by
            # Mulliken assignment to an ``l`` channel, then energy order within it). Deriving
            # it again from ``mo_occ`` alone would mean grouping by occupation *value*, which
            # merges two open shells that happen to share a fraction — possible in principle
            # and silent when it happens.
            shells[:] = []
            for l, n_electrons in enumerate(configuration.occupations):
                degeneracy = 4 * l + 2
                q = n_electrons % degeneracy
                if not q:
                    continue
                index = np.where((assigned == l) & (occ > 1e-12) & (occ < 1.0 - 1e-12))[0]
                if index.size != degeneracy:
                    raise RuntimeError(
                        "the {} open shell came back on {} spinors where {} were filled "
                        "fractionally".format(SHELL_LETTERS[l], index.size, degeneracy))
                shells.append(OpenShell(l, q, degeneracy, index))
            state["shells"] = shells
        return occ
    return get_occ


def spinor_to_spin_orbital(mol) -> np.ndarray:
    """``U`` mapping the j-adapted spinor basis onto the spin-blocked ``[alpha; beta]`` one.

    Returns ``(2*nao, nao_2c)``. See point 1 of the module docstring; the unitarity is
    checked rather than assumed, because every matrix this module returns is transformed with
    it and a silent failure would look like a plausible but wrong spin-orbit operator.
    """
    ca, cb = mol.sph2spinor_coeff()
    u = np.vstack([np.asarray(ca), np.asarray(cb)])
    err = float(np.max(np.abs(u @ u.conj().T - np.eye(u.shape[0]))))
    if err > 1e-10:
        raise RuntimeError(
            "the spinor-to-spin-orbital transformation is not unitary (max |U U^dag - 1| = "
            "{:.2e}). Every matrix this backend returns is transformed with it, so this "
            "cannot be allowed through.".format(err))
    return u


def density_anisotropy(mol, density: np.ndarray) -> float:
    """How far the converged atomic density is from spherical, dimensionless.

    The traceless quadrupole moment of the charge density about the nucleus,
    ``Q_ij = <3 x_i x_j - r^2 d_ij>``, divided by ``<r^2>``. An **atom** is spherically
    symmetric, so a correct solution gives zero up to SCF convergence whatever its
    configuration; anything else does not.

    ⚠ **This replaces a frontier-gap test, which does not work, and the reason is worth
    recording.** A partially filled degenerate manifold *ought* to show a zero HOMO-LUMO gap —
    but an aufbau SCF spontaneously breaks the spherical symmetry to lower its energy, splits
    the manifold, and converges to a state with a perfectly healthy gap. Measured on
    oxygen (2p^4): the gap test passes and the density is anisotropic at order one. The gap
    test would therefore have let exactly the case it was written to catch straight through,
    and produced an atomic mean field with a spurious spatial orientation baked into it.

    ⚠ **Under average-of-configuration this becomes the assertion that the averaging worked**,
    which is a stronger use of the same measurement than the closed-shell
    guard it began as. Fractional occupation of a whole ``l`` shell restores the symmetry the
    aufbau occupation broke; if the density still comes back anisotropic then the frontier
    manifold was not the one the configuration named, and the mean field again carries a
    spurious orientation. Measured on average-of-configuration solutions: 1.5e-12 (C 2p2),
    3.4e-13 (O 2p4), 7.8e-13 (Ti(3+) 3d1) — the same population as a closed shell.

    Only the large component is used. The small component carries ``O(1/c^2)`` of the density
    and cannot rescue a sphericity that the large component does not have.

    ``density`` is either the spin-blocked ``(2 nao, 2 nao)`` large-component density — traced
    over spin here — or an already spin-traced scalar ``(nao, nao)`` one, which is what a
    spin-restricted average-of-configuration SCF produces. **Public and shared** for the
    reason the filling rule is: the four-component backend and the front-end's scalar AOC SCF
    are asserting the same physical property of the same atom, and a second measurement of it
    could disagree about what "spherical" means.
    """
    nao = mol.nao
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        rr = np.asarray(mol.intor("int1e_rr")).reshape(3, 3, nao, nao)
        r2 = np.asarray(mol.intor("int1e_r2"))
    density = np.asarray(density)
    if density.shape[0] == 2 * nao:
        scalar = np.real(density[:nao, :nao] + density[nao:, nao:])    # trace over spin
    elif density.shape[0] == nao:
        scalar = np.real(density)
    else:
        raise ValueError(
            "a density of dimension {} is neither the scalar ({}) nor the spin-blocked ({}) "
            "AO basis of this Mole".format(density.shape[0], nao, 2 * nao))
    q = np.einsum("xyij,ji->xy", rr, scalar)
    mean_r2 = float(np.einsum("ij,ji->", r2, scalar))
    q = 3.0 * q - np.trace(q) * np.eye(3)
    return float(np.max(np.abs(q))) / (abs(mean_r2) or 1.0)


def _basis_label(basis: object, uncontract: bool) -> str:
    """A short human-readable label for provenance. Parsed basis data has no name, so it is
    described by its **measured** contraction type and size rather than given one it does not
    have — see :func:`kuiva.basis.registry.classify_contraction` for why measuring beats
    trusting a name here."""
    if isinstance(basis, str):
        name = basis
    else:
        try:
            kind = classify_contraction(basis).value
        except (ValueError, TypeError, IndexError):
            kind = "unclassifiable"
        try:
            name = "custom {} ({} shells)".format(kind, len(basis))
        except TypeError:
            name = "custom {}".format(kind)
    return name + (" [uncontracted]" if uncontract else "")


#: Relative tolerance on the contraction round-trip check of :func:`_validate_contraction`.
#: It is a *structural* check, not a numerical one — the identity holds exactly in exact
#: arithmetic, and the measured residual across every supported basis family is 1e-16 to 1e-14
#: (worst: ANO-RCC, whose 21-primitive general contractions cancel hardest). The bound is
#: therefore two orders above the worst measurement and still four below anything a genuine
#: ordering or normalization mismatch could produce, which is order unity.
CONTRACTION_TOL = 1e-11


def _validate_contraction(mol, xmol, contraction: np.ndarray) -> float:
    """Assert that ``contraction`` really maps the primitive basis onto the molecular one.

    The whole of the contract-back path rests on one claim: that
    ``A_molecular = C^T A_primitive C`` for any one-electron operator, with ``C`` the matrix
    ``Mole.decontract_basis`` returns. That claim has two independent failure modes and only
    one of them is caught by anything else:

    * a **size** mismatch, which :meth:`PySCFDiracBackend.coulomb_mean_field` already refuses
      and which is loud;
    * an **AO ordering or normalization** mismatch, which is silent, produces a Hermitian
      correction of an entirely plausible magnitude, and would be attributed to the physics.

    The second is the trap this function exists for, and it costs two atomic one-electron
    integral evaluations to close. Both operators are used deliberately: the
    overlap is what a normalization error breaks, and the nuclear attraction is a *different*
    operator with a different radial weighting, so a permutation that happened to preserve
    the overlap could not also preserve it. Returns the worst relative residual, for the log.

    ⚠ Note what is **not** assumed: nothing here reconstructs ``C`` from the shell structure.
    ``decontract_basis(aggregate=True)`` merges primitives shared between different shells of
    the same angular momentum, so a per-shell block-diagonal reconstruction is wrong for any
    basis that reuses an exponent — and it is wrong *silently*, since it still has the right
    shape. Taking the matrix from the basis and checking it is the point.
    """
    worst = 0.0
    for intor in ("int1e_ovlp", "int1e_nuc"):
        target = np.asarray(mol.intor_symmetric(intor))
        back = contraction.T @ np.asarray(xmol.intor_symmetric(intor)) @ contraction
        scale = float(np.max(np.abs(target))) or 1.0
        worst = max(worst, float(np.max(np.abs(back - target))) / scale)
    if worst > CONTRACTION_TOL:
        raise NotImplementedError(
            "the decontraction of this basis does not round-trip: C^T A_primitive C differs "
            "from A_molecular by {:.2e} relative, against a tolerance of {:.0e}. The atomic "
            "mean-field correction is computed in the primitive basis and contracted back "
            "with C, so a correction built from this basis would be expressed in a basis "
            "that is not the molecule's — silently, and with an entirely plausible "
            "magnitude. Either the AO ordering or the normalization of the two bases "
            "disagrees. Refusing rather than returning a wrong answer; pass uncontract=False "
            "to do the decoupling in the contracted basis instead (the basis-set policy advises "
            "against it, at a measured cost of a few tenths of a percent on a "
            "spin-orbit splitting).".format(worst, CONTRACTION_TOL))
    return worst


# --- The backend --------------------------------------------------------------------------

class PySCFDiracBackend:
    """Four-component atomic Dirac-Hartree-Fock through PySCF (see the module docstring)."""

    name = "pyscf"

    @property
    def version(self) -> str:
        import pyscf
        return str(pyscf.__version__)

    # -- the atomic Mole ------------------------------------------------------------------

    @staticmethod
    def _build_mole(element: str, basis: object, charge: int = 0,
                    uncontract: bool = True):
        """The ``Mole`` the four-component problem is solved in, and its contraction matrix.

        Deterministic in its arguments alone — which is what lets
        :meth:`coulomb_mean_field` rebuild exactly the same basis later from the solution's
        ``basis_spec``, instead of the solution having to carry a live PySCF object.

        This is where **all** of the contraction handling lives. The three
        steps are decontract, solve, contract back; the first and third are here and in
        :meth:`kuiva.amf.backend.AtomicDiracSolution.contract`, and :mod:`kuiva.amf.decouple`
        is deliberately basis-agnostic and never sees any of it.
        """
        from pyscf import gto

        n_electrons = int(gto.charge(element)) - int(charge)
        mol = gto.M(atom=[(element, (0.0, 0.0, 0.0))], basis={element: basis},
                    charge=charge, spin=n_electrons % 2, verbose=0)
        if mol.has_ecp():
            raise NotImplementedError(
                "the atomic mean-field correction requires an all-electron basis; {} was "
                "given one with an ECP. The supported families are all-electron by "
                "design, and X2C has no meaning with a pseudopotential core.".format(element))
        if not uncontract:
            return mol, None
        try:
            xmol, contraction = mol.decontract_basis(aggregate=True)
        except Exception as exc:                                            # noqa: BLE001
            raise NotImplementedError(
                "the basis given for {} cannot be decontracted ({}: {}). X2C decoupling "
                "belongs in the primitive basis, so this basis cannot be "
                "used with uncontract=True; pass uncontract=False to decouple in the basis "
                "as given.".format(element, type(exc).__name__, exc))
        contraction = np.ascontiguousarray(np.asarray(contraction, dtype=float))
        if contraction.shape != (int(xmol.nao), int(mol.nao)):
            raise NotImplementedError(
                "decontracting the basis for {} gave a contraction matrix of shape {} where "
                "({}, {}) was needed. The correction could not be expressed in the "
                "molecular basis.".format(element, contraction.shape, xmol.nao, mol.nao))
        residual = _validate_contraction(mol, xmol, contraction)
        log.debug("%s basis decontracted %d -> %d functions, round-trip residual %.1e",
                  element, mol.nao, xmol.nao, residual)
        return xmol, contraction

    # -- the protocol ---------------------------------------------------------------------

    def solve(self, element: str, basis: object, *, charge: int = 0,
              configuration=None, interaction: str = "coulomb",
              uncontract: bool = True, conv_tol: float = 1e-10,
              max_cycle: int = 100, spherical: bool = True) -> AtomicDiracSolution:
        """Converge a four-component atomic calculation, average-of-configuration if open.

        Parameters
        ----------
        element : str
            Element symbol.
        basis : str or parsed PySCF basis data
            Passed straight to ``gto.M``. Passing the *parsed* basis of the molecule being
            corrected (``mol._basis[symbol]``) rather than a family name is what guarantees
            the atomic calculation uses the same functions the molecule does; a name would be
            re-resolved and could differ.
        charge : int
            ⚠ A hint only. The **configuration** decides the charge state that is actually
            solved, so that the two can never disagree; passing a charge without a matching
            configuration gets the neutral reference and says so at DEBUG.
        configuration : AtomicConfiguration, str or None
            The reference configuration the mean field is taken over — ``"[Ar]3d1"``,
            ``"[Xe]4f9"``, or an :class:`~kuiva.amf.configuration.AtomicConfiguration`.
            ``None`` takes the neutral ground configuration, which is defined only for a
            neutral species: an ion's is chemistry rather than arithmetic and is refused
            rather than guessed (see :mod:`kuiva.amf.configuration`). Open shells are occupied
            **fractionally and equally** over the whole frontier ``l`` shell, which is what
            keeps the atomic mean field spherical.
        interaction : str
            ``"coulomb"``, ``"gaunt"`` or ``"breit"``.
        uncontract : bool
            Solve in the fully decontracted basis and return the contraction matrix. The
            default, and the physically correct choice — X2C decoupling belongs in the
            primitive basis.
        spherical : bool
            Constrain the SCF to spherically symmetric solutions by projecting the Fock
            operator onto its rank-zero part each cycle
            (:func:`kuiva.amf.configuration.spherical_projector`). ⚠ **On by default and
            never off in production**: without it the spherical solution is an *unstable*
            fixed point for an open shell — the anisotropy grows about an order of magnitude
            per cycle from roundoff until the density guard below refuses the result, or,
            worse, stops just under it. ``False`` exists to measure exactly that and warns.

        Notes
        -----
        The **speed of light is not an argument**: it is PySCF's process-global and must be
        set for the whole of a correction, not for one call inside it. Use the
        :func:`light_speed` context manager, as :func:`kuiva.amf.atomic.atomic_correction`
        does.
        """
        from pyscf.scf import dhf

        if interaction not in INTERACTIONS:
            raise ValueError("unknown two-electron interaction {!r}; expected one of {}"
                             .format(interaction, INTERACTIONS))

        from pyscf import gto

        element = element.capitalize()
        config = AtomicConfiguration.coerce(configuration, element)
        # ⚠ The configuration is the single source of truth for the charge state, not the
        # ``charge`` argument. They cannot then disagree, and the default — a neutral
        # reference for every element, whatever the molecule's charge — is stated in one
        # place (:mod:`kuiva.amf.configuration`) rather than reconstructed here.
        reference_charge = int(gto.charge(element)) - config.n_electrons
        if int(charge) and int(charge) != reference_charge:
            log.debug("%s: reference configuration %s implies charge %+d, not the %+d asked "
                      "for; the configuration wins", element, config.canonical,
                      reference_charge, int(charge))
        xmol, contraction = self._build_mole(element, basis, reference_charge, uncontract)
        nao = int(xmol.nao)

        # Both estimates are exact functions of nao, which is known here and nowhere
        # earlier. The SCF's own arrays are `require`d rather than `reserve`d because they are
        # transient — PySCF frees them when the SCF object goes out of scope.
        res.require("four-component atomic SCF on {} ({} functions)".format(element, nao),
                    dirac_scf_memory_gb(nao),
                    advice=["use a smaller basis for this element",
                            "pass uncontract=False, which roughly halves the basis at the "
                            "cost of doing the X2C decoupling in a contracted basis "
                            "(decoupling in a contracted basis is advised against)"])
        reservation = res.reserve(
            "atomic four-component solution for {}".format(element), solution_memory_gb(nao),
            note="{} scalar functions".format(nao))

        with timer("atomic 4c DHF"):
            mf = dhf.DHF(xmol)
            mf.verbose = 0
            mf.conv_tol = conv_tol
            mf.max_cycle = max_cycle
            mf.with_gaunt = interaction in ("gaunt", "breit")
            mf.with_breit = interaction == "breit"
            # Point 2 of the module docstring: without this the uncontracted 4c SCF does not
            # converge, and reports a 200 Eh error as "not converged" and nothing else.
            mf._eigh = eigh_canonical(METRIC_LINDEP_THRESHOLD)
            # ...and, because that changes how many eigenvectors come back, the occupation
            # must be chosen by energy rather than by index. See the function's docstring:
            # the two changes are inseparable, and applying only the first is worse than
            # applying neither. It is also where average-of-configuration lives — one
            # function, so a closed shell and an open shell take the same code path and the
            # open-shell machinery is exercised by every test in the suite.
            # ``shells`` is filled by ``get_occ`` each cycle and read by the shell-dependent
            # Fock installed below — the two are deliberately coupled, because which spinors
            # form each open shell is resolved in exactly one place (see the comment there).
            state = {}
            mf.get_occ = _average_of_configuration_occupation(xmol, config,
                                                             current_light_speed(), state)
            if not config.is_closed_shell:
                # ⚠ The occupations alone are not average of configuration — the *energy* has
                # to be too. See :func:`_install_configuration_average`; without it this is
                # fractional-occupation Hartree-Fock, whose open-shell energies were measured
                # 0.30-0.47 Eh above four-component DIRAC's. A closed shell installs
                # nothing and takes the identical path it always did.
                install_configuration_average(mf, xmol, state)
            if spherical:
                # ⚠ **And the occupations are not sphericity either**, which is the third
                # thing this SCF needs and the one that was missing. Occupying a whole ``l``
                # shell equally makes the density spherical *given* spherical orbitals; it
                # does not stop the iteration from finding the lower, symmetry-broken
                # solutions a fractionally occupied Hartree-Fock functional has. Projecting
                # the Fock onto its rank-zero part each cycle imposes the symmetry that
                # defines the state instead. Wrapped **outside** the effective Fock above, so
                # what the eigensolver sees — after the coupling operator, after DIIS — is
                # spherical.
                project = spherical_projector(spinor_symmetry_groups(xmol),
                                              2 * int(xmol.nao_2c()), blocks=2)
                inner_get_fock = mf.get_fock
                mf.get_fock = lambda *a, **k: project(inner_get_fock(*a, **k))
            else:
                log.warning(
                    "the four-component atomic SCF for %s is running WITHOUT the spherical "
                    "constraint. An open shell then converges to a symmetry-broken solution "
                    "whose mean field carries an arbitrary spatial orientation; this setting "
                    "exists to measure that and is never a production one.", element)
            e_tot = float(mf.kernel())

        if not mf.converged:
            log.error("four-component atomic SCF for %s did not converge in %d cycles "
                      "(E = %.8f Eh). Everything the atomic mean-field correction says about "
                      "this element rests on these orbitals; the solution is returned marked "
                      "converged=False so the failure is visible rather than swallowed.",
                      element, max_cycle, e_tot)

        u = spinor_to_spin_orbital(xmol)

        def to_spin_orbital(a) -> FourComponentBlocks:
            """Split a 4c matrix into blocks and take each into the spin-blocked spin-orbital basis."""
            b = FourComponentBlocks.from_matrix(np.asarray(a))
            return FourComponentBlocks(
                ll=np.ascontiguousarray(u @ b.ll @ u.conj().T),
                ls=np.ascontiguousarray(u @ b.ls @ u.conj().T),
                sl=np.ascontiguousarray(u @ b.sl @ u.conj().T),
                ss=np.ascontiguousarray(u @ b.ss @ u.conj().T))

        dm = mf.make_rdm1()
        blocks = {"hcore": to_spin_orbital(mf.get_hcore(xmol)),
                  "overlap": to_spin_orbital(mf.get_ovlp(xmol)),
                  "density": to_spin_orbital(dm),
                  "veff": to_spin_orbital(mf.get_veff(xmol, dm))}

        # Point 3 of the module docstring, checked on the *converged density* rather than on
        # the orbital energies — see density_anisotropy for why the obvious frontier-gap test
        # lets an open shell straight through. Under average-of-configuration this is the
        # assertion that the averaging did its job, not a closed-shell restriction.
        anisotropy = density_anisotropy(xmol, blocks["density"].ll)
        if anisotropy > SPHERICAL_DENSITY_TOLERANCE:
            raise RuntimeError(
                "the converged four-component density for {}{:+d} in configuration {} is "
                "anisotropic by {:.2e}, where an atom is spherical to about 1e-12. The mean "
                "field therefore carries an arbitrary spatial orientation, which would be "
                "baked into the correction and would look entirely reasonable downstream. "
                "For an open shell this means the fractional occupation did not land on the "
                "manifold the configuration names — check that the configuration is the "
                "aufbau one for this species, since a hole below an occupied shell of the "
                "same l cannot be represented (kuiva.amf.configuration).".format(
                    element, reference_charge, config.canonical, anisotropy))

        solution = AtomicDiracSolution(
            element=element, atomic_number=int(xmol.atom_charge(0)),
            charge=reference_charge,
            basis=_basis_label(basis, uncontract), basis_spec=basis,
            configuration=config,
            interaction=interaction, light_speed=current_light_speed(),
            hcore=blocks["hcore"], overlap=blocks["overlap"],
            density=blocks["density"], veff=blocks["veff"],
            contraction=contraction, uncontracted=bool(uncontract),
            mo_energy=np.ascontiguousarray(np.asarray(mf.mo_energy, dtype=float)),
            mo_occ=np.ascontiguousarray(np.asarray(mf.mo_occ, dtype=float)),
            e_tot=e_tot, converged=bool(mf.converged),
            backend=self.name, backend_version=self.version)
        log.debug("atomic 4c DHF %s (%s): E = %.10f Eh, nao %d -> %d, %s, %.4f GB resident",
                  element, config.canonical, e_tot, solution.nao, solution.nao_target,
                  interaction, reservation.gb)
        return solution

    def coulomb_mean_field(self, solution: AtomicDiracSolution,
                           dm: np.ndarray) -> np.ndarray:
        """Non-relativistic ``J - K`` of a two-component density in the solver's basis.

        This is the *untransformed* Coulomb mean field that the X2CAMF subtraction removes — the
        operator
        the molecular Hamiltonian already carries through ``mol.intor("int2e")``. It is built
        here, in the backend, rather than in :mod:`kuiva.amf.decouple`, because it needs the
        two-electron integrals over exactly the basis the four-component solution used: a
        separately constructed basis would make the subtraction a difference between two
        slightly different things, which is the failure mode :mod:`kuiva.amf.decouple` warns
        about.

        The ``Mole`` is rebuilt from ``solution.basis_spec``, deterministically and in
        milliseconds, so that :class:`AtomicDiracSolution` stays free of framework objects.
        """
        from pyscf import scf

        mol, _ = self._build_mole(solution.element, solution.basis_spec, solution.charge,
                                  solution.uncontracted)
        if int(mol.nao) != solution.nao:
            raise RuntimeError(
                "rebuilding the atomic basis for {} gave {} functions where the solution has "
                "{}. The subtracted mean field would then be over a different basis from the "
                "transformed one, which is exactly what must not happen.".format(
                    solution.element, mol.nao, solution.nao))
        dm = np.ascontiguousarray(dm, dtype=np.complex128)
        if dm.shape != (2 * solution.nao, 2 * solution.nao):
            raise ValueError("expected a ({0}, {0}) two-component density, got {1}".format(
                2 * solution.nao, dm.shape))
        # GHF's J/K handles the spin-blocked density correctly: J couples only the diagonal
        # spin blocks through the total density, K couples all four. Doing it by hand is a
        # standing invitation to drop the alpha-beta exchange block, which is exactly the part
        # that carries the spin-orbit screening.
        vj, vk = scf.GHF(mol).get_jk(mol, dm, hermi=1)
        return np.ascontiguousarray(np.asarray(vj) - np.asarray(vk))


__all__ = ["PySCFDiracBackend", "light_speed", "current_light_speed",
           "spinor_to_spin_orbital", "spinor_symmetry_groups", "eigh_canonical",
           "density_anisotropy", "METRIC_LINDEP_THRESHOLD",
           "SPHERICAL_DENSITY_TOLERANCE", "CLOSED_SHELL_ANISOTROPY", "CONTRACTION_TOL"]
