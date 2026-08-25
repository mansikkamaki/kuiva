"""Phase-invariant analysis of SOC multiplets and magnetic-moment matrices.

Motivation
----------
The phases of the property matrices Kuiva dumps are **arbitrary** — they are
fixed downstream by the external ITO/crystal-field code. Worse, within a *degenerate* group
of SOC eigenstates the eigenvectors are defined only up to an arbitrary unitary mixing. Any
element-by-element comparison of magnetic-moment matrices — against a previous Kuiva run,
against OpenMolcas/DIRAC, or against analytic theory — is therefore **meaningless**.

This module supplies the quantities that *are* well defined: functions of the moment matrices
that are invariant under (a) arbitrary phases and (b) arbitrary unitary rotation inside a
degenerate block. These are what the Tier-1/Tier-2 test suites compare, and what any future
validation of ``props/dump.py`` output must use.

The central invariant
---------------------
For a degenerate block ``b`` of dimension ``d``, form the real symmetric 3x3 tensor

    M_ij = Tr_b( mu_i mu_j )        [mu_B^2],   i, j in {x, y, z}

Under ``mu -> U^dag mu U`` with ``U`` unitary on the block, the trace is invariant, so ``M``
and its eigenvalues are invariant. ``M`` is the block analogue of the g-tensor:

* **Kramers doublet** (``d = 2``). With the pseudospin convention of Abragam & Bleaney,
  ``mu_i = -(1/2) mu_B sum_k g_ik sigma_k``, so ``Tr(mu_i mu_j) = (1/2) mu_B^2 (g g^T)_ij``
  and the principal g values are ``sqrt(eig(2 M / mu_B^2))``.
* **Free-ion J multiplet** (``d = 2J+1``, ``mu = -g_J mu_B J``). Then
  ``M_ij = delta_ij g_J^2 J(J+1)(2J+1)/3``, so the Lande factor follows as
  ``g_J = sqrt(3 M_ii / (J(J+1)(2J+1)))``.

Both are the same formula, :func:`multiplet_g_values`, with ``J = (d-1)/2``. That the two
coincide is not a coincidence: a Kramers doublet is the ``J = 1/2`` case. This gives every
free-ion test system an **analytic** target that no program's conventions can affect
(Ce(3+) ``2F5/2``: g = 6/7; Yb(3+) ``2F7/2``: g = 8/7; Dy(3+) ``6H15/2``: g = 4/3).

⚠ The non-Kramers pseudo-doublet, and why it needs saying
----------------------------------------------------------
An **integer**-spin ion — Tb(3+) ``7F6``, Ho(3+) ``5I8``, the Ln SMMs this program exists for
— has no Kramers protection. Its ground "doublet" is two *singlets* split by a tunnelling gap
``Delta`` that is anything from 1e-5 to tens of wavenumbers, so the two states do not group at
the default tolerance and each arrives here as a block of size 1. A size-1 block has
``J = 0`` and therefore no magnetic moment at all — which is the honest answer for a genuine
``J = 0`` level and a badly misleading one for half of a tunnelling-split pair.

Two things follow, and they are the whole of the non-Kramers handling here:

* ⚠ **A size-1 block returns no g values — an empty tuple — never ``(0, 0, 0)``.** Zero is a
  measurement; the absence of one is not, and the two must not print the same. The spectrum
  additionally **warns once** when two singlets sit within :data:`PSEUDO_DOUBLET_HINT_CM` of
  each other, naming ``Delta`` and the knob, because that is the situation a reader has to
  decide about rather than have decided for them.
* **Grouping the pair is opt-in**, through ``pseudo_doublet_tol_cm`` on
  :func:`analyse_spectrum`. It is a request, never inferred: whether two nearby singlets are
  one pseudo-doublet or two crystal-field levels is physics the energies alone cannot settle.

For a grouped pair the presentation follows the standard non-Kramers convention (Griffith;
Abragam & Bleaney ch. 3, 18): the doublet is described by an effective spin ``S~ = 1/2`` whose
**transverse components vanish identically**, ``g_x = g_y = 0``, with only ``g_z`` and the
off-diagonal ``Delta`` surviving. So :attr:`Multiplet.g_z` is the number to quote and
:attr:`Multiplet.tunnelling_gap_cm` goes beside it.

⚠ **The two transverse principal values are still computed and kept, as a residual that can
fail.** For a true non-Kramers doublet they are zero by symmetry; a large one is evidence that
these two states are *not* a pseudo-doublet and that the grouping request was wrong. A
diagnostic whose only possible value is the one you assumed is not a diagnostic.

References
----------
* Pseudospin Hamiltonians and the g-tensor from ab initio SOC states: L. F. Chibotaru,
  L. Ungur, J. Chem. Phys. 137, 064112 (2012), doi:10.1063/1.4739763. The ``g g^T``
  construction used in :func:`multiplet_g_values` follows this work (and it is the same
  quantity OpenMolcas's SINGLE_ANISO reports).
* Pseudospin / g-tensor conventions for Kramers doublets: A. Abragam, B. Bleaney,
  "Electron Paramagnetic Resonance of Transition Ions", Clarendon Press, Oxford (1970).
* The **non-Kramers** doublet, its tunnelling splitting and the effective-spin form in which
  the transverse g components vanish identically (``g_x = g_y = 0``, only ``g_z`` and
  ``Delta`` survive), which is the convention :attr:`Multiplet.g_z` reports:
  J. S. Griffith, Phys. Rev. 132, 316 (1963), doi:10.1103/PhysRev.132.316; A. Abragam,
  B. Bleaney, ibid., ch. 3.11 and 18.3.
* Lande g factor and free-ion multiplets: standard atomic theory, e.g. R. D. Cowan,
  "The Theory of Atomic Structure and Spectra", Univ. California Press (1981), ch. 11.
* Free electron g factor: CODATA 2018 recommended values, doi:10.1103/RevModPhys.93.025010.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..util.logging import get_logger

log = get_logger(__name__)

#: Free-electron g factor (CODATA 2018). Used in mu = -(L + g_e S) mu_B.
G_ELECTRON = 2.00231930436256

#: Hartree -> wavenumber conversion (CODATA 2018).
HARTREE_TO_CM = 219474.6313632

#: ⚠ **Advisory only, and not a physical tolerance.** Two singlets closer than this warn that
#: they *may* be a tunnelling-split non-Kramers pseudo-doublet (module docstring). It states
#: "close enough that you have to decide", nothing more: nothing is grouped, no number moves,
#: and the decision stays the user's through ``pseudo_doublet_tol_cm``. The value is generous
#: on purpose — a missed warning is a silent wrong answer, a spurious one costs a glance.
PSEUDO_DOUBLET_HINT_CM = 100.0

#: ⚠ **Advisory, and deliberately loose.** Relative separation of two principal g values below
#: which their axes are treated as spanning a plane rather than naming directions
#: (:func:`axis_is_defined`). It is a statement about what may be *quoted*, not a numerical
#: tolerance: a free ion's `j = 1/2` doublet is isotropic by symmetry and still comes out of a
#: real calculation with its top two g values differing by ~1e-3 of themselves — basis and
#: picture-change anisotropy, not physics. Calling that an easy axis would name a direction
#: that is an artefact of the last few digits, and print it beside an axiality of 1.00.
#:
#: ⚠ **Not the same number as** :data:`kuiva.props.pseudospin.AXIS_DEGENERACY_RTOL`, and the
#: difference is intentional: that one guards a *labelling* axis, where any in-plane choice is
#: equally valid and self-consistency is all that is required, so a tight threshold costs
#: nothing. This one gates a claim someone will put in a paper.
AXIS_DEFINED_RTOL = 1.0e-2


@dataclass(frozen=True)
class Multiplet:
    """One group of (near-)degenerate SOC eigenstates and its moment invariants.

    Attributes
    ----------
    start, size : int
        Index of the first state in the block and the block dimension.
    energy_cm : float
        Block energy relative to the ground state [cm^-1] (mean over the block).
    spread_cm : float
        Max-min energy inside the block [cm^-1]; a measure of how exactly degenerate it is.
    m_tensor : np.ndarray, shape (3, 3)
        The invariant ``M_ij = Tr_b(mu_i mu_j)`` in mu_B^2. ``None`` if no moments given.
    g_values : tuple of float
        Principal g values from :func:`multiplet_g_values` (ascending), or ``()`` — which is
        also what a ``size = 1`` block returns, since it carries no moment. ⚠ Never
        ``(0, 0, 0)``: a measured zero and the absence of a measurement must not print alike.
    g_axes : np.ndarray, shape (3, 3)
        Principal magnetic axes as **columns**, column ``k`` belonging to ``g_values[k]`` — so
        the last column is the easy axis of an axial block. ``None`` if no moments were given.
        ⚠ Meaningful only where the corresponding g value is non-degenerate; see
        :meth:`easy_axis_is_defined`.
    g_sign : float
        Sign of ``det(g)`` for a Kramers doublet (:func:`g_determinant_sign`), ``0.0`` where
        it is not defined. ⚠ A property of the states, not a convention — and the part
        :attr:`m_tensor` cannot carry, being quadratic in ``mu``.
    non_kramers : bool
        This block was assembled from two singlets by an explicit ``pseudo_doublet_tol_cm``
        request (module docstring). ⚠ Read :attr:`g_z`, not :attr:`g_values`.
    tunnelling_gap_cm : float or None
        ``Delta``, the splitting of the two singlets that were grouped [cm^-1]. ``None`` for
        every ordinary block.
    d_tensor : np.ndarray, shape (3, 3)
        The invariant ``D_ij = Tr_b(d_i d_j)`` in ``(e a_0)^2`` (:func:`block_dipole_tensor`),
        or ``None`` if no dipole matrices were given. ⚠ It is the block's *internal* electric
        second moment and says nothing about a transition **out** of the block — that is
        :func:`block_line_strengths` — and for a charged molecule it moves with the gauge
        origin.
    """
    start: int
    size: int
    energy_cm: float
    spread_cm: float
    m_tensor: Optional[np.ndarray] = None
    g_values: Tuple[float, ...] = ()
    g_axes: Optional[np.ndarray] = None
    g_sign: float = 0.0
    non_kramers: bool = False
    tunnelling_gap_cm: Optional[float] = None
    d_tensor: Optional[np.ndarray] = None

    @property
    def j(self) -> float:
        """Effective angular momentum implied by the block dimension, ``J = (size-1)/2``."""
        return (self.size - 1) / 2.0

    @property
    def g_iso(self) -> float:
        """Isotropic average of the principal g values."""
        return float(np.mean(self.g_values)) if self.g_values else float("nan")

    @property
    def g_z(self) -> float:
        """The axial g of a non-Kramers pseudo-doublet — the only component that survives.

        In the Griffith / Abragam-Bleaney description of a non-Kramers doublet the transverse
        components vanish identically, so the largest principal value of ``M`` *is* ``g_z``
        and the other two are the residual of :attr:`g_transverse_residual`. Reported instead
        of :attr:`g_values` for such a block, and ``nan`` for any other.
        """
        if not self.non_kramers or not self.g_values:
            return float("nan")
        return float(max(self.g_values))

    @property
    def g_transverse_residual(self) -> float:
        """Largest transverse principal value of a non-Kramers block — **zero by symmetry**.

        ⚠ The check that can fail. For a true non-Kramers pseudo-doublet this is zero; a value
        comparable to :attr:`g_z` says the two grouped singlets are not a pseudo-doublet and
        the ``pseudo_doublet_tol_cm`` request was wrong. ``nan`` for any other block.
        """
        if not self.non_kramers or len(self.g_values) < 2:
            return float("nan")
        return float(sorted(self.g_values)[-2])

    @property
    def easy_axis(self) -> Optional[np.ndarray]:
        """The axis of the largest principal g value — the easy axis of an axial block.

        ⚠ Check :meth:`easy_axis_is_defined` first. An easy-plane or isotropic block has no
        easy *axis*, and what comes back for one is a vector the eigensolver happened to
        choose inside a degenerate plane.
        """
        return None if self.g_axes is None else np.asarray(self.g_axes)[:, -1]

    def easy_axis_is_defined(self, rtol: float = AXIS_DEFINED_RTOL) -> bool:
        """Whether :attr:`easy_axis` is a direction rather than a choice within a plane."""
        return bool(self.g_axes is not None and axis_is_defined(self.g_values, -1, rtol))

    @property
    def axiality(self) -> float:
        """``g_max / g_perp`` with ``g_perp`` the mean of the other two — how axial the block is.

        A single number for "is this an Ising-like doublet or an isotropic one", which is what
        the axes are usually consulted for. ``inf`` for a perfectly axial block, ``1`` for an
        isotropic one, ``nan`` where there are no g values.
        """
        if len(self.g_values) < 3:
            return float("nan")
        g = sorted(self.g_values)
        perp = 0.5 * (g[0] + g[1])
        return float("inf") if perp <= 0.0 else float(g[2] / perp)


def degenerate_blocks(energies_cm: Sequence[float], tol_cm: float = 1.0) -> List[Tuple[int, int]]:
    """Group sorted state energies into degenerate blocks.

    Parameters
    ----------
    energies_cm : sequence of float
        State energies [cm^-1], assumed ascending (they are sorted defensively).
    tol_cm : float
        Two consecutive states belong to the same block if they differ by less than this.
        The default of 1 cm^-1 is far above the numerical noise of a converged CI
        (~1e-3 cm^-1) and far below any physical splitting we test against.

    Returns
    -------
    list of (start, size)
    """
    e = np.sort(np.asarray(energies_cm, dtype=float))
    if e.size == 0:
        return []
    blocks: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, e.size):
        if e[i] - e[i - 1] > tol_cm:
            blocks.append((start, i - start))
            start = i
    blocks.append((start, e.size - start))
    return blocks


def magnetic_moment_matrices(l_matrices: np.ndarray, s_matrices: np.ndarray,
                             g_e: float = G_ELECTRON) -> np.ndarray:
    """Magnetic-moment operator matrices ``mu = -(L + g_e S) mu_B`` in the SOC eigenbasis.

    Parameters
    ----------
    l_matrices, s_matrices : np.ndarray, shape (3, n, n)
        Orbital-angular-momentum and spin matrices in the basis of the SOC eigenstates
        (complex, Hermitian), in units of hbar.

    Returns
    -------
    np.ndarray, shape (3, n, n), complex — moments in Bohr magnetons.

    Notes
    -----
    The overall sign convention is irrelevant to every invariant in this module (they are
    all quadratic in ``mu``), which is precisely why they survive the dump's arbitrary phases.
    """
    l = np.asarray(l_matrices)
    s = np.asarray(s_matrices)
    if l.shape != s.shape or l.ndim != 3 or l.shape[0] != 3:
        raise ValueError(f"expected two (3, n, n) arrays, got {l.shape} and {s.shape}")
    return -(l + g_e * s)


def block_operator_tensor(op: np.ndarray, start: int, size: int) -> np.ndarray:
    """The invariant ``T_ij = Tr_b(A_i A_j)`` over one degenerate block, for any Hermitian
    vector operator ``A`` given as ``(3, n, n)``.

    ⚠ **One implementation, because the invariance argument is one argument.** A block trace of
    a product of two Hermitian matrices is invariant under any unitary mixing within the block
    and under any per-state phase; that is what makes it the only sound way to compare a stored
    property matrix with anything, and it holds for the electric dipole exactly as it holds for
    the magnetic moment. :func:`block_moment_tensor` and :func:`block_dipole_tensor` are the
    two named readings of it, and they differ only in units.
    """
    sl = slice(start, start + size)
    blk = np.asarray(op)[:, sl, sl]
    m = np.einsum("iab,jba->ij", blk, blk)
    # Tr(A_i A_j) is real for Hermitian A_i; symmetrise to kill numerical asymmetry.
    m = np.real(m)
    return 0.5 * (m + m.T)


def block_moment_tensor(mu: np.ndarray, start: int, size: int) -> np.ndarray:
    """The invariant ``M_ij = Tr_b(mu_i mu_j)`` [mu_B^2] over one degenerate block.

    Invariant under any unitary mixing within the block and under any per-state phase, so
    this is the only sound way to compare moment matrices between codes.
    """
    return block_operator_tensor(mu, start, size)


def block_dipole_tensor(d: np.ndarray, start: int, size: int) -> np.ndarray:
    """The invariant ``D_ij = Tr_b(d_i d_j)`` [(e a_0)^2] over one degenerate block.

    The electric counterpart of :func:`block_moment_tensor`, and the reduction any validation
    of the dump's ``d`` matrices must go through: the file fixes no phase convention and
    degenerate states mix arbitrarily, so an element-by-element comparison of ``d`` compares
    arbitrary phases and nothing else.

    ⚠ **For a charged molecule this quantity moves with the gauge origin**, because the block's
    internal elements include the diagonal and the diagonal carries ``-q R_G``. Compare two
    charged systems' ``D`` only at the same origin — which the dump header states.
    ⚠ **It is not an oscillator strength and must not be read as one.** It is a trace over one
    block; what a transition needs is :func:`block_line_strengths`, between two of them.
    """
    return block_operator_tensor(d, start, size)


def block_line_strengths(d: np.ndarray, blocks: Sequence[Tuple[int, int]]) -> np.ndarray:
    """``S_AB = sum_k sum_{I in A, J in B} |d_k[I,J]|^2`` [(e a_0)^2], as ``(n_b, n_b)`` real.

    The phase-invariant statement about a *transition*: a double sum of squared moduli over two
    whole blocks is invariant under any unitary mixing inside either block and under any
    per-state phase, which is exactly what a degenerate manifold leaves undetermined. This is
    what a selection rule is checked against — a forbidden transition has ``S_AB = 0`` however
    the eigensolver happened to rotate the two manifolds — and what a cross-code comparison of
    transition dipoles compares.

    ⚠ **This is a line strength and NOT an oscillator strength, a rate, or an intensity.**
    Turning it into one requires the transition energy, the refractive index and a convention
    for what is being measured; that analysis belongs to the external property code, as the
    ITO and crystal-field analysis does. Nothing here divides, weights or degeneracy-averages.

    ⚠ **Off-diagonal blocks (``A != B``) are origin-independent whatever the molecule's
    charge**; the diagonal ``S_AA`` is not, for the reason
    :func:`block_dipole_tensor` gives.
    """
    d = np.asarray(d)
    if d.ndim != 3 or d.shape[0] != 3:
        raise ValueError("the dipole matrices must be (3, n, n), got {}".format(d.shape))
    n = len(blocks)
    out = np.zeros((n, n))
    for a, (sa, na) in enumerate(blocks):
        for b, (sb, nb) in enumerate(blocks):
            block = d[:, sa:sa + na, sb:sb + nb]
            out[a, b] = float(np.sum(np.abs(block) ** 2))
    return out


def multiplet_g_axes(m_tensor: np.ndarray) -> np.ndarray:
    """Principal magnetic axes of a block: the ``(3, 3)`` eigenvectors of ``M``, as columns.

    Column ``k`` is the axis belonging to :func:`multiplet_g_values`'s ``k``-th value, so the
    two are read together and the last column is the **easy axis** of an axial system — the
    quantity a crystal-field analysis is usually after and the one this module used to compute
    and throw away.

    ⚠ **An axis is defined only up to its sign, and only when its g value is
    non-degenerate.** The sign is fixed here by making the largest-magnitude component
    positive, which is a convention and nothing more (a magnetic axis is a *line*). Where two
    principal values coincide — an isotropic or easy-plane block — their axes span a plane and
    any pair in it is as good: the eigenvector routine returns one such pair and it means
    nothing on its own. :func:`axis_is_defined` is the test, and it is worth applying before
    quoting a direction.

    Returned as a **proper** triad (``det = +1``), so it is a rotation into the principal
    frame rather than a rotation with a reflection hidden in it.
    """
    m = np.asarray(m_tensor, dtype=float)
    _, vectors = np.linalg.eigh(0.5 * (m + m.T))
    vectors = np.array(vectors, dtype=float)
    for k in range(3):                       # a magnetic axis is a line; fix a sign for print
        j = int(np.argmax(np.abs(vectors[:, k])))
        if vectors[j, k] < 0.0:
            vectors[:, k] = -vectors[:, k]
    if np.linalg.det(vectors) < 0.0:         # keep it a rotation, not a rotoreflection
        vectors[:, 0] = -vectors[:, 0]
    return vectors


def axis_is_defined(g_values: Sequence[float], k: int = -1,
                    rtol: float = AXIS_DEFINED_RTOL) -> bool:
    """Whether principal axis ``k`` is a direction at all, rather than a choice in a plane.

    ⚠ The check that stops an arbitrary vector being quoted as an easy axis. Two coincident
    principal values leave their axes spanning a plane, and the eigensolver then returns some
    pair from it — self-consistent, reproducible, and physically meaningless as a direction.
    Default ``k = -1``: the easy axis of an axial block.

    ⚠ **The tolerance is loose on purpose** (:data:`AXIS_DEFINED_RTOL`): the degeneracy that
    matters here is the *physical* one, and a symmetry-isotropic doublet still comes out of a
    real calculation anisotropic in its last few digits. See that constant.
    """
    g = np.sort(np.asarray(g_values, dtype=float))
    if g.size < 2:
        return False
    scale = float(np.max(np.abs(g)))
    if scale <= 0.0:
        return False
    k = k % g.size
    gaps = []
    if k > 0:
        gaps.append(g[k] - g[k - 1])
    if k < g.size - 1:
        gaps.append(g[k + 1] - g[k])
    return bool(min(gaps) > rtol * scale)


def g_determinant_sign(mu: np.ndarray, start: int, size: int) -> float:
    """Sign of ``det(g)`` for a Kramers doublet — the part ``M`` throws away.

    ``M = Tr_b(mu_i mu_j)`` is quadratic in ``mu``, so it fixes ``|det g|`` and loses the
    sign. The sign is nevertheless a property of the states and not a convention, and it is
    recoverable from a **third-order** invariant. Writing
    ``mu_i = -(1/2) mu_B sum_k g_ik sigma_k`` on the doublet and using
    ``Tr(sigma_a [sigma_b, sigma_c]) = 4i eps_abc``,

    ::

        Tr( mu_x [mu_y, mu_z] ) = -(i/2) mu_B^3 det(g),

    so ``det(g) = 2i Tr(mu_x [mu_y, mu_z]) / mu_B^3``. It is a trace of block-restricted
    operators, hence **invariant under any unitary mixing inside the block and under every
    per-state phase** — the same footing as ``M`` itself, which is what makes it quotable
    across programs at all (module docstring; Chibotaru & Ungur 2012).

    Returns ``+1.0`` or ``-1.0``, or ``0.0`` — never a guess. Three cases give ``0.0``:

    * ⚠ **the block is not a doublet.** For a larger multiplet the three-fold product is not
      the determinant of anything, and a number here would look like one.
    * the block carries no moment at all.
    * ⚠ **``det(g)`` is numerically zero**, which is the case a strongly axial doublet
      actually presents: with two principal g values at ~1e-6 the product is ~1e-12 against an
      ``O(1)`` scale, and its *sign* is then a property of the rounding rather than of the
      states. Reporting one would be inventing information. Observed on the TiCl3 doublets of
      example 6, whose axialities reach 1e6.
    """
    if int(size) != 2:
        return 0.0
    sl = slice(start, start + size)
    blk = np.asarray(mu)[:, sl, sl]
    x, y, z = blk[0], blk[1], blk[2]
    triple = np.trace(x @ (y @ z - z @ y))
    # Purely imaginary for Hermitian mu; the real part is rounding and is discarded.
    det_g = float(np.real(2.0j * triple))
    scale = float(np.max(np.abs(blk))) ** 3
    if scale <= 0.0 or abs(det_g) <= 1.0e-10 * scale:
        return 0.0
    return 1.0 if det_g > 0.0 else -1.0


def multiplet_g_values(m_tensor: np.ndarray, size: int) -> Tuple[float, ...]:
    """Principal g values of a degenerate block from its moment tensor.

    Uses ``g_k = sqrt(3 * eig_k(M) / (J(J+1)(2J+1)))`` with ``J = (size-1)/2``, which is the
    Chibotaru-Ungur ``g g^T`` construction for a Kramers doublet (``size = 2``) and the Lande
    factor for a free-ion ``2J+1`` multiplet. See the module docstring for the derivation.

    Returns the three principal values in ascending order — or an **empty tuple** for a
    ``size = 1`` block, which has ``J = 0`` and carries no magnetic moment.

    ⚠ **The empty tuple is the point, and it used to be ``(0.0, 0.0, 0.0)``.** A measured zero
    and "this quantity is not defined here" are different statements and must not print the
    same. They coincide for a genuine ``J = 0`` level and diverge completely for the case that
    matters: half of a tunnelling-split **non-Kramers** pseudo-doublet, where the pair carries
    a large axial moment and each singlet on its own carries none (module docstring). Silently
    reporting ``g = 0`` for a Tb/Ho-type ground state is a plausible wrong answer, which is
    what this project refuses to emit. ``()`` is already what a block analysed without moment
    matrices returns, so no consumer learns a new value.
    """
    j = (size - 1) / 2.0
    norm = j * (j + 1.0) * (2.0 * j + 1.0)
    if norm <= 0.0:
        return ()
    eigs = np.linalg.eigvalsh(np.asarray(m_tensor, dtype=float))
    # Tiny negative eigenvalues can appear from rounding on an (almost) zero tensor.
    eigs = np.clip(eigs, 0.0, None)
    return tuple(float(x) for x in np.sqrt(3.0 * eigs / norm))


def _pair_singlets(blocks: List[Tuple[int, int]], e_cm: np.ndarray,
                   tol_cm: float) -> List[Tuple[int, int, Optional[float]]]:
    """Merge adjacent size-1 blocks lying within ``tol_cm`` into non-Kramers pseudo-doublets.

    Returns ``(start, size, tunnelling_gap_cm)`` triples; the gap is ``None`` for everything
    that was not merged. Only *adjacent, both size-1* blocks are candidates, and a singlet
    already consumed by one pair cannot join a second — three near-equal singlets are not a
    doublet and a half-open greedy pass would silently make two of them into one.
    """
    out: List[Tuple[int, int, Optional[float]]] = []
    i = 0
    while i < len(blocks):
        start, size = blocks[i]
        if size == 1 and i + 1 < len(blocks) and blocks[i + 1][1] == 1:
            gap = float(e_cm[blocks[i + 1][0]] - e_cm[start])
            if gap <= tol_cm:
                out.append((start, 2, gap))
                i += 2
                continue
        out.append((start, size, None))
        i += 1
    return out


def _warn_on_close_singlets(blocks: List[Tuple[int, int]], e_cm: np.ndarray) -> None:
    """⚠ Warn once when two singlets sit close enough to be a tunnelling-split pair.

    Advisory and nothing else: nothing is grouped and no number moves. It exists because the
    alternative — reporting each singlet's ``g`` as absent, with no hint why two of them are
    almost on top of each other — leaves the reader of a Tb/Ho spectrum with no signal at all
    that the interesting object is the *pair*. Once per spectrum, naming the closest pair.
    """
    pairs = [(i, float(e_cm[blocks[i + 1][0]] - e_cm[blocks[i][0]]))
             for i in range(len(blocks) - 1)
             if blocks[i][1] == 1 and blocks[i + 1][1] == 1
             and e_cm[blocks[i + 1][0]] - e_cm[blocks[i][0]] <= PSEUDO_DOUBLET_HINT_CM]
    if not pairs:
        return
    i, gap = min(pairs, key=lambda p: p[1])
    log.warning(
        "states %d and %d are singlets %.4g cm^-1 apart, and %d such pair(s) are within "
        "%.0f cm^-1: on an integer-spin (non-Kramers) ion this is a tunnelling-split "
        "pseudo-doublet, whose moment belongs to the PAIR -- each singlet alone carries none, "
        "which is why their g values are reported as undefined rather than as zero. Pass "
        "pseudo_doublet_tol_cm= to group them and get g_z beside the gap",
        blocks[i][0], blocks[i + 1][0], gap, len(pairs), PSEUDO_DOUBLET_HINT_CM)


def analyse_spectrum(energies_hartree: Sequence[float],
                     mu: Optional[np.ndarray] = None,
                     tol_cm: float = 1.0,
                     pseudo_doublet_tol_cm: Optional[float] = None,
                     d: Optional[np.ndarray] = None) -> List[Multiplet]:
    """Full phase-invariant description of a SOC spectrum: blocks + moment invariants.

    This is the canonical reduction applied to Kuiva's own output and to the OpenMolcas /
    DIRAC reference data before any comparison is made.

    Parameters
    ----------
    energies_hartree : sequence of float
        SOC eigenstate energies [Eh] (any origin; they are shifted to the ground state).
    mu : np.ndarray, shape (3, n, n), optional
        Magnetic-moment matrices in the same (energy-ordered) basis, in mu_B.
    tol_cm : float
        Degeneracy tolerance, see :func:`degenerate_blocks`.
    pseudo_doublet_tol_cm : float, optional
        ⚠ **Opt-in, and a physical claim rather than a tolerance.** Adjacent *singlets* closer
        than this are grouped into one **non-Kramers pseudo-doublet** block carrying
        :attr:`Multiplet.g_z` and :attr:`Multiplet.tunnelling_gap_cm` (module docstring). Off
        by default and never inferred: whether two nearby singlets are one tunnelling-split
        doublet or two crystal-field levels is not something their energies can settle, and a
        wrong grouping produces a plausible g out of two unrelated states. Check
        :attr:`Multiplet.g_transverse_residual` on the result — it is zero by symmetry for a
        real pseudo-doublet, and that is the part of this that can fail.
    d : np.ndarray, shape (3, n, n), optional
        Electric dipole matrices in the same basis [e a_0]. Fills :attr:`Multiplet.d_tensor`
        and nothing else; the blocks and the grouping are decided by the energies exactly as
        before, so passing this changes no existing number.
    """
    e = np.asarray(energies_hartree, dtype=float)
    order = np.argsort(e)
    e_cm = (e[order] - e[order][0]) * HARTREE_TO_CM
    mu_sorted = None
    if mu is not None:
        mu_sorted = np.asarray(mu)[:, order, :][:, :, order]
    d_sorted = None
    if d is not None:
        d_sorted = np.asarray(d)[:, order, :][:, :, order]

    blocks = degenerate_blocks(e_cm, tol_cm=tol_cm)
    if pseudo_doublet_tol_cm is None:
        grouped = [(s, n, None) for s, n in blocks]
        _warn_on_close_singlets(blocks, e_cm)
    else:
        grouped = _pair_singlets(blocks, e_cm, float(pseudo_doublet_tol_cm))

    out: List[Multiplet] = []
    for start, size, gap in grouped:
        blk_e = e_cm[start:start + size]
        m_tensor = None
        g_vals: Tuple[float, ...] = ()
        g_axes = None
        g_sign = 0.0
        if mu_sorted is not None:
            m_tensor = block_moment_tensor(mu_sorted, start, size)
            g_vals = multiplet_g_values(m_tensor, size)
            if g_vals:
                g_axes = multiplet_g_axes(m_tensor)
                g_sign = g_determinant_sign(mu_sorted, start, size)
        d_tensor = (None if d_sorted is None
                    else block_dipole_tensor(d_sorted, start, size))
        out.append(Multiplet(start=start, size=size,
                             energy_cm=float(np.mean(blk_e)),
                             spread_cm=float(blk_e.max() - blk_e.min()),
                             m_tensor=m_tensor, g_values=g_vals,
                             g_axes=g_axes, g_sign=g_sign,
                             non_kramers=gap is not None, tunnelling_gap_cm=gap,
                             d_tensor=d_tensor))
    log.debug("analysed SOC spectrum: %d states -> %d multiplets (tol=%.3g cm-1, %d "
              "non-Kramers pair(s))", e.size, len(out), tol_cm,
              sum(1 for m in out if m.non_kramers))
    return out


def spectrum_line_strengths(energies_hartree: Sequence[float], d: np.ndarray,
                            multiplets: Sequence[Multiplet]) -> np.ndarray:
    """``S_AB`` over the blocks :func:`analyse_spectrum` found, ``(n_b, n_b)`` [(e a_0)^2].

    ⚠ **``multiplets`` must be the result of :func:`analyse_spectrum` on the same energies**,
    because its ``start``/``size`` are indices into the *energy-sorted* basis and the dipole
    matrices arrive in the caller's order. The one line of sorting below is that convention
    restated, and it is why this takes the multiplets rather than re-grouping: two independent
    groupings of one spectrum would eventually disagree at a tolerance boundary and the mismatch
    would be silent.
    """
    order = np.argsort(np.asarray(energies_hartree, dtype=float))
    d_sorted = np.asarray(d)[:, order, :][:, :, order]
    return block_line_strengths(d_sorted, [(m.start, m.size) for m in multiplets])


def lande_g(l: float, s: float, j: float) -> float:
    """Analytic Lande g factor of a free-ion ``(L, S) J`` level (g_e taken as exactly 2).

    ``g_J = 3/2 + [S(S+1) - L(L+1)] / [2 J(J+1)]``. Used to give the free-ion test systems an
    implementation-independent target (Ce(3+) 2F5/2 -> 6/7, Dy(3+) 6H15/2 -> 4/3).
    """
    if j <= 0.0:
        return 0.0
    return 1.5 + (s * (s + 1.0) - l * (l + 1.0)) / (2.0 * j * (j + 1.0))


def degeneracy_pattern(multiplets: Sequence[Multiplet]) -> Tuple[int, ...]:
    """Block sizes in energy order — the coarsest, most robust cross-code fingerprint.

    Degeneracies are dictated by symmetry, so they must match *exactly* between codes even
    when energies differ by percent (e.g. Ce(3+) 4f^1 must give ``(6, 8)``).
    """
    return tuple(m.size for m in multiplets)


__all__ = [
    "G_ELECTRON", "HARTREE_TO_CM", "Multiplet", "analyse_spectrum", "block_moment_tensor",
    "block_operator_tensor", "block_dipole_tensor", "block_line_strengths",
    "spectrum_line_strengths",
    "degeneracy_pattern", "degenerate_blocks", "lande_g", "magnetic_moment_matrices",
    "multiplet_g_values",
    "AXIS_DEFINED_RTOL", "PSEUDO_DOUBLET_HINT_CM", "axis_is_defined", "g_determinant_sign", "multiplet_g_axes",
]
