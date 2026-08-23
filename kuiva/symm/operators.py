"""Representation matrices of the spatial operations, and the labelling of orbitals by them.

Why the matrices are built here rather than asked for
-----------------------------------------------------
⚠ **PySCF's own symmetry machinery reorients the molecule.** Switching ``mol.symmetry`` on
moves the geometry to a standard frame, which silently moves the gauge origin and with it
every angular-momentum and magnetic-moment operator that was fixed at ingestion — the
property matrices are then in a frame nothing else in the run knows about, and no error
message appears anywhere. So the operator matrices are built here, in the **input frame**,
from the AO layout the front end already carries, and PySCF's ``orbsym`` is used only as an
independent cross-check in the test suite.

The consequence a user has to know is stated at the point of selection: **the operations are
tested in the frame the geometry was given in**. A molecule whose two-fold axis is not ``z``
detects a smaller group than it has, and the fix is to orient the input — not to let the
program move it.

What an operation does to an AO
-------------------------------
Every operation of the ``D2h`` chain maps a real solid-harmonic basis function to
``+/-`` itself on a (possibly different) atom, so ``U(g)`` factorizes exactly into an atom
permutation and a per-AO sign, with no dense matrix and no rotation matrix anywhere:

===============  ======================  ==============================
operation        coordinates             sign on a real ``Y_lm``
===============  ======================  ==============================
``i``            ``r -> -r``             ``(-1)^l``
``C2(z)``        ``(x,y,z)->(-x,-y,z)``  ``(-1)^|m|``
``sigma(xy)``    ``(x,y,z)->(x,y,-z)``   ``(-1)^(l+|m|)``
===============  ======================  ==============================

with the real-harmonic convention ``m > 0 <-> cos(|m| phi)``, ``m < 0 <-> sin(|m| phi)`` that
:mod:`kuiva.basis.layout` records in ``ao_m`` (and whose ``l = 1`` special case — the
integral library lays a p shell out as ``px, py, pz``, i.e. ``m = +1, -1, 0`` — is handled
there and consumed here, not re-derived).

⚠ **Degenerate orbitals are symmetry-adapted, not refused.** Within an abelian group no
degeneracy is required by symmetry, so a degenerate pair in the scalar SCF is either
accidental or a partner pair of a *larger* group the abelian one cannot see. Either way the
SCF is free to return an arbitrary mixture of the two, and the mixture is an eigenvector of
nothing. The fix is a rotation **inside the degenerate block**, which changes no density, no
energy and no observable, and it is applied here before the labels are read off; a residual
that survives it is refused, with the orbital and the operation named. The alternative —
refusing every degenerate block outright — would refuse most real molecules for a reason that
is not a defect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util.degeneracy import DEFAULT_GROUP_RTOL
from ..util.logging import get_logger
from .groups import SPATIAL_NAMES, Generator, Group

log = get_logger(__name__)

#: Distance below which a rotated/reflected nucleus is taken to be the image of another one,
#: in bohr. Loose enough for a geometry quoted to five decimals in Angstrom, tight enough that
#: two genuinely distinct atoms never merge.
DEFAULT_ATOM_TOL = 1.0e-4

#: How far an orbital may be from an eigenvector of an operation before the labelling is
#: refused, measured as the largest off-diagonal element of ``C^T S U(g) C`` after the
#: degenerate blocks have been adapted. Well above the ``1e-13``-ish residual a converged SCF
#: leaves and far below anything a broken symmetry produces.
DEFAULT_LABEL_TOL = 1.0e-6

#: The spin-1/2 factor ``D^(1/2)(g)`` of each spatial operation, in the ``(alpha, beta)``
#: basis. Only the ``z`` rotation acts (``exp(-i pi sigma_z / 2) = diag(-i, +i)``); inversion
#: does not touch spin, and ``sigma(xy) = i C2(z)`` therefore carries the rotation's factor.
#: ⚠ The branch — ``theta`` in ``[0, 2 pi)`` — is what fixes which double-group element is
#: called ``C2(z)`` and which ``C2(z)bar``; see :mod:`kuiva.symm.groups`.
SPIN_FACTOR: Dict[Tuple[int, int], np.ndarray] = {
    (0, 0): np.eye(2, dtype=np.complex128),
    (1, 0): np.diag([-1j, 1j]).astype(np.complex128),
    (0, 1): np.eye(2, dtype=np.complex128),
    (1, 1): np.diag([-1j, 1j]).astype(np.complex128),
}


@dataclass(frozen=True)
class AOOperation:
    """``U(g)`` on the AO basis, as an index image and a sign — never a dense matrix.

    ``image[mu]`` is the AO that ``g chi_mu`` lands on and ``sign[mu]`` its coefficient, so
    applying it to MO coefficients is a scatter. :meth:`matrix` materializes it for tests and
    for the ``tr U(g)`` self-consistency check, which is the only place a dense form is wanted.
    """

    name: str
    spatial: Tuple[int, int]
    image: np.ndarray            # (nao,) int
    sign: np.ndarray             # (nao,) float, +-1
    atom_image: np.ndarray       # (natm,) int

    @property
    def nao(self) -> int:
        return int(self.image.size)

    def apply(self, c: np.ndarray) -> np.ndarray:
        """``U(g) c`` for AO-basis coefficient columns ``(nao, n)``."""
        c = np.asarray(c)
        out = np.zeros_like(c)
        out[self.image] = self.sign[:, None] * c if c.ndim == 2 else self.sign * c
        return out

    def matrix(self) -> np.ndarray:
        u = np.zeros((self.nao, self.nao))
        u[self.image, np.arange(self.nao)] = self.sign
        return u

    def two_component(self) -> np.ndarray:
        """``U(g)`` on the spin-blocked two-component AO basis ``[alpha ; beta]``.

        ``kron(D^(1/2), U_spatial)``: the spin factor multiplies the whole ``alpha`` and
        ``beta`` blocks, which is what the row ordering of :mod:`kuiva.spinor.expand` makes
        it. Complex, because ``D^(1/2)(C2(z))`` is.
        """
        return np.kron(SPIN_FACTOR[self.spatial], self.matrix()).astype(np.complex128)


def _atom_signatures(layout) -> List[Tuple]:
    """Per-atom fingerprint: element plus the exact shell list, so an operation may only map
    an atom onto one carrying the *same basis*. Two atoms of one element with different
    per-atom bases are not images of each other, however symmetric the geometry is."""
    per_atom: Dict[int, List[Tuple]] = {a: [] for a in range(layout.natm)}
    for sh in layout.shells:
        per_atom[int(sh.atom)].append(
            (int(sh.l), tuple(np.round(sh.exponents, 10)), tuple(np.round(sh.coefficients, 10))))
    return [(layout.atom_symbols[a], float(layout.atom_charges[a]), tuple(per_atom[a]))
            for a in range(layout.natm)]


def _transform_coords(coords: np.ndarray, spatial: Tuple[int, int]) -> np.ndarray:
    """``g r`` for the four spatial operations, as a diagonal sign on ``(x, y, z)``."""
    r, p = spatial
    d = np.ones(3)
    if r:                       # C2(z): x, y flip
        d[0] = d[1] = -1.0
    if p:                       # inversion: all three flip
        d *= -1.0
    return coords * d[None, :]


def atom_permutation(layout, spatial: Tuple[int, int], *,
                     tol: float = DEFAULT_ATOM_TOL) -> Optional[np.ndarray]:
    """``atom -> image atom`` under a spatial operation, or ``None`` if it is not a symmetry.

    Atoms match only when their element, nuclear charge and full shell list agree, so a
    per-atom basis assignment breaks a symmetry the geometry alone would have.
    """
    coords = np.asarray(layout.coords_bohr, dtype=float)
    moved = _transform_coords(coords, spatial)
    sig = _atom_signatures(layout)
    image = np.full(layout.natm, -1, dtype=int)
    for a in range(layout.natm):
        d = np.linalg.norm(coords - moved[a][None, :], axis=1)
        hit = np.nonzero(d <= tol)[0]
        if hit.size != 1 or sig[int(hit[0])] != sig[a]:
            return None
        image[a] = int(hit[0])
    if np.unique(image).size != layout.natm:
        return None
    return image


def ao_signs(layout, spatial: Tuple[int, int]) -> np.ndarray:
    """Per-AO sign of a spatial operation on the real solid harmonics (see the module table)."""
    l = np.asarray(layout.ao_l, dtype=int)
    m = np.abs(np.asarray(layout.ao_m, dtype=int))
    r, p = spatial
    exponent = np.zeros_like(l)
    if r:                       # C2(z): (-1)^|m|
        exponent = exponent + m
    if p:                       # inversion: (-1)^l
        exponent = exponent + l
    return np.where(exponent % 2 == 0, 1.0, -1.0)


def ao_operation(layout, spatial: Tuple[int, int], *,
                 tol: float = DEFAULT_ATOM_TOL) -> Optional[AOOperation]:
    """``U(g)`` for a spatial operation, or ``None`` when the molecule does not have it."""
    atom_image = atom_permutation(layout, spatial, tol=tol)
    if atom_image is None:
        return None
    ao_atom = np.asarray(layout.ao_atom, dtype=int)
    # AOs are laid out atom-major and, for atoms with identical shell lists, in identical
    # order within the atom -- which the signature check above is what guarantees.
    offsets = {a: np.nonzero(ao_atom == a)[0] for a in range(layout.natm)}
    image = np.empty(layout.nao, dtype=int)
    for a in range(layout.natm):
        src, dst = offsets[a], offsets[int(atom_image[a])]
        if src.size != dst.size:
            return None
        image[src] = dst
    return AOOperation(name=SPATIAL_NAMES[spatial], spatial=spatial, image=image,
                       sign=ao_signs(layout, spatial), atom_image=atom_image)


def detect_operations(layout, *, tol: float = DEFAULT_ATOM_TOL) -> Tuple[str, ...]:
    """Which of ``C2(z)``, ``i``, ``sigma(xy)`` the geometry has **in the input frame**.

    ⚠ Frame-dependent by design (see the module docstring): a molecule whose symmetry axis
    is ``x`` reports nothing here, and that is the honest answer for a spinor basis quantized
    along ``z``.
    """
    found = []
    for spatial in ((1, 0), (0, 1), (1, 1)):
        if ao_operation(layout, spatial, tol=tol) is not None:
            found.append(SPATIAL_NAMES[spatial])
    return tuple(found)


def group_operations(layout, group: Group, *,
                     tol: float = DEFAULT_ATOM_TOL) -> Dict[str, AOOperation]:
    """``{generator name: U(g)}`` for the spatial generators of ``group``.

    Refuses when the molecule does not have an operation the group needs, because a label
    read off an operation that is not a symmetry is a number with no meaning.
    """
    ops: Dict[str, AOOperation] = {}
    for gen in group.generators:
        if gen.spatial == (0, 0):        # Ebar: no spatial action, nothing to build
            continue
        op = ao_operation(layout, gen.spatial, tol=tol)
        if op is None:
            raise ValueError(
                "the geometry does not have {}, which {} needs. The operations are tested in "
                "the frame the geometry was given in and the molecule is never reoriented "
                "(that would move the gauge origin and every property operator fixed with "
                "it); orient the input so the symmetry axis is z, or ask for a smaller group"
                .format(gen.name, group.name))
        ops[gen.name] = op
    return ops


# --- Labelling ------------------------------------------------------------------------------

@dataclass
class LabellingReport:
    """What the labelling did and how well it held, for the output block and the tests."""

    n_adapted: int = 0             # degenerate blocks rotated into symmetry-adapted form
    max_rotation: float = 0.0      # largest |off-diagonal| of the block rotation applied
    max_residual: float = 0.0      # worst off-diagonal of C^T S U C after adaptation
    max_character_error: float = 0.0   # worst |chi| deviation from exactly +-1


def _scalar_exponent(gen: Generator, chi: float) -> int:
    """The label component of a scalar (spatial) orbital with character ``chi = +-1``.

    ``+1`` is the identity exponent and ``-1`` is the half-turn ``modulus // 2`` — which is
    ``1`` for inversion and ``2`` for a generator that squares to ``Ebar``. One rule, so a
    new generator does not need a new branch.
    """
    return 0 if chi > 0 else gen.modulus // 2


def _adapt_block(a: List[np.ndarray], start: int, stop: int) -> Tuple[np.ndarray, float]:
    """Simultaneously diagonalize the commuting ``+-1`` operators on one degenerate block.

    Splits the block on the first operator's eigenvalue, recurses on the rest. Returns the
    orthogonal rotation and the largest off-diagonal element it had to remove (zero when the
    block was already adapted, which is the common case).
    """
    n = stop - start
    rot = np.eye(n)
    off = 0.0
    for m in a:
        sub = rot.T @ m[start:stop, start:stop] @ rot
        off = max(off, float(np.max(np.abs(sub - np.diag(np.diag(sub))))) if n > 1 else 0.0)
        if n == 1 or np.max(np.abs(sub - np.diag(np.diag(sub)))) < 1e-12:
            continue
        # Eigenvalues are exactly +-1, so the eigenvectors of the symmetric sub-block are the
        # adapted directions; eigh's ascending order groups the -1 subspace first, which is a
        # deterministic (and therefore reproducible) choice.
        _, vec = np.linalg.eigh(0.5 * (sub + sub.T))
        rot = rot @ vec
    return rot, off


def label_scalar_orbitals(mo_coeff: np.ndarray, s_ao: np.ndarray, layout, group: Group, *,
                          mo_energy: Optional[np.ndarray] = None,
                          tol: float = DEFAULT_LABEL_TOL,
                          rtol: float = DEFAULT_GROUP_RTOL,
                          atom_tol: float = DEFAULT_ATOM_TOL,
                          ) -> Tuple[np.ndarray, np.ndarray, LabellingReport]:
    """Label one set of **real** scalar MOs, adapting degenerate blocks first.

    Returns ``(labels, coeff, report)``: ``labels`` is ``(nmo, group.width)``, ``coeff`` is the
    orbital set actually labelled (identical to the input unless a degenerate block had to be
    rotated), and ``report`` carries the diagnostics the output block prints.

    ⚠ The rotation is inside a degenerate block only, so the occupied density, the SCF energy
    and every observable are unchanged; the invariance is asserted by the suite rather than
    asserted here on every call.
    """
    c = np.asarray(mo_coeff, dtype=float)
    if c.ndim != 2:
        raise ValueError("label_scalar_orbitals takes one (nao, nmo) real MO set; got shape "
                         "{}".format(c.shape))
    s = np.asarray(s_ao, dtype=float)
    ops = group_operations(layout, group, tol=atom_tol)
    order = [g for g in group.generators if g.spatial != (0, 0)]
    rep = LabellingReport()

    mats = [c.T @ (s @ ops[g.name].apply(c)) for g in order]
    nmo = c.shape[1]
    if mats and nmo:
        bounds = _degenerate_blocks(mo_energy, nmo, rtol)
        rot_full = np.eye(nmo)
        for start, stop in bounds:
            rot, off = _adapt_block(mats, start, stop)
            if stop - start > 1 and off > 1e-12:
                rep.n_adapted += 1
                rep.max_rotation = max(rep.max_rotation, off)
            rot_full[start:stop, start:stop] = rot
        if rep.n_adapted:
            c = c @ rot_full
            mats = [c.T @ (s @ ops[g.name].apply(c)) for g in order]

    labels = np.zeros((nmo, group.width), dtype=int)
    for gen in order:
        j = group.generators.index(gen)
        m = mats[order.index(gen)]
        chi = np.diag(m).copy()
        offdiag = m - np.diag(chi)
        worst = float(np.max(np.abs(offdiag))) if nmo > 1 else 0.0
        rep.max_residual = max(rep.max_residual, worst)
        rep.max_character_error = max(rep.max_character_error,
                                      float(np.max(np.abs(np.abs(chi) - 1.0))) if nmo else 0.0)
        if worst > tol:
            bad = int(np.unravel_index(np.argmax(np.abs(offdiag)), offdiag.shape)[0])
            raise ValueError(
                "orbital {} is not an eigenvector of {}: the largest off-diagonal element of "
                "C^T S U(g) C is {:.3e}, above the tolerance {:.1e}. The orbitals are not "
                "symmetry-pure in {}, so no label can be read off them — check that the "
                "geometry really has this symmetry in the frame it was given in, and that the "
                "SCF converged".format(bad, gen.name, worst, tol, group.name))
        labels[:, j] = [_scalar_exponent(gen, float(x)) for x in chi]
    return labels, c, rep


def _degenerate_blocks(mo_energy, nmo: int, rtol: float) -> List[Tuple[int, int]]:
    """Runs of orbitals close enough in energy that the SCF may return them mixed.

    Without energies every orbital is its own block, which is the right default for a set
    that did not come from a diagonalization (an already-rotated or read-in orbital set).
    """
    if mo_energy is None:
        return [(i, i + 1) for i in range(nmo)]
    e = np.asarray(mo_energy, dtype=float).ravel()
    if e.size != nmo:
        raise ValueError("{} orbital energies for {} orbitals".format(e.size, nmo))
    blocks: List[Tuple[int, int]] = []
    start = 0
    scale = max(float(np.max(np.abs(e))), 1.0) if e.size else 1.0
    for i in range(1, nmo + 1):
        if i == nmo or abs(e[i] - e[i - 1]) > rtol * scale:
            blocks.append((start, i))
            start = i
    return blocks


__all__ = ["AOOperation", "DEFAULT_ATOM_TOL", "DEFAULT_LABEL_TOL", "LabellingReport",
           "SPIN_FACTOR", "ao_operation", "ao_signs", "atom_permutation", "detect_operations",
           "group_operations", "label_scalar_orbitals"]
