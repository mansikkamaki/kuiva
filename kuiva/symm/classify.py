"""Classifying converged states by the irreps of the full (non-abelian) double group.

⚠ **Classification, not adaptation.** Everything the calculation itself does — blocking the
CI, masking the orbital rotation, conserving a tensor-network quantum number — runs in the
abelian subgroup and is untouched by this module. What happens here happens **after** a solve:
the full point double group's operators are applied to the converged CI states, the block
traces are projected onto the group's characters, and each degenerate block gets the name of
the multiplet it is. There is no symmetry-adapted many-particle basis anywhere in it, no
double-group coupling coefficients, and nothing here may grow into either; non-abelian
symmetry *adaptation* stays out of scope and this layer is not a first instalment of it.

Why the labels are worth the cost
---------------------------------
Two things the abelian labels cannot say. First, **what a degenerate manifold is**: inside a
group whose double group is non-abelian a physically degenerate pair carries two *different*
abelian labels, so a per-irrep count can cut a multiplet in half and every abelian check
passes. With the multiplet named, its dimension is fixed by theory and a count that cuts it is
refused (:func:`assert_multiplet_boundary`). Second, **which physical multiplet a per-irrep
request selects**, which is the subduction table :meth:`kuiva.symm.double.DoubleGroup.
subduction` prints beside the two character tables.

⚠ It protects nothing across *near*-degeneracies of different irreps: two multiplets a few
wavenumbers apart are a different problem, and the state-average boundary gap and the
spin-non-invariance report stay load-bearing with this layer on.

How ``U(g)`` reaches a CI vector
--------------------------------
A non-abelian element mixes partner spinors, so its action on the determinant space is not a
signed permutation and cannot be applied the way the abelian sector labels are. What is used
here instead is the exact factorization every unitary has: ``u = G_1 G_2 ... G_m D`` with each
``G`` a rotation of two **adjacent** modes and ``D`` diagonal, and the Fock-space
representation is a homomorphism, so ``U(u)`` is the same product of the elementary factors.
Each factor costs one pass over the determinant coefficients:

* both modes empty or the mode diagonal — a multiply;
* both modes occupied — a multiply by ``det G``;
* exactly one occupied — a 2x2 mix of a determinant with the one that differs from it by
  moving that electron, with the fermionic phase of the swap.

⚠ **Adjacent modes are the point of choosing them.** The phase of the swap is
``(-1)^(occupied modes strictly between the two)``, which is ``+1`` when they are neighbours;
a general Givens pair would need that popcount, and a sign error there is norm-preserving,
Hermitian-looking and wrong. The elimination order below is the ordinary Givens QR restricted
to adjacent rows, which is why it can be.

This is **orchestration, not a kernel**: it runs once per converged block per class of the
group, never inside an iteration, and its cost is ``O(n_act^2 ndet)`` against the
``O(ndet n_act^2 n_elec)`` of a single application of ``H``. It is timed like any other region
so nobody mistakes a diagnostic for a solver step.

References
----------
* Projection of a reducible representation onto irreducible characters: M. Tinkham, "Group
  Theory and Quantum Mechanics", McGraw-Hill (1964), ch. 3.
* The two-mode factorization of a unitary and its Fock-space image: J. Thouless,
  "Stability conditions and nuclear rotations in the Hartree-Fock theory", Nucl. Phys. **21**,
  225 (1960), doi:10.1016/0029-5582(60)90048-1; the adjacent-pair elimination order is the
  Givens QR of G. H. Golub, C. F. Van Loan, "Matrix Computations", 4th ed., JHU Press (2013),
  section 5.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util.logging import get_logger
from ..util.timing import timer
from .double import DEFAULT_PROJECTION_TOL, DoubleGroup
from .operators import DEFAULT_ATOM_TOL
from .rotations import ao_transform

log = get_logger(__name__)

#: How far the active or inactive block of ``u`` may be from unitary before the space is
#: refused as not closed under the group. A closed space gives roundoff; an open one gives a
#: number of order the amplitude that leaked out, so the two are never close.
CLOSURE_TOL = 1.0e-6


# --- The orbital rotation on a CI vector ----------------------------------------------------

def _adjacent_pairs(space, p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(indices with p only, indices with p+1 only, indices with both)`` for adjacent modes.

    The first two are **aligned**: entry ``k`` of the second is the determinant that differs
    from entry ``k`` of the first by moving the electron from ``p`` to ``p+1``. Determinant
    addressing is :mod:`kuiva.ci.strings`' and is never re-derived here.
    """
    masks = space.masks
    bit_p = np.uint64(1) << np.uint64(p)
    bit_q = np.uint64(1) << np.uint64(p + 1)
    occ_p = (masks & bit_p) != 0
    occ_q = (masks & bit_q) != 0
    only_p = np.nonzero(occ_p & ~occ_q)[0]
    both = np.nonzero(occ_p & occ_q)[0]
    partner = space.rank(masks[only_p] ^ bit_p ^ bit_q)
    return only_p, np.asarray(partner, dtype=np.intp), both


def _apply_pair(vectors: np.ndarray, tables, block: np.ndarray) -> None:
    """One two-mode unitary, in place. ``block`` is the 2x2 acting on ``(a_p^dag, a_q^dag)``."""
    only_p, only_q, both = tables
    a, b, c, d = block[0, 0], block[0, 1], block[1, 0], block[1, 1]
    left = vectors[:, only_p].copy()
    right = vectors[:, only_q].copy()
    vectors[:, only_p] = a * left + b * right
    vectors[:, only_q] = c * left + d * right
    vectors[:, both] *= (a * d - b * c)


def givens_factors(u: np.ndarray) -> Tuple[List[Tuple[int, np.ndarray]], np.ndarray]:
    """``([(p, G), ...], diagonal)`` with ``u = G_1 G_2 ... G_m diag(D)``, all ``G`` adjacent.

    The ordinary Givens QR of a unitary matrix restricted to neighbouring rows: a unitary
    upper-triangular matrix is diagonal, so the triangularization terminates in a phase vector
    rather than in a general triangle.
    """
    work = np.array(u, dtype=np.complex128, copy=True)
    n = work.shape[0]
    factors: List[Tuple[int, np.ndarray]] = []
    for column in range(n):
        for row in range(n - 1, column, -1):
            top, bottom = work[row - 1, column], work[row, column]
            if abs(bottom) < 1.0e-14:
                continue
            norm = float(np.hypot(abs(top), abs(bottom)))
            g = np.array([[top, -np.conj(bottom)], [bottom, np.conj(top)]]) / norm
            work[row - 1:row + 1, :] = np.conj(g).T @ work[row - 1:row + 1, :]
            factors.append((row - 1, g))
    diagonal = np.diag(work).copy()
    off = work - np.diag(diagonal)
    if float(np.max(np.abs(off))) > 1.0e-8:
        raise ValueError("the orbital transformation is not unitary: triangularizing it left "
                         "an off-diagonal residual of {:.2e}".format(np.max(np.abs(off))))
    return factors, diagonal


def apply_orbital_rotation(space, u: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """``U(u) |Psi>`` for one CI vector or a stack, exactly.

    ``u`` is the transformation of the **active** spinors, ``U(u) a_p^dag U(u)^dag =
    sum_q a_q^dag u_qp``. Returns a new array; the input is not touched.
    """
    v = np.atleast_2d(np.array(vectors, dtype=np.complex128, copy=True))
    if v.shape[1] != space.ndet:
        raise ValueError("a CI vector over this space has length {}, got {}"
                         .format(space.ndet, v.shape[1]))
    factors, diagonal = givens_factors(u)
    occupations = space.occupations()
    # D acts on a determinant as the product of its occupied modes' phases. ⚠ Accumulated one
    # mode at a time under a boolean mask rather than as ``occ @ log(d)``: the cast that
    # product needs is ``16 * ndet * n_spinor`` bytes, which is larger than the determinant
    # masks it is derived from, and for one complex number per determinant.
    phases = np.ones(space.ndet, dtype=np.complex128)
    for mode in range(diagonal.size):
        np.multiply(phases, diagonal[mode], out=phases, where=occupations[:, mode])
    v *= phases[None, :]
    cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for p, g in reversed(factors):
        if p not in cache:
            cache[p] = _adjacent_pairs(space, p)
        _apply_pair(v, cache[p], g)
    return v if np.ndim(vectors) == 2 else v[0]


# --- The classifier -------------------------------------------------------------------------

@dataclass
class BlockIrreps:
    """One degenerate block of converged states and the multiplet it is."""

    start: int
    stop: int
    name: str
    multiplicities: "Dict[int, int]"
    residual: float
    traces: np.ndarray

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def classified(self) -> bool:
        return bool(self.multiplicities)


@dataclass
class Classification:
    """The classification of a whole spectrum: one :class:`BlockIrreps` per degenerate block."""

    group: DoubleGroup
    blocks: Tuple[BlockIrreps, ...]
    n_states: int

    def names(self) -> Tuple[str, ...]:
        """One irrep name per **state**, so it can be a column of the state table."""
        out_names = [""] * self.n_states
        for block in self.blocks:
            for i in range(block.start, block.stop):
                out_names[i] = block.name
        return tuple(out_names)

    @property
    def max_residual(self) -> float:
        return max((b.residual for b in self.blocks), default=0.0)

    @property
    def unclassified(self) -> Tuple[BlockIrreps, ...]:
        return tuple(b for b in self.blocks if not b.classified)


class StateClassifier:
    """Applies the full double group's operators to converged CI states and names the blocks.

    Built once per (group, orbital set, active space) and reused for every block, because the
    expensive parts — the AO operator matrices and the active-space transformations — depend on
    the orbitals and not on the states.

    ⚠ **The active space must be closed under the full group**, and this is checked rather than
    assumed: if an operation of the group maps an active spinor partly onto an inactive or
    virtual one, no operator on the CI space exists at all and the states cannot be classified.
    The refusal names the operation and how much leaked, because the fix is to the active space
    (add the missing partners) and not to the symmetry.
    """

    def __init__(self, group: DoubleGroup, layout, coeff: np.ndarray, s_ao: np.ndarray,
                 spaces, space, *, occupied_inactive: Optional[Sequence[int]] = None,
                 atom_tol: float = DEFAULT_ATOM_TOL) -> None:
        self.group = group
        self.space = space
        self.spaces = spaces
        self._layout = layout
        self._s_ao = s_ao
        self._occupied_inactive = occupied_inactive
        self._atom_tol = atom_tol
        coeff = np.asarray(coeff, dtype=np.complex128)
        s_ao = np.asarray(s_ao, dtype=float)
        nao = s_ao.shape[0]
        if coeff.shape[0] != 2 * nao:
            raise ValueError("the spinor coefficients have {} rows and the overlap {} AOs; "
                             "the rows are spin-blocked [alpha ; beta]"
                             .format(coeff.shape[0], nao))
        active = np.asarray(spaces.active, dtype=int)
        inactive = (np.asarray(spaces.inactive, dtype=int) if occupied_inactive is None
                    else np.asarray(occupied_inactive, dtype=int))
        self._active_u: List[np.ndarray] = []
        self._core_phase: List[complex] = []
        with timer("state classification: operator matrices"):
            for k in range(group.n_irreps):
                element = group.elements[group.class_representative(k)]
                transform = ao_transform(layout, element.cart, tol=atom_tol)
                if transform is None:
                    raise ValueError(
                        "the geometry does not have {}, which {} needs. The operations are "
                        "tested in the frame the geometry was given in and the molecule is "
                        "never reoriented".format(element.full_name(), group.name))
                moved = _spin_apply(transform, element.spin, coeff)
                overlap = np.empty_like(moved)
                overlap[:nao] = s_ao @ moved[:nao]
                overlap[nao:] = s_ao @ moved[nao:]
                u = np.conj(coeff).T @ overlap
                self._active_u.append(self._closed_block(u, active, element.full_name(),
                                                         "active"))
                core = self._closed_block(u, inactive, element.full_name(), "inactive")
                self._core_phase.append(complex(np.linalg.det(core)) if core.size
                                        else 1.0 + 0.0j)

    def rebuild(self, coeff: np.ndarray, spaces=None, space=None) -> "StateClassifier":
        """The same group and space at a **different orbital set**.

        ⚠ An operator matrix belongs to the orbitals it was computed from, so a CASSCF that
        has rotated them needs new ones. Making that a method rather than leaving it to the
        caller is what stops a converged spectrum from being classified with the starting
        orbitals' operators, which would be Hermitian, plausible and wrong.
        """
        return StateClassifier(self.group, self._layout, coeff, self._s_ao,
                               spaces if spaces is not None else self.spaces,
                               space if space is not None else self.space,
                               occupied_inactive=self._occupied_inactive,
                               atom_tol=self._atom_tol)

    @staticmethod
    def _closed_block(u: np.ndarray, columns: np.ndarray, name: str, what: str) -> np.ndarray:
        block = u[np.ix_(columns, columns)]
        if block.size == 0:
            return block
        residual = float(np.max(np.abs(np.conj(block).T @ block - np.eye(block.shape[0]))))
        if residual > CLOSURE_TOL:
            raise ValueError(
                "the {} space is not closed under {}: its block of U(g) is not unitary "
                "(residual {:.2e}). A state cannot be classified by an operation that takes "
                "it out of the space it was solved in -- the space needs the partner orbitals "
                "the operation maps into, which is what a degenerate manifold of the full "
                "group means".format(what, name, residual))
        return block

    def class_traces(self, vectors: np.ndarray, blocks: Sequence[Tuple[int, int]]) -> np.ndarray:
        """``(n_blocks, n_classes)`` block traces ``sum_i <Psi_i|U(g)|Psi_i>``.

        Invariant under any unitary rotation inside a block, which is what makes it a
        statement about the calculation rather than about the eigensolver's arbitrary choice
        of basis inside a degenerate manifold.
        """
        v = np.atleast_2d(np.asarray(vectors, dtype=np.complex128))
        traces = np.zeros((len(blocks), self.group.n_irreps), dtype=np.complex128)
        with timer("state classification: characters"):
            for k in range(self.group.n_irreps):
                moved = apply_orbital_rotation(self.space, self._active_u[k], v)
                moved *= self._core_phase[k]
                for b, (start, stop) in enumerate(blocks):
                    traces[b, k] = np.einsum("ij,ij->", np.conj(v[start:stop]),
                                             moved[start:stop])
        return traces

    def classify(self, vectors: np.ndarray, energies=None, *,
                 degeneracy_tol: float = 1.0e-6,
                 tol: float = DEFAULT_PROJECTION_TOL) -> Classification:
        """Name every degenerate block of a converged spectrum.

        ⚠ **Per block, never per state.** A single state inside a degenerate manifold has no
        irrep: the eigensolver may return any rotation of the block, and the character of a
        rotation is not a character of anything. The trace over the whole block is what is
        invariant, and a block that does not decompose into whole irreps to ``tol`` is
        reported as **not classified** rather than forced onto the nearest integers.
        """
        from ..rdm.rdm import degenerate_blocks
        v = np.atleast_2d(np.asarray(vectors, dtype=np.complex128))
        n_states = v.shape[0]
        bounds = (degenerate_blocks(energies, tol=degeneracy_tol) if energies is not None
                  else [(i, i + 1) for i in range(n_states)])
        traces = self.class_traces(v, bounds)
        sizes = np.array([len(m) for m in self.group.classes], dtype=float)
        blocks: List[BlockIrreps] = []
        for b, (start, stop) in enumerate(bounds):
            weights = (np.conj(self.group.characters) * sizes[None, :]) @ traces[b] \
                / self.group.order
            rounded = np.rint(np.real(weights))
            residual = float(np.max(np.abs(weights - rounded)))
            if residual <= tol and np.all(rounded >= -tol):
                multiplicities = {r: int(rounded[r]) for r in range(self.group.n_irreps)
                                  if rounded[r] > 0}
                name = " + ".join(
                    ("{} x ".format(m) if m > 1 else "") + self.group.irrep_names[r]
                    for r, m in multiplicities.items())
            else:
                multiplicities, name = {}, "?"
            blocks.append(BlockIrreps(start=start, stop=stop, name=name,
                                      multiplicities=multiplicities, residual=residual,
                                      traces=traces[b]))
        return Classification(group=self.group, blocks=tuple(blocks), n_states=n_states)


def _spin_apply(transform, spin: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """``U(g)`` on spin-blocked coefficients with the group **element's** spin factor.

    ⚠ The factor is the element's, not one recomputed from its spatial matrix: ``g`` and
    ``g Ebar`` share a spatial operation and differ by exactly this sign, and taking it from
    the matrix would collapse the double group back onto the point group without any error
    message.
    """
    nao = transform.nao
    upper = transform.apply(coeff[:nao])
    lower = transform.apply(coeff[nao:])
    out_array = np.empty_like(coeff)
    out_array[:nao] = spin[0, 0] * upper + spin[0, 1] * lower
    out_array[nao:] = spin[1, 0] * upper + spin[1, 1] * lower
    return out_array


# --- The gate -------------------------------------------------------------------------------

def assert_multiplet_boundary(classification: Classification, *, on_split: str = "raise",
                              what: str = "") -> None:
    """Refuse a state selection that cuts a multiplet whose dimension theory fixes.

    ⚠ **This is what the classification layer is *for*, and it is also all it can promise.**
    A block that does not decompose into whole irreps of the group is either the fragment of a
    manifold the state count truncated — the honest reading when it is the **last** block —
    or a sign that the orbitals are no longer pure enough for the labels to mean anything.
    The message says which is which and names the block either way.

    It protects nothing across *near*-degeneracies of different irreps: a manifold that is
    split by a few wavenumbers is not one block, and the state-average boundary gap and the
    spin-non-invariance report are what speak to that.
    """
    bad = classification.unclassified
    if not bad:
        return
    last = classification.blocks[-1]
    where = ", ".join("{}-{}".format(b.start, b.stop - 1) for b in bad)
    message = (
        "{}states {} do not decompose into whole irreps of {} (worst residual {:.2e}). A "
        "degenerate block of the full double group has a dimension theory fixes, so a block "
        "that is not a whole number of multiplets is a fragment of one"
        .format(what and (what + ": "), where, classification.group.name,
                max(b.residual for b in bad)))
    if bad[-1] is last:
        message += (". The last block is the truncated one: the state count cuts a multiplet, "
                    "and a density averaged over a fragment of one depends on the arbitrary "
                    "rotation the eigensolver chose inside it")
    else:
        message += (". No truncated block is at the end, so this is the orbitals rather than "
                    "the count: they are no longer symmetry-pure enough to carry the labels")
    if on_split == "raise":
        raise ValueError(message)
    if on_split != "warn":
        raise ValueError("on_split must be 'raise' or 'warn', got {!r}".format(on_split))
    log.warning("%s", message)


def report(classification: Classification, logger=None, *, level=None) -> None:
    """The block table: which multiplet each degenerate block is, and its residual."""
    logger = logger or log
    group = classification.group
    width = max(12, max((len(b.name) for b in classification.blocks), default=12))
    table = out.Table(logger, [out.Column("states", "{}", 10, align="<"),
                               out.col_count("size", 6),
                               out.Column("multiplet", "{}", width, align="<"),
                               out.col_sci("residual")],
                      **({} if level is None else {"level": level}))
    table.start("state multiplets in {} (classification only; the math ran in the abelian "
                "subgroup)".format(group.name))
    for block in classification.blocks:
        table.row("{}-{}".format(block.start, block.stop - 1), block.size,
                  block.name, block.residual)
    table.end("a block reported as ? does not decompose into whole irreps: either the state "
              "count cut a multiplet or the orbitals are no longer symmetry-pure")


__all__ = ["BlockIrreps", "CLOSURE_TOL", "Classification", "StateClassifier",
           "apply_orbital_rotation", "assert_multiplet_boundary", "givens_factors", "report"]
