"""Determinant sectors: the label of a determinant, and the blocking it induces on ``H``.

A determinant is a product of occupied spinors, so its label is the **group sum** of their
labels — componentwise addition modulo the moduli, which for the groups here is one
matrix-vector product against the occupation matrix and a modulus. The two-electron
Hamiltonian conserves it exactly when the orbitals are symmetry-pure, so ``H`` is block
diagonal over sectors and each sector can be solved on its own.

Two things this buys, and they are different in kind:

* **state selection per irrep** — "the lowest two states of ``1E1/2g``" is a request the
  general path cannot express at all;
* **a structural fix for a biased Davidson guess.** A Krylov method cannot leave the invariant
  subspaces its starting vectors lie in, and a conserved label that is diagonal in the
  determinant basis is exactly such a subspace. A guess built *per sector* cannot miss a
  sector, whereas the lowest-diagonal determinants can all lie in one.

⚠ **The sector is exact only while the orbitals are pure.** A CASSCF rotates orbitals every
macro-iteration and a general-complex optimizer is not obliged to stay in the
symmetry-preserving subgroup; :mod:`kuiva.mcscf.orbopt` can be told to mask the rotation, and
where it is not, the purity is *measured* rather than assumed. Solving a sector on orbitals
that have drifted gives a converged, plausible, wrong answer, which is why the leakage is
checked and reported rather than trusted.

⚠ **Sector membership is not degeneracy-completeness.** Two members of a physically degenerate
manifold routinely carry *different* labels (any atom, any molecule reduced from a group whose
double group is non-abelian), so a per-sector count can split a manifold that the general path
would have kept whole. The state-averaging gate and the boundary diagnostic are what catch
that, and they run unchanged with symmetry on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util import resources as res
from ..util.logging import get_logger
from .groups import Group

log = get_logger(__name__)


def determinant_labels_gb(ndet: int, width: int) -> float:
    """Size [GB] of the per-determinant label array (exact sizing function)."""
    return res.array_gb((int(ndet), int(width)), np.int64)


def sector_index_gb(ndet: int) -> float:
    """Size [GB] of the resident determinant-to-sector index (exact sizing function)."""
    return res.array_gb((int(ndet),), np.int32)


def determinant_labels(occupations: np.ndarray, labels: np.ndarray,
                       moduli: Sequence[int]) -> np.ndarray:
    """``(ndet, width)`` labels from an ``(ndet, n_spinor)`` occupation matrix.

    The occupation matrix is :meth:`kuiva.ci.strings.CASSpace.occupations` — determinant
    machinery is imported from there and never re-derived here; this module only knows how to
    add labels up.

    ⚠ Accumulated one spinor at a time under a boolean mask rather than as
    ``occ.astype(int64) @ labels``: the matrix product's cast alone is ``8 * ndet * n_spinor``
    bytes, which is 160 MB at a million determinants and twenty spinors — larger than the
    determinant masks it is derived from, and for a quantity that is two small integers per
    determinant.
    """
    occ = np.asarray(occupations)
    lab = np.asarray(labels, dtype=np.int64)
    if occ.shape[1] != lab.shape[0]:
        raise ValueError("{} spinors in the occupations and {} labelled"
                         .format(occ.shape[1], lab.shape[0]))
    ndet, width = occ.shape[0], lab.shape[1]
    res.require("determinant irrep labels ({} determinants)".format(ndet),
                determinant_labels_gb(ndet, width),
                note="the group sum over each determinant's occupied spinors",
                advice=["reduce the active space"])
    total = np.zeros((ndet, width), dtype=np.int64)
    for m in range(lab.shape[0]):
        np.add(total, lab[m], out=total, where=occ[:, m][:, None])
    return total % np.asarray(moduli, dtype=np.int64)[None, :]


@dataclass
class SectorTable:
    """Which sector each determinant of a CAS space belongs to.

    Attributes
    ----------
    group : :class:`~kuiva.symm.groups.Group`
    sector_of : ``(ndet,)`` int
        Index into :attr:`sectors` for every determinant.
    sectors : list of label tuples
        The sectors that are actually **occupied**, in the group's canonical order. A sector
        with no determinants is not listed: asking for a state of it is a refusal, not an
        empty result.
    """

    group: Group
    sector_of: np.ndarray
    sectors: List[Tuple[int, ...]]

    @classmethod
    def build(cls, occupations: np.ndarray, labels: np.ndarray, group: Group) -> "SectorTable":
        res.reserve("determinant sector index ({} determinants)".format(occupations.shape[0]),
                    sector_index_gb(occupations.shape[0]),
                    note="one sector per determinant, resident for the whole solve",
                    advice=["drop the per-irrep selection, or reduce the active space"])
        det = determinant_labels(occupations, labels, group.moduli)
        keys = [tuple(int(x) for x in row) for row in det]
        present = [t for t in group.labels() if t in set(keys)]
        index = {t: k for k, t in enumerate(present)}
        return cls(group=group, sector_of=np.array([index[t] for t in keys], dtype=np.int32),
                   sectors=present)

    @property
    def n_sectors(self) -> int:
        return len(self.sectors)

    def size(self, sector) -> int:
        return int(np.count_nonzero(self.sector_of == self.index(sector)))

    def sizes(self) -> "Dict[Tuple[int, ...], int]":
        counts = np.bincount(self.sector_of, minlength=self.n_sectors)
        return {t: int(counts[k]) for k, t in enumerate(self.sectors)}

    def index(self, sector) -> int:
        """Position of a sector, given its label tuple or its irrep name."""
        label = self.group.label_of(sector)
        try:
            return self.sectors.index(label)
        except ValueError:
            raise ValueError(
                "{} holds no determinant of {}; this space has {}".format(
                    "this active space", self.group.irrep_name(label),
                    ", ".join("{} ({})".format(self.group.irrep_name(t), n)
                              for t, n in self.sizes().items())))

    def mask(self, sector) -> np.ndarray:
        """Boolean mask of the determinants in one sector."""
        return self.sector_of == self.index(sector)

    def indices(self, sector) -> np.ndarray:
        return np.nonzero(self.mask(sector))[0]

    def sector_weights(self, vectors: np.ndarray) -> np.ndarray:
        """``(n_states, n_sectors)`` weight of each state in each sector."""
        v = np.atleast_2d(np.asarray(vectors))
        power = np.abs(v) ** 2
        weight = np.zeros((v.shape[0], self.n_sectors))
        for k in range(self.n_sectors):
            weight[:, k] = power[:, self.sector_of == k].sum(axis=1)
        return weight

    def classify(self, vectors: np.ndarray, *, energies=None,
                 degeneracy_tol: float = 1e-6, tol: float = 1e-6
                 ) -> "Tuple[List[str], np.ndarray]":
        """``(names, leakage)`` for CI vectors — which sector each state lives in.

        ⚠ **A single state inside a degenerate block has no sector, and asking for one is the
        same error as reading a single spinor's populations inside a degenerate manifold.**
        Two conjugate sectors routinely meet at exactly the same energy — every Kramers pair
        of a molecule whose group is larger than the abelian one used does this — and the
        eigensolver is then entitled to return any rotation of the block, which is a mixture
        of both. Measured on a ``d^1`` ligand field: half the states of a plain lowest-n solve
        come back as 50/50 mixtures, and calling that "impure orbitals" would be a warning
        about the eigensolver's freedom rather than about the calculation.

        So the classification is **per degenerate block**, which is invariant: the block's
        per-sector weights must be whole numbers, and the block is named by the multiset of
        sectors it decomposes into (``"1E1/2 + 2E1/2"`` for the case above). ``leakage`` is how
        far those weights are from integers — zero for any legitimate mixture, and nonzero
        only when the label has actually stopped being conserved.

        ``energies`` groups the blocks; without them every state is its own block, which is
        the right reading for a set that did not come from one diagonalization (a per-irrep
        solve, where each state *is* sector-pure by construction).
        """
        from ..rdm.rdm import degenerate_blocks
        weight = self.sector_weights(vectors)
        n_states = weight.shape[0]
        blocks = (degenerate_blocks(energies, tol=degeneracy_tol) if energies is not None
                  else [(i, i + 1) for i in range(n_states)])
        names: List[str] = [""] * n_states
        leakage = np.zeros(n_states)
        for start, stop in blocks:
            total = weight[start:stop].sum(axis=0)
            rounded = np.rint(total)
            off = float(np.max(np.abs(total - rounded)))
            present = [self.group.irrep_name(self.sectors[k])
                       for k in range(self.n_sectors) if rounded[k] > 0]
            label = " + ".join(present) if off <= tol else "?"
            for i in range(start, stop):
                names[i] = label
                leakage[i] = off
        return names, leakage

    def name(self, sector) -> str:
        return self.group.irrep_name(self.group.label_of(sector))

    def __repr__(self) -> str:
        return "SectorTable({}, {} sectors of {} determinants)".format(
            self.group.name, self.n_sectors, self.sector_of.size)


def mode_bases(labels, group: Group):
    """Per-spinor :class:`~kuiva.dmrg.ttno.ModeBasis` widened by the irrep labels.

    Mode ``m`` is empty (``N = 0``, identity label) or occupied (``N = 1``, label of spinor
    ``m``), so the network's conserved quantum number becomes ``(N, m_1, ..., m_k)`` and a
    sweep cannot leave the sector by construction — the ``QuantumNumber`` widening the
    tensor-network layer was designed for, used as designed.

    ⚠ The cyclic components carry their **moduli** (:class:`kuiva.dmrg.block.QuantumNumber`).
    A finite cyclic group is not a subgroup of the integers, and a label added without its
    modulus drops Hamiltonian terms that are perfectly legal — Hermitian, correctly labelled
    and missing.

    Imported here rather than at module scope so ``import kuiva.symm`` does not drag the
    tensor-network layer in: the dependency runs symm -> nothing, and this is the one place a
    consumer's type is constructed on its behalf.
    """
    from ..dmrg.block import QuantumNumber
    from ..dmrg.ttno import ModeBasis
    lab = np.atleast_2d(np.asarray(getattr(labels, "labels", labels), dtype=int))
    moduli = (None,) + tuple(int(m) for m in group.moduli)
    empty = QuantumNumber(*([0] * (group.width + 1)), moduli=moduli)
    return {m: ModeBasis(2, (empty, QuantumNumber(*([1] + [int(x) for x in row]),
                                                  moduli=moduli)))
            for m, row in enumerate(lab)}


def sector_charge(sector, group: Group, n_elec: int):
    """The tensor-network target charge of ``n_elec`` electrons in one irrep."""
    from ..dmrg.block import QuantumNumber
    label = group.label_of(sector)
    moduli = (None,) + tuple(int(m) for m in group.moduli)
    return QuantumNumber(*([int(n_elec)] + [int(x) for x in label]), moduli=moduli)


#: Relative breach of sector conservation the integrals may show before a sector-blocked
#: solve is refused.
#:
#: ⚠ **Sized against the two-electron FACTORIZATION, not against SCF roundoff**, which is what
#: makes it far looser than the time-reversal check next door. A symmetry-forbidden ``(pq|rs)``
#: is zero in exact arithmetic, but the Cholesky decomposition reproduces the integrals only to
#: its own threshold, so the floor is that threshold measured **relative to the largest active
#: integral** — and that ratio grows as an active space reaches into diffuse virtuals whose
#: integrals are small. Measured at the default 1e-8 threshold: 2.8e-11 on a ligand-field
#: CAS(1, 10), then 3.8e-10 -> 3.4e-09 -> **6.2e-07** across one molecule's CAS(8, 12),
#: CAS(8, 14) and CAS(8, 16) — three orders over four spaces, from the factorization alone.
#: A genuinely broken symmetry is not in this band: an orbital set drifted by an angle
#: ``theta`` breaks conservation at ``O(theta)``, so this catches everything above ~1e-5 rad
#: and nothing below, which is exactly what the factorization noise permits. A tighter
#: ``cholesky_tol`` lowers the floor; this constant is not the knob for that.
SECTOR_TOL = 1.0e-5


def sector_violation(h: np.ndarray, eri: np.ndarray, labels: np.ndarray,
                     moduli: Sequence[int]) -> Tuple[float, float]:
    """``(one-electron, two-electron)`` relative breach of the sector conservation law.

    ``E_pq`` shifts a determinant's label by ``label(p) - label(q)``, so ``h_pq`` must vanish
    unless the two labels agree, and ``(pq|rs)`` unless ``l_p - l_q + l_r - l_s`` is the
    identity. Both are one pass over the active integrals — microseconds against a Davidson
    solve, and the only thing standing between orbitals that have drifted out of the
    symmetry and a converged, plausible, wrong per-irrep spectrum.
    """
    lab = np.asarray(labels, dtype=np.int64)
    mod = np.asarray(moduli, dtype=np.int64)
    n = lab.shape[0]
    diff = (lab[:, None, :] - lab[None, :, :]) % mod[None, None, :]      # (n, n, width)
    allowed_1 = np.all(diff == 0, axis=2)
    scale_h = max(float(np.max(np.abs(h))), 1e-300)
    err_h = float(np.max(np.abs(np.where(allowed_1, 0.0, np.abs(h)))))
    total = (diff[:, :, None, None, :] + diff[None, None, :, :, :]) % mod
    allowed_2 = np.all(total == 0, axis=4)
    scale_eri = max(float(np.max(np.abs(eri))), 1e-300)
    err_eri = float(np.max(np.abs(np.where(allowed_2, 0.0, np.abs(eri)))))
    return err_h / scale_h, err_eri / scale_eri


def assert_sector_symmetry(h: np.ndarray, eri: np.ndarray, labels: np.ndarray,
                           moduli: Sequence[int], *, tol: float = SECTOR_TOL,
                           what: str = "") -> None:
    """Raise unless the active integrals conserve the sector label to ``tol``.

    The message names the **orbitals**, because that is what the caller can act on: a CASSCF
    is entitled to rotate out of the symmetry-preserving subgroup unless it is told not to.
    """
    err_h, err_eri = sector_violation(h, eri, labels, moduli)
    if max(err_h, err_eri) <= tol:
        return
    raise ValueError(
        "{}the active-space integrals do not conserve the irrep label (relative breach "
        "{:.2e} one-electron, {:.2e} two-electron, against {:.1e}). The active orbitals are "
        "no longer symmetry-pure, so a per-irrep spectrum computed from them would be "
        "converged, plausible and wrong. Optimize with the symmetry-preserving orbital "
        "rotation mask, or drop the per-irrep selection"
        .format(what and (what + ": "), err_h, err_eri, tol))


def resolve_state_request(request, table: SectorTable) -> "List[Tuple[Tuple[int, ...], int]]":
    """Normalize ``n_states={irrep: n}`` into ``[(label, n), ...]`` in canonical order.

    Names and label tuples are both accepted (they are the same vocabulary), an unknown or
    empty sector is refused **naming what is available**, and a non-positive count is refused
    rather than treated as "none".
    """
    if not isinstance(request, dict):
        raise TypeError("a per-irrep state request is a mapping {irrep: n_states}, got {!r}"
                        .format(request))
    resolved: Dict[Tuple[int, ...], int] = {}
    for key, count in request.items():
        label = table.group.label_of(key)
        table.index(label)                    # refuses, naming the available sectors
        if int(count) <= 0:
            raise ValueError("asked for {} states of {}; a sector is either requested or left "
                             "out".format(count, table.group.irrep_name(label)))
        if label in resolved:
            raise ValueError("{} is requested twice".format(table.group.irrep_name(label)))
        resolved[label] = int(count)
    if not resolved:
        raise ValueError("a per-irrep state request must name at least one irrep")
    order = {t: k for k, t in enumerate(table.group.labels())}
    return sorted(resolved.items(), key=lambda kv: order[kv[0]])


__all__ = ["SECTOR_TOL", "SectorTable", "assert_sector_symmetry",
           "determinant_labels", "determinant_labels_gb", "mode_bases",
           "resolve_state_request", "sector_index_gb",
           "sector_charge", "sector_violation"]
