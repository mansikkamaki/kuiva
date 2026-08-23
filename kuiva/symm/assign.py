"""Assigning labels to a molecule's orbitals, and the containers that carry them onward.

This is the plumbing layer of :mod:`kuiva.symm`: it takes the AO layout and the scalar MOs
the front end produced, decides which group is actually being used, labels the orbitals, and
returns two plain-array containers — one for the scalar reference and one for the spinors —
that ride on the ingested data all the way to the CI.

⚠ **Symmetry is opt-in and stays opt-in.** ``point_group="auto"`` *detects and reports* the
operations the geometry has, and labels with them; but per-irrep state selection changes what
``n_states`` means, so a detected group never changes the meaning of a request the user did
not make: without ``n_states={irrep: n}`` every selection is the plain "lowest n" it has
always been, whatever the labels say.

⚠ **Absent labels mean "no symmetry", and every consumer treats that as the behaviour it had
before this module existed.** A container built by hand in a test, an unrestricted reference
whose labels could not be assigned, a molecule with no operations in its input frame — all
of them reach the CI as ``symmetry = None`` and take exactly the old path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util.degeneracy import DEFAULT_GROUP_RTOL
from ..util.logging import get_logger
from .groups import GROUPS, REDUCTION, Group, resolve_group
from .operators import (DEFAULT_ATOM_TOL, DEFAULT_LABEL_TOL, LabellingReport, detect_operations,
                        label_scalar_orbitals)

log = get_logger(__name__)

#: Which label group each detected operation set gives. ``sigma(xy)`` alone is ``Cs(xy)``;
#: with either of the other two present it is their product and adds nothing.
_DETECTED_GROUP = {
    ("C2(z)", "i", "sigma(xy)"): "C2h(z)",
    ("C2(z)",): "C2(z)",
    ("i",): "Ci",
    ("sigma(xy)",): "Cs(xy)",
    (): "C1",
}


def group_from_operations(found: Sequence[str]) -> Group:
    """The largest label group the detected operations support."""
    return GROUPS[_DETECTED_GROUP[tuple(found)]]


@dataclass(frozen=True)
class OrbitalLabels:
    """Labels for one set of orbitals: the group and one integer tuple per orbital.

    Plain arrays and one descriptor — the whole content of what crosses the ingestion
    boundary, exactly as the Hamiltonian provenance records do.
    """

    group: Group
    labels: np.ndarray                 # (n, width) int

    def __post_init__(self) -> None:
        arr = np.ascontiguousarray(np.atleast_2d(np.asarray(self.labels, dtype=int)))
        if arr.shape[1] != self.group.width:
            raise ValueError("labels are {} wide; {} has {} components"
                             .format(arr.shape[1], self.group.name, self.group.width))
        object.__setattr__(self, "labels", arr)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def tuples(self) -> List[Tuple[int, ...]]:
        return [tuple(int(x) for x in row) for row in self.labels]

    def names(self) -> List[str]:
        return [self.group.irrep_name(t) for t in self.tuples()]

    def take(self, columns) -> "OrbitalLabels":
        """The labels of a subset of the orbitals, in the order given."""
        idx = np.asarray(columns, dtype=int).ravel()
        return OrbitalLabels(group=self.group, labels=self.labels[idx])

    def counts(self) -> "Dict[Tuple[int, ...], int]":
        """``{label: how many orbitals carry it}``, in the group's canonical order."""
        out: Dict[Tuple[int, ...], int] = {}
        for t in self.group.labels():
            out[t] = 0
        for t in self.tuples():
            out[t] = out.get(t, 0) + 1
        return {k: v for k, v in out.items() if v or self.group.is_fermion(k)}

    def __repr__(self) -> str:
        return "OrbitalLabels({}, {} orbitals)".format(self.group.name, len(self))


@dataclass(frozen=True)
class MolecularSymmetry:
    """The symmetry of an ingested scalar reference: group, provenance, and scalar labels.

    ``scalar`` holds one :class:`OrbitalLabels` per MO set — one for a restricted reference,
    two for an unrestricted one, matching :meth:`~kuiva.interface.pyscf_bridge.ScalarX2CData.
    mo_sets`.
    """

    group: Group
    requested: str
    detected: Tuple[str, ...]
    scalar: Tuple[OrbitalLabels, ...]
    report: LabellingReport
    reduced_from: Optional[str] = None
    unused: Tuple[str, ...] = ()
    #: The molecule's **full** point double group
    #: (:class:`kuiva.symm.double.DoubleGroup`), or ``None`` when the classification layer is
    #: off or has nothing to add. ⚠ It is used for **labelling converged states only** — the
    #: mathematics of every stage runs in :attr:`group`, the abelian one — and it is present
    #: precisely when the abelian group is *not* the whole story, which is when a per-irrep
    #: count can cut a physically degenerate manifold without any abelian check noticing.
    full_group: Optional[object] = None
    #: Whether the finite classification group is a truncation of an infinite one (a linear
    #: molecule). Reported, because the labels are then those of the subgroup, not of D(inf)h.
    axial_truncation: bool = False

    @property
    def unrestricted(self) -> bool:
        return len(self.scalar) == 2

    def spinor_labels(self) -> OrbitalLabels:
        """Labels of the spinor set the scalar orbitals expand into.

        Restricted: each scalar orbital becomes a Kramers pair carrying conjugate labels
        (:meth:`kuiva.symm.groups.Group.spinor_labels`). Unrestricted: spinor ``2p`` is the
        ``p``-th **alpha** orbital and ``2p+1`` the ``p``-th **beta** one, so the two shifts
        are applied to two different label sets and the pair is not conjugate — which is the
        same statement as ``kramers_paired = False``.
        """
        if not self.unrestricted:
            return OrbitalLabels(group=self.group,
                                 labels=self.group.spinor_labels(self.scalar[0].labels))
        shift = np.array([g.spin_shift for g in self.group.generators], dtype=int)
        mod = np.asarray(self.group.moduli, dtype=int)
        a, b = self.scalar[0].labels, self.scalar[1].labels
        out = np.empty((a.shape[0] + b.shape[0], self.group.width), dtype=int)
        out[0::2] = (a + shift[:, 0][None, :]) % mod[None, :]
        out[1::2] = (b + shift[:, 1][None, :]) % mod[None, :]
        return OrbitalLabels(group=self.group, labels=out)

    def provenance(self) -> Dict[str, object]:
        """The symmetry as JSON-able metadata, for a stored product's header."""
        return {
            "group": self.group.name,
            "requested": self.requested,
            "detected": list(self.detected),
            "reduced_from": self.reduced_from,
            "classification_group": None if self.full_group is None else self.full_group.name,
            "generators": [g.name for g in self.group.generators],
            "moduli": list(self.group.moduli),
            "frame": "input (the molecule is never reoriented)",
        }

    def __repr__(self) -> str:
        return "MolecularSymmetry({}, requested={!r}, detected={})".format(
            self.group.name, self.requested, self.detected)


def resolve_classification(layout, abelian: Group, request, *,
                           atom_tol: float = DEFAULT_ATOM_TOL):
    """``(full double group or None, truncated)`` for a classification request.

    ``"auto"`` detects the largest supported group the geometry has **in its input frame** and
    activates the layer only when that group is genuinely larger than the abelian label group
    — where the abelian group already *is* the whole story there is nothing for a non-abelian
    label to add, and three more tables in the output would be noise. ``False``/``None``
    switches it off; a named group is verified rather than assumed.
    """
    from .double import detect_point_group, double_group, has_group, is_linear
    if request in (False, None, "off", "none"):
        return None, False
    # ⚠ The label group must sit *inside* the classification group, or the correspondence
    # between their irreps does not exist and the two would be two vocabularies rather than
    # one. These are the abelian group's own spatial operations, as matrices rather than as
    # names, so the requirement is the same object the subduction is computed from.
    required = [_spatial_matrix(abelian.element_spatial(e)[0]) for e in abelian.elements()]
    if isinstance(request, str) and request.lower() != "auto":
        group = double_group(request)
        if not has_group(layout, group.name, tol=atom_tol):
            raise ValueError(
                "the geometry does not have every operation of {}, which the classification "
                "layer was asked for. The operations are tested in the frame the geometry was "
                "given in and the molecule is never reoriented; orient the input, or ask for "
                "a smaller group".format(group.name))
        group.subduction(abelian)          # refuses if the label group is not inside it
        return group, is_linear(layout)
    group = double_group(detect_point_group(layout, tol=atom_tol, require=required))
    if group.order <= abelian.order:
        return None, False
    return group, is_linear(layout)


def _spatial_matrix(spatial) -> np.ndarray:
    """The 3x3 matrix of an abelian element's ``(C2(z) bit, inversion bit)``."""
    r, p = spatial
    cart = np.diag([-1.0, -1.0, 1.0]) if r else np.eye(3)
    return -cart if p else cart


def analyze(layout, mo_sets: Sequence[np.ndarray], s_ao: np.ndarray, *,
            point_group: str = "auto", mo_energy=None, classification="auto",
            tol: float = DEFAULT_LABEL_TOL, rtol: float = DEFAULT_GROUP_RTOL,
            atom_tol: float = DEFAULT_ATOM_TOL,
            ) -> Tuple[MolecularSymmetry, Tuple[np.ndarray, ...]]:
    """Label a scalar reference. Returns ``(symmetry, mo_sets)``.

    The returned MO sets are the input ones unless a degenerate block had to be rotated into
    symmetry-adapted form, which is a rotation inside a degenerate eigenspace and changes no
    observable (:mod:`kuiva.symm.operators`).

    ``point_group="auto"`` uses whatever the geometry has in its own frame. A named group is
    **verified**, not assumed: an operation the molecule does not have is refused rather than
    quietly dropped, because a label read off a non-symmetry is a number with no meaning.
    """
    detected = detect_operations(layout, tol=atom_tol)
    requested = str(point_group)
    reduced_from = None
    if requested.lower() == "auto":
        group = group_from_operations(detected)
    else:
        group, was_reduced = resolve_group(requested)
        if was_reduced:
            reduced_from = requested
    available = group_from_operations(detected)
    spanned = {group.element_name(e) for e in group.elements()}
    unused = tuple(op for op in detected if op not in spanned)

    sets: List[OrbitalLabels] = []
    coeffs: List[np.ndarray] = []
    report = LabellingReport()
    energies = _energy_sets(mo_energy, len(mo_sets))
    for c, e in zip(mo_sets, energies):
        labels, adapted, rep = label_scalar_orbitals(c, s_ao, layout, group, mo_energy=e,
                                                     tol=tol, rtol=rtol, atom_tol=atom_tol)
        sets.append(OrbitalLabels(group=group, labels=labels))
        coeffs.append(adapted)
        report = LabellingReport(
            n_adapted=report.n_adapted + rep.n_adapted,
            max_rotation=max(report.max_rotation, rep.max_rotation),
            max_residual=max(report.max_residual, rep.max_residual),
            max_character_error=max(report.max_character_error, rep.max_character_error))

    full, truncated = resolve_classification(layout, group, classification, atom_tol=atom_tol)
    symmetry = MolecularSymmetry(group=group, requested=requested, detected=detected,
                                 scalar=tuple(sets), report=report,
                                 reduced_from=reduced_from, unused=tuple(sorted(set(unused))),
                                 full_group=full, axial_truncation=bool(truncated))
    if full is not None and truncated:
        log.warning("the molecule is linear, so its true point group is infinite; the "
                    "classification layer uses %s, the largest finite group it tests, and the "
                    "multiplet labels are that group's rather than the axial ones",
                    full.name)
    if available.order > group.order and reduced_from is None and requested.lower() != "auto":
        log.warning("the geometry has %s but %s was asked for; the extra operations are not "
                    "used and the sectors are correspondingly coarser",
                    available.name, group.name)
    return symmetry, tuple(coeffs)


def _energy_sets(mo_energy, n_sets: int):
    if mo_energy is None:
        return [None] * n_sets
    e = np.asarray(mo_energy, dtype=float)
    if n_sets == 1:
        return [e.ravel()]
    if e.ndim != 2 or e.shape[0] != 2:
        raise ValueError("an unrestricted reference needs a (2, nmo) orbital-energy array")
    return [e[0], e[1]]


__all__ = ["MolecularSymmetry", "OrbitalLabels", "analyze", "group_from_operations",
           "resolve_classification"]
