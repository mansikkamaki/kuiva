"""The molecule's surroundings: point charges, and what they are allowed to touch.

A single-molecule magnet is measured in a crystal. A bare gas-phase 3+ or 3- ion is not the
same system as the one in the lattice — it has a qualitatively different ligand field, and no
amount of care about the Hamiltonian recovers what the environment was never told to the
program. This module is the smallest honest way to say what surrounds the molecule: a set of
**point charges**, embedded electrostatically.

**What it touches, and what it must not.** An embedding of this kind adds
one term to the one-electron Hamiltonian and one term to the classical energy:

    V_emb(mu,nu) = - sum_A q_A <mu| 1/|r - R_A| |nu>          (electrons feel the charges)
    E_qn         = + sum_A sum_I q_A Z_I / |R_A - R_I|        (nuclei feel the charges)

and **nothing else in the program changes**. The multireference layer never learns that the
charges exist: it is handed a one-electron Hamiltonian, as it always was. That is not a
convenience — it is what makes an embedded calculation the *same* calculation, so that every
invariant, every degeneracy check and every stored product means what it meant before.

⚠ **The charge-charge energy of the environment with itself is not computed and not
reported.** It is a constant of the lattice, not of this calculation, and a program that
folded it into a total energy would produce a number no other program's total is comparable
with. What is reported is ``E_qn`` above, on its own line, so an embedded total stays
separable into the part that is chemistry and the part that is the field it sits in.

⚠ **The gauge origin does not move.** Charges have no mass and no nucleus, so the centre of
mass and the centre of nuclear charge are the molecule's own — deliberately, because the
orbital angular momentum in a property file has to be defined about the *molecule* whether or
not a lattice was included, or two calculations of the same complex could not be compared.

⚠ **Symmetry is the molecule's *and* the field's.** Point charges break symmetry as
effectively as atoms do, and a calculation that detected its point group from the nuclei alone
and then embedded it in a field of lower symmetry would label its states with irreps they do
not have. The detection therefore sees the charges (as zero-mass, zero-basis centres carrying
their charge), and a declared point group the field breaks is refused rather than used.

**The picture change** (user decision): the embedding potential is added **bare** — the
non-relativistic operator above, used unchanged in the two-component basis — and the
transformed variant is available through ``Environment(picture_change=True)``. The argument for
the default is that the charges sit outside the molecule, so their potential is smooth exactly
where the small component is large; the argument for measuring it rather than asserting it is
that "smooth enough" is a quantitative claim. ⚠ The transformed variant costs a
four-component-shaped integral **per charge**, where the bare one is a single batched grid
call, so a lattice of ten thousand charges is a different proposition in the two cases.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util import resources as res
from ..util.logging import get_logger
from ..util.timing import timer

log = get_logger(__name__)

#: Length units accepted for a charge field, and the factor onto bohr.
_UNITS = {"bohr": 1.0, "b": 1.0, "au": 1.0, "a.u.": 1.0,
          "angstrom": 1.0 / 0.52917721092, "ang": 1.0 / 0.52917721092,
          "a": 1.0 / 0.52917721092}

#: Charges smaller than this in magnitude are dropped, with a count reported. A lattice
#: summation or an Ewald fit routinely produces a tail of numerically-zero charges, and
#: evaluating an integral for each of them is pure cost.
NEGLIGIBLE_CHARGE = 1e-12

#: ⚠ How close a point charge may come to a nucleus before the embedding is refused. A point
#: charge sitting on top of an atom is not a physical model — the electronic energy diverges
#: as the electron density polarizes onto it without limit, and the SCF converges to something
#: that looks like a number. In bohr.
MIN_NUCLEUS_DISTANCE = 0.5


def _to_bohr(coords: np.ndarray, unit: str) -> np.ndarray:
    key = str(unit or "bohr").strip().lower()
    if key not in _UNITS:
        raise ValueError(
            "unknown length unit {!r} for a charge field; expected 'bohr' or "
            "'angstrom'".format(unit))
    return np.asarray(coords, dtype=float) * _UNITS[key]


@dataclass(frozen=True)
class PointCharges:
    """A field of classical point charges: magnitudes, positions, and the unit they are in.

    ⚠ **The unit is part of the data and is never inferred from the numbers.** A charge field
    copied out of a crystallographic file is in Angstrom, a field written by a
    quantum-chemistry program is usually in bohr, and the two differ by a factor of 1.89 that
    produces a perfectly plausible ligand field at the wrong distance. ``unit=""`` means "the
    molecule's own unit", which is the only inference that cannot be wrong, since it is the
    one the geometry beside it was written in.
    """

    charges: np.ndarray
    coords: np.ndarray
    unit: str = ""

    def __post_init__(self) -> None:
        q = np.ascontiguousarray(np.asarray(self.charges, dtype=float).ravel())
        r = np.ascontiguousarray(np.asarray(self.coords, dtype=float).reshape(-1, 3))
        if q.size != r.shape[0]:
            raise ValueError(
                "{} charges and {} positions: a point-charge field needs one of each".format(
                    q.size, r.shape[0]))
        object.__setattr__(self, "charges", q)
        object.__setattr__(self, "coords", r)

    def __len__(self) -> int:
        return int(self.charges.size)

    def in_bohr(self, molecule_unit: str = "Bohr") -> "PointCharges":
        """The same field with its coordinates in bohr, resolving ``unit=""``."""
        unit = self.unit or molecule_unit
        return PointCharges(charges=self.charges, coords=_to_bohr(self.coords, unit),
                            unit="bohr")

    @property
    def net_charge(self) -> float:
        return float(self.charges.sum())

    def digest(self) -> str:
        """A stable hash of the field — what identifies it in a stored product's header.

        A lattice of ten thousand charges cannot go into a file header, and a count and a net
        charge do not identify it. This does: two files carrying the same digest were embedded
        in the same field, and the check costs nothing.
        """
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.charges, dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(self.coords, dtype="<f8").tobytes())
        h.update(str(self.unit).encode("utf-8"))
        return h.hexdigest()

    def __repr__(self) -> str:
        return "PointCharges({} charges, net {:+.4f}, unit={!r})".format(
            len(self), self.net_charge, self.unit or "molecule's")


def coerce_point_charges(value, unit: str = "") -> Optional[PointCharges]:
    """Accept the forms a user actually has, and return one object.

    ``None``; a :class:`PointCharges`; a list of ``(q, (x, y, z))``; or a pair of arrays
    ``(charges, coords)`` — which is what a lattice generator or a file reader produces, and
    what a list of ten thousand tuples should not be built for.
    """
    if value is None:
        return None
    if isinstance(value, PointCharges):
        return value if not unit else PointCharges(value.charges, value.coords, unit)
    if isinstance(value, tuple) and len(value) == 2 \
            and np.ndim(value[0]) == 1 and np.ndim(value[1]) == 2:
        return PointCharges(charges=value[0], coords=value[1], unit=unit)
    entries = list(value)
    if not entries:
        return None
    charges, coords = [], []
    for entry in entries:
        try:
            q, xyz = entry
            x, y, z = xyz
        except (TypeError, ValueError):
            raise ValueError(
                "a point charge is written (q, (x, y, z)); got {!r}. A whole field may also "
                "be given as a (charges, coords) pair of arrays.".format(entry))
        charges.append(float(q))
        coords.append([float(x), float(y), float(z)])
    return PointCharges(charges=np.asarray(charges), coords=np.asarray(coords), unit=unit)


@dataclass(frozen=True)
class Environment:
    """What surrounds the molecule. Today: point charges.

    Parameters
    ----------
    point_charges
        The field, in any of the forms :func:`coerce_point_charges` accepts.
    unit : str
        The unit of those coordinates. Empty means **the molecule's own** — the only default
        that cannot silently be wrong.
    picture_change : bool
        Transform the embedding potential through the same X2C decoupling the Hamiltonian
        uses, instead of adding the bare non-relativistic operator. **Off by default** (user
        decision); see this module's docstring for the argument and the cost.
    label : str
        Free text carried into the provenance — what this field is, and where it came from.
    """

    point_charges: Optional[PointCharges] = None
    unit: str = ""
    picture_change: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_charges",
                           coerce_point_charges(self.point_charges, self.unit))

    @property
    def is_empty(self) -> bool:
        return self.point_charges is None or len(self.point_charges) == 0

    def resolved(self, molecule_unit: str = "Bohr") -> "Environment":
        """The same environment with its coordinates in bohr."""
        if self.is_empty:
            return self
        return Environment(point_charges=self.point_charges.in_bohr(molecule_unit),
                           unit="bohr", picture_change=self.picture_change, label=self.label)

    def __repr__(self) -> str:
        if self.is_empty:
            return "Environment(empty)"
        return "Environment({}, picture_change={})".format(
            repr(self.point_charges), self.picture_change)


@dataclass(frozen=True)
class EmbeddingRecord:
    """What environment a Hamiltonian was built in — the embedding half of provenance.

    ⚠ A **contract with stored data**, like the screening, decoupling and nuclear records
    beside it: a property matrix computed in a crystal field and one computed in vacuum are
    different physics wearing the same shape, and a file that does not say which it is cannot
    be interpreted. Add fields; do not rename them.

    The field itself is **not** stored here — a lattice does not belong in a file header — but
    :attr:`digest` identifies it exactly, so two files can be compared without it.
    """

    embedded: bool = False
    model: str = "none"
    n_charges: int = 0
    net_charge: float = 0.0
    total_abs_charge: float = 0.0
    min_nucleus_distance: float = 0.0
    max_nucleus_distance: float = 0.0
    picture_change: bool = False
    digest: str = ""
    label: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "embedded": bool(self.embedded),
            "model": str(self.model),
            "n_charges": int(self.n_charges),
            "net_charge": float(self.net_charge),
            "total_abs_charge": float(self.total_abs_charge),
            "min_nucleus_distance_bohr": float(self.min_nucleus_distance),
            "max_nucleus_distance_bohr": float(self.max_nucleus_distance),
            "picture_change": bool(self.picture_change),
            "digest": str(self.digest),
            "label": str(self.label),
        }

    def report(self, logger=None) -> None:
        logger = logger or log
        if not self.embedded:
            return
        out.entry(logger, "environment", "{} point charges".format(self.n_charges),
                  note=self.label or None)
        out.entry(logger, "net environment charge", self.net_charge, "e", fmt="{:+.6f}")
        out.entry(logger, "closest charge to a nucleus", self.min_nucleus_distance, "bohr",
                  fmt="{:.3f}")
        out.entry(logger, "embedding potential",
                  "picture-changed" if self.picture_change
                  else "bare (non-relativistic operator in the 2c basis)")


@dataclass(frozen=True)
class EmbeddingOperator:
    """The embedding terms in the molecule's own AO basis.

    ``h_sf`` is the spin-free potential (``nao, nao``, real symmetric) and ``w`` its
    spin-dependent partner in the conventions of :mod:`kuiva.spinor.expand` — ``None`` for the
    bare operator, which a scalar potential has no spin-dependent part of. ``e_nuclear`` is
    the classical charge-nucleus energy [Eh].
    """

    h_sf: np.ndarray
    w: Optional[np.ndarray]
    e_nuclear: float
    record: EmbeddingRecord

    @property
    def scale(self) -> float:
        return float(np.max(np.abs(self.h_sf))) if self.h_sf.size else 0.0


def _charge_nucleus_energy(mol, charges: np.ndarray, coords: np.ndarray) -> float:
    """``sum_A sum_I q_A Z_I / |R_A - R_I|`` [Eh], with the ghosts contributing nothing."""
    z = np.asarray(mol.atom_charges(), dtype=float)
    r = np.asarray(mol.atom_coords(), dtype=float)
    keep = z != 0.0
    if not keep.any() or charges.size == 0:
        return 0.0
    d = np.linalg.norm(coords[:, None, :] - r[None, keep, :], axis=2)
    return float(np.einsum("a,ai,i->", charges, 1.0 / d, z[keep]))


def _nucleus_distances(mol, coords: np.ndarray) -> Tuple[float, float]:
    z = np.asarray(mol.atom_charges(), dtype=float)
    r = np.asarray(mol.atom_coords(), dtype=float)[z != 0.0]
    if coords.size == 0 or r.size == 0:
        return 0.0, 0.0
    d = np.linalg.norm(coords[:, None, :] - r[None, :, :], axis=2)
    return float(d.min()), float(d.max())


def _bare_potential(mol, charges: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """``-sum_A q_A <mu| 1/|r - R_A| |nu>`` over the molecule's AO basis.

    Evaluated with the integral library's **batched** grid call rather than one
    ``int1e_rinv`` per charge — measured ten times faster and agreeing to 8e-16 — with the
    batch sized against the transient budget once, outside the loop: the batched integral is
    ``(n_batch, nao, nao)`` and a lattice would otherwise ask for hundreds of gigabytes.
    """
    nao = int(mol.nao)
    per_charge = nao * nao * 8.0 / res.BYTES_PER_GB
    budget = max(res.transient_gb(), per_charge)
    block = max(1, min(int(budget / per_charge), int(charges.size)))
    v = np.zeros((nao, nao))
    for start in range(0, int(charges.size), block):
        stop = min(start + block, int(charges.size))
        g = mol.intor("int1e_grids", grids=np.ascontiguousarray(coords[start:stop]))
        v -= np.einsum("g,gij->ij", charges[start:stop], np.asarray(g))
    return 0.5 * (v + v.T)


def _picture_changed_potential(mol, charges, coords, *, approx, decoupling_options,
                               light_speed):
    """The embedding potential transformed through the Hamiltonian's own X2C decoupling.

    The four-component form of a scalar potential is exactly the nuclear attraction's, one
    centre at a time::

        LL = V_A            SS = (sigma.p V_A sigma.p) / (4 c^2)          LS = SL = 0

    so this is the same construction :func:`kuiva.interface.pyscf_bridge.four_component_one_electron`
    makes for the nuclei, evaluated over the charge field instead — and transformed with the
    **same** ``X`` and ``R``, so the embedding potential and the Hamiltonian it is added to
    describe one picture rather than two.

    ⚠ **The correction is transformed, not the total.** Strictly, a potential added before the
    decoupling changes ``X`` itself; transforming the added operator with the charge-free
    ``X`` neglects that response, which is second order in a potential that is already a
    perturbation. It is the same approximation every picture-changed property operator in this
    program makes, and it is what makes this affordable at all.
    """
    from ..spinor.expand import (decompose_two_component, spin_block_diagonal,
                                 two_component_operator)
    from ..x2c.decouple import FourComponentBlocks, picture_change
    from .pyscf_bridge import _property_decoupling, four_component_one_electron

    from pyscf.x2c import x2c

    fc = four_component_one_electron(mol, uncontract=True, light_speed=light_speed)
    helper = x2c.SpinOrbitalX2CHelper(mol)
    helper.xuncontract = True
    xmol, _ = helper.get_xmol(mol)
    c = float(fc.light_speed)

    n = int(xmol.nao)
    v_ll = np.zeros((n, n))
    w_sf = np.zeros((n, n))
    w_so = np.zeros((3, n, n))
    with timer("embedding picture change"):
        for q, r in zip(charges, coords):
            xmol.set_rinv_origin(np.asarray(r, dtype=float))
            v_ll -= q * np.asarray(xmol.intor("int1e_rinv"))
            raw = np.asarray(xmol.intor("int1e_sprinvsp"))
            w_sf -= q * raw[3].real
            w_so -= q * raw[0:3].real
        ll = spin_block_diagonal(v_ll).astype(np.complex128)
        ss = two_component_operator(w_sf, w_so) * (0.25 / c ** 2)
        zero = np.zeros_like(ll)
        x, r_mat = _property_decoupling(mol, fc, approx, decoupling_options, "embedding")
        two_c = fc.contract(picture_change(
            FourComponentBlocks(ll=ll, ls=zero, sl=zero.copy(), ss=ss), x, r_mat))
    return decompose_two_component(two_c)


def embedding_operator(mol, environment: Optional[Environment], *, approx: str = "1e",
                       decoupling_options: Optional[Dict[str, object]] = None,
                       light_speed: Optional[float] = None) -> Optional[EmbeddingOperator]:
    """The embedding terms for ``mol``, or ``None`` when there is no environment.

    ⚠ **Coordinates must already be in bohr** — call :meth:`Environment.resolved` with the
    molecule's unit first. This function does not know what unit the geometry was written in
    and must not guess.
    """
    if environment is None or environment.is_empty:
        return None
    field = environment.point_charges
    if str(field.unit).lower() not in ("bohr", "b", "au", "a.u."):
        raise ValueError(
            "the charge field reaching embedding_operator() is in {!r}; resolve it to bohr "
            "against the molecule's unit first (Environment.resolved)".format(field.unit))

    charges, coords = np.asarray(field.charges), np.asarray(field.coords)
    small = np.abs(charges) < NEGLIGIBLE_CHARGE
    if small.any():
        log.debug("dropping %d point charges below %.1e e", int(small.sum()),
                  NEGLIGIBLE_CHARGE)
        charges, coords = charges[~small], coords[~small]
    if charges.size == 0:
        return None

    d_min, d_max = _nucleus_distances(mol, coords)
    if d_min < MIN_NUCLEUS_DISTANCE:
        raise ValueError(
            "a point charge sits {:.3f} bohr from a nucleus (the limit is {:.1f}). A charge "
            "on top of an atom is not a physical model: the density polarizes onto it without "
            "bound and the SCF converges to a number that means nothing. Move the charge, or "
            "model that centre as an atom.".format(d_min, MIN_NUCLEUS_DISTANCE))

    record = EmbeddingRecord(
        embedded=True, model="point-charges", n_charges=int(charges.size),
        net_charge=float(charges.sum()), total_abs_charge=float(np.abs(charges).sum()),
        min_nucleus_distance=d_min, max_nucleus_distance=d_max,
        picture_change=bool(environment.picture_change), digest=field.digest(),
        label=str(environment.label))

    with timer("environment embedding potential"):
        if environment.picture_change:
            h_sf, w = _picture_changed_potential(
                mol, charges, coords, approx=approx,
                decoupling_options=decoupling_options, light_speed=light_speed)
        else:
            h_sf, w = _bare_potential(mol, charges, coords), None
        e_nuc = _charge_nucleus_energy(mol, charges, coords)
    return EmbeddingOperator(h_sf=np.ascontiguousarray(h_sf), w=w, e_nuclear=float(e_nuc),
                             record=record)


__all__ = ["EmbeddingOperator", "EmbeddingRecord", "Environment", "MIN_NUCLEUS_DISTANCE",
           "NEGLIGIBLE_CHARGE", "PointCharges", "coerce_point_charges", "embedding_operator"]
