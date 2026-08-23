"""Full point double groups: their elements, their character tables, and their subduction.

⚠ **This layer CLASSIFIES; it does not adapt.** The mathematics of a Kuiva calculation runs in
the abelian subgroup of :mod:`kuiva.symm.groups` — that is what blocks the CI, masks the
orbital rotation and widens the tensor-network quantum number. What this module adds is the
*labelling* of already-converged states by the irreps of the molecule's full (in general
non-abelian) double group, by projecting characters. There is no symmetry-adapted many-particle
basis here, no double-group Clebsch-Gordan machinery and no non-abelian tensor network, and
nothing in this module may grow into one: non-abelian symmetry **adaptation** is out of scope,
and classification is in scope precisely because it costs a fraction of adaptation.

What is stored, and what is computed
------------------------------------
⚠ **The character tables are computed, not transcribed**, which is a deliberate departure from
the usual practice of shipping tabulated ones. What is stored is only the **generators** of
each supported group as explicit ``(3x3 spatial, 2x2 spin)`` matrices — a handful of rotations
and reflections, whose correctness is visible by inspection — and the group is the closure of
those under multiplication in ``SO(3) x SU(2)``. The double group then arises by construction
rather than by convention: ``C2(z)`` squares to ``diag(-1, -1)``, which *is* ``Ebar``, so an
element is a spatial operation **paired with the spin factor it was reached by**, and ``g`` and
``g Ebar`` are two different elements of the closure.

The characters follow from the elements by Burnside's class-sum construction (below), so the
printed table is provably the table of the operators the calculation actually used. That is
what a transcribed table cannot promise: a table and a set of operator matrices that were
never compared can disagree about which axis is ``z``, and every number they produce is then
subtly about a different calculation with no error message anywhere.

The irrep **names** are assigned by rule rather than tabulated, for the same reason: Mulliken's
rules for the single-valued irreps and the ``|omega|`` of the principal axis for the
double-valued ones. The suite checks the resulting names against the published tables for
``C2v``, ``C3v``, ``D3h`` and ``D2h``, which is where transcription belongs — in a test that
can fail, not in the code that would then be right by definition.

The subduction table
--------------------
Each non-abelian irrep restricted to the abelian label group decomposes into abelian sectors,
and that correspondence is what connects a per-irrep ``n_states`` request to the physical
multiplet it selects. It is **computed** from the two character tables by the ordinary
projection formula, so a mistake in either table breaks the printed correspondence rather than
hiding in it.

References
----------
* Class-sum construction of a character table: W. Burnside, "Theory of Groups of Finite
  Order", 2nd ed., Cambridge University Press (1911), ch. XV; the numerical form used here
  (simultaneous eigenvectors of the class-sum matrices) follows J. D. Dixon, "High speed
  computation of group characters", Numer. Math. **10**, 446 (1967), doi:10.1007/BF02162876.
* Irrep naming: R. S. Mulliken, "Report on Notation for the Spectra of Polyatomic Molecules",
  J. Chem. Phys. **23**, 1997 (1955), doi:10.1063/1.1740655.
* The double groups and the published tables the suite checks the names against:
  G. F. Koster, J. O. Dimmock, R. G. Wheeler, H. Statz, "Properties of the Thirty-Two Point
  Groups", MIT Press (1963); S. L. Altmann, P. Herzig, "Point-Group Theory Tables", Clarendon
  Press (1994).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util.logging import get_logger
from .groups import SPATIAL_NAMES, Group
from .operators import DEFAULT_ATOM_TOL, SPIN_FACTOR
from .rotations import (ELEMENT_TOL, atom_permutation_general, canonical_axis_angle,
                        operation_name, reflection, rotation)

log = get_logger(__name__)

#: Highest rotation order the detector tests. A linear molecule has an infinite axis and no
#: finite group contains it; the largest tested subgroup is what gets used, and the detector
#: says so rather than pretending the answer is exact.
MAX_ROTATION_ORDER = 6

#: Residual above which a block of states is reported as **not classified** rather than being
#: forced onto the nearest integer decomposition. Sized well above the roundoff of a character
#: sum over a converged block (~1e-10) and far below the ~0.5 a genuinely mixed block shows.
DEFAULT_PROJECTION_TOL = 1.0e-3


def _spatial_generators(name: str) -> Tuple[str, List[np.ndarray]]:
    """``(canonical name, generator matrices)`` for a Schoenflies point-group name.

    The generators, not the elements: ``D3h`` is ``C3(z)``, ``C2(x)`` and ``sigma(xy)``, and
    the other twenty-one elements are their products. ⚠ The axis convention is fixed here and
    is the same one everything else in :mod:`kuiva.symm` uses — the principal axis is ``z`` and
    the first twofold axis (or vertical plane normal) is ``x``. The molecule is never
    reoriented to match, so a geometry in another frame detects a smaller group and the report
    says so.
    """
    text = str(name).strip()
    key = text[:1].upper() + text[1:].lower()
    z, x, y = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    if key in ("C1",):
        return "C1", []
    if key in ("Ci", "S2"):
        return "Ci", [-np.eye(3)]
    if key == "Cs":
        return "Cs", [reflection(z)]
    body, suffix = key[0], key[-1] if len(key) > 2 else ""
    digits = "".join(ch for ch in key if ch.isdigit())
    if not digits:
        raise ValueError("unknown point group {!r}".format(name))
    n = int(digits)
    turn = rotation(z, 2.0 * np.pi / n)
    if key == "C{}".format(n):
        return key, [turn]
    if key == "C{}h".format(n):
        return key, [turn, reflection(z)]
    if key == "C{}v".format(n):
        return key, [turn, reflection(y)]
    if key == "D{}".format(n):
        return key, [turn, rotation(x, np.pi)]
    if key == "D{}h".format(n):
        return key, [turn, rotation(x, np.pi), reflection(z)]
    if key == "D{}d".format(n):
        return key, [reflection(z) @ rotation(z, np.pi / n), rotation(x, np.pi)]
    if key == "S{}".format(n) and n % 2 == 0:
        return key, [reflection(z) @ rotation(z, 2.0 * np.pi / n)]
    raise ValueError(
        "unknown or unsupported point group {!r}. The classification layer has C1, Ci, Cs, "
        "and the Cn / Cnh / Cnv / Dn / Dnh / Dnd / S2n families up to n = {}; the cubic groups "
        "are not implemented because no system in this project needs them yet"
        .format(name, MAX_ROTATION_ORDER))


#: Names and spin factors, memoized on the rounded spatial matrix. ⚠ Not an optimization of
#: taste: closing a group of order 48 forms a few thousand products, each of which would
#: otherwise re-derive an axis and an angle through an eigendecomposition. Keyed on the matrix
#: itself, so it is a pure function's cache and cannot go stale.
_NAME_CACHE: Dict[Tuple, str] = {}
_SPIN_CACHE: Dict[Tuple, np.ndarray] = {}


def _cart_key(cart: np.ndarray) -> Tuple:
    return tuple(np.round(np.asarray(cart, dtype=float).ravel(), 6) + 0.0)


def _named(cart: np.ndarray) -> str:
    key = _cart_key(cart)
    if key not in _NAME_CACHE:
        _NAME_CACHE[key] = operation_name(cart)
    return _NAME_CACHE[key]


def _spin_of(cart: np.ndarray) -> np.ndarray:
    from .rotations import spin_factor
    key = _cart_key(cart)
    if key not in _SPIN_CACHE:
        _SPIN_CACHE[key] = spin_factor(cart)
    return _SPIN_CACHE[key]


def _key(cart: np.ndarray, spin: np.ndarray) -> Tuple:
    """Hashable identity of a double-group element, tolerant to roundoff."""
    cart = np.round(np.asarray(cart, dtype=float).ravel(), 6) + 0.0
    spin = np.round(np.ascontiguousarray(spin, dtype=np.complex128).ravel()
                    .view(np.float64), 6) + 0.0
    return (tuple(cart), tuple(spin))


@dataclass(frozen=True)
class Element:
    """One element of a double group: a spatial operation **and** the spin factor it carries.

    ⚠ The pair is the element. ``g`` and ``g Ebar`` share a :attr:`cart` and differ only in
    :attr:`spin`, which is exactly why a double group is not a point group with extra labels.
    """

    cart: np.ndarray
    spin: np.ndarray
    name: str

    @property
    def bar(self) -> bool:
        """Whether this is the ``Ebar`` partner of the operation its name describes."""
        return bool(np.linalg.norm(self.spin + _spin_of(self.cart)) < 1.0e-6)

    def __mul__(self, other: "Element") -> "Element":
        cart = self.cart @ other.cart
        return Element(cart=cart, spin=self.spin @ other.spin, name=_named(cart))

    def full_name(self) -> str:
        return (self.name + "bar") if self.bar and self.name != "E" else (
            "Ebar" if self.bar else self.name)


def _parity_suffix(chi: np.ndarray, dim: int, cls: Optional[int],
                   labels: Tuple[str, str]) -> str:
    """``g``/``u`` or ``'``/``''`` when the character is really ``+-dim``, else nothing."""
    if cls is None:
        return ""
    value = chi[cls] / dim
    if abs(value.imag) > 1.0e-6 or abs(abs(value.real) - 1.0) > 1.0e-6:
        return ""
    return labels[0] if value.real > 0.0 else labels[1]


class DoubleGroup:
    """A molecule's full point double group, built by closure from its generators.

    Everything below — the elements, the classes, the characters, the names and the subduction
    onto an abelian label group — is derived from :attr:`elements`, so the printed tables are
    the tables of the operators the run actually uses.
    """

    def __init__(self, name: str, generators: Sequence[np.ndarray]) -> None:
        self.name = str(name)
        identity = Element(np.eye(3), np.eye(2, dtype=np.complex128), "E")
        # ⚠ Ebar is always a generator. For a group containing any twofold rotation or any
        # reflection it is a product of the spatial generators anyway (a C2 squares to it),
        # but C1 and Ci contain neither, and their double groups are still twice the size of
        # the point group -- which is the whole reason a spinor label needs a graded component.
        ebar = Element(np.eye(3), -np.eye(2, dtype=np.complex128), "E")
        seeds = ([identity, ebar]
                 + [Element(g, _spin_of(g), _named(g)) for g in generators])
        elements: List[Element] = []
        index: Dict[Tuple, int] = {}
        frontier = []
        for e in seeds:
            k = _key(e.cart, e.spin)
            if k not in index:
                index[k] = len(elements)
                elements.append(e)
                frontier.append(e)
        while frontier:
            a = frontier.pop()
            for b in list(elements):
                for product in (a * b, b * a):
                    k = _key(product.cart, product.spin)
                    if k not in index:
                        index[k] = len(elements)
                        elements.append(product)
                        frontier.append(product)
            if len(elements) > 200:
                raise ValueError("the closure of {} exceeded 200 elements; a generator is "
                                 "probably not of finite order".format(name))
        self.elements: Tuple[Element, ...] = tuple(elements)
        self._index = index
        self._table = self._multiplication_table()
        self.classes: Tuple[Tuple[int, ...], ...] = self._conjugacy_classes()
        self.characters, self.dimensions = self._character_table()
        self.irrep_names: Tuple[str, ...] = self._name_irreps()

    # -- structure ---------------------------------------------------------------------
    @property
    def order(self) -> int:
        return len(self.elements)

    @property
    def n_irreps(self) -> int:
        return len(self.classes)

    def position(self, element: Element) -> int:
        return self._index[_key(element.cart, element.spin)]

    def _multiplication_table(self) -> np.ndarray:
        n = self.order
        table = np.empty((n, n), dtype=np.int32)
        for i, a in enumerate(self.elements):
            for j, b in enumerate(self.elements):
                table[i, j] = self._index[_key(a.cart @ b.cart, a.spin @ b.spin)]
        return table

    def _inverse(self) -> np.ndarray:
        identity = self._index[_key(np.eye(3), np.eye(2, dtype=np.complex128))]
        return np.array([int(np.nonzero(self._table[i] == identity)[0][0])
                         for i in range(self.order)], dtype=np.int32)

    def _conjugacy_classes(self) -> Tuple[Tuple[int, ...], ...]:
        inverse = self._inverse()
        seen = np.zeros(self.order, dtype=bool)
        classes: List[Tuple[int, ...]] = []
        for a in range(self.order):
            if seen[a]:
                continue
            orbit = sorted({int(self._table[self._table[g, a], inverse[g]])
                            for g in range(self.order)})
            for m in orbit:
                seen[m] = True
            classes.append(tuple(orbit))
        return tuple(sorted(classes, key=self._class_sort_key))

    def _class_sort_key(self, members: Sequence[int]):
        """Print order: ``E``, ``Ebar``, then proper operations by turn, then improper ones,
        each followed by its barred partner."""
        unbarred = [m for m in members if not self.elements[m].bar]
        e = self.elements[unbarred[0] if unbarred else members[0]]
        proper = float(np.linalg.det(e.cart)) > 0.0
        axis, angle = canonical_axis_angle(e.cart if proper else -e.cart)
        return (0 if e.name == "E" else 1, 0 if proper else 1, round(angle, 6),
                1 if e.bar else 0, tuple(np.round(axis, 4)))

    def class_of(self, element: int) -> int:
        for k, members in enumerate(self.classes):
            if element in members:
                return k
        raise KeyError(element)

    def class_representative(self, k: int) -> int:
        """The member a class is named and tested by — an unbarred one where the class has one.

        ⚠ A class of a double group routinely contains both ``g`` and ``g Ebar`` (conjugating
        a twofold rotation by a perpendicular one inverts it, and an inverse differs from the
        original by ``Ebar``). Naming such a class after whichever member the closure happened
        to produce first would print ``2sigma(yz)bar`` for the class every table calls
        ``2sigma(yz)``.
        """
        members = self.classes[k]
        unbarred = [m for m in members if not self.elements[m].bar]
        candidates = unbarred or list(members)
        def turn(m):
            e = self.elements[m]
            if np.linalg.det(e.cart) > 0.0:
                return round(canonical_axis_angle(e.cart)[1], 6)
            # An improper element is named as a rotation-reflection, so the angle that orders
            # it is S's, not its proper part's: otherwise a class {S4, S4^3} prints as S4^3.
            angle = canonical_axis_angle(-e.cart)[1]
            return round((angle + np.pi) % (2.0 * np.pi), 6)
        return min(candidates, key=turn)

    def class_name(self, k: int) -> str:
        members = self.classes[k]
        name = self.elements[self.class_representative(k)].full_name()
        return name if len(members) == 1 else "{}{}".format(len(members), name)

    # -- characters --------------------------------------------------------------------
    def _class_coefficients(self) -> np.ndarray:
        """``c[i, j, k]``: how many ways one element of class ``k`` factorizes as (i)(j)."""
        n = self.n_irreps
        of_class = np.empty(self.order, dtype=np.int32)
        for k, members in enumerate(self.classes):
            for m in members:
                of_class[m] = k
        c = np.zeros((n, n, n))
        representative = [members[0] for members in self.classes]
        for i, ci in enumerate(self.classes):
            for j, cj in enumerate(self.classes):
                products = self._table[np.ix_(list(ci), list(cj))].ravel()
                counts = np.bincount(products, minlength=self.order)
                for k in range(n):
                    c[i, j, k] = counts[representative[k]]
        return c

    #: Seed of the weights that combine the class-sum matrices into one whose eigenvectors
    #: are the characters. ⚠ A *fixed* seed, not a draw: the eigenvectors come back in the
    #: order the eigensolver puts them, so an unseeded combination would permute the irreps
    #: from run to run -- and the irrep order is what the printed table, the subduction table
    #: and every reported label share. Small-integer weights leave accidental degeneracies
    #: (D2h has one), which is why the weights are generic reals rather than the primes.
    _WEIGHT_SEED = 20260823
    _WEIGHT_TRIES = 8

    def _character_table(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n_irreps
        sizes = np.array([len(members) for members in self.classes], dtype=float)
        c = self._class_coefficients()
        # K_i K_j = sum_k c[i, j, k] K_k, so the vector omega_k = |C_k| chi(C_k) / dim is a
        # right eigenvector of c[i] with eigenvalue omega_i. ⚠ Of c[i], not of its transpose:
        # the transpose has just as simple a spectrum and gives the conjugate irrep.
        matrices = [c[i] for i in range(n)]
        identity_class = self.class_of(
            self._index[_key(np.eye(3), np.eye(2, dtype=np.complex128))])
        rng = np.random.default_rng(self._WEIGHT_SEED)
        for _ in range(self._WEIGHT_TRIES):
            weights = rng.random(n)
            mix = sum(w * m for w, m in zip(weights, matrices))
            values, vectors = np.linalg.eig(mix)
            if np.unique(np.round(values, 6)).size != n:
                continue
            omega = np.empty((n, n), dtype=np.complex128)
            for r in range(n):
                v = vectors[:, r]
                v = v / v[identity_class]
                omega[r] = v
            dims = np.sqrt(self.order / np.sum(np.abs(omega) ** 2 / sizes[None, :], axis=1))
            chi = omega * dims[:, None] / sizes[None, :]
            order = sorted(range(n), key=lambda r: (
                int(np.rint(np.real(chi[r, identity_class]))) != int(np.rint(np.real(dims[r]))),
                self._irrep_sort_key(chi[r], dims[r])))
            chi, dims = chi[order], dims[order]
            if self._orthogonal(chi, sizes):
                return np.round(chi, 12), np.rint(np.real(dims)).astype(int)
        raise RuntimeError(
            "the character table of {} could not be separated into {} irreps; the class-sum "
            "matrices did not have a simple spectrum under any of the fixed weightings"
            .format(self.name, n))

    def _irrep_sort_key(self, chi: np.ndarray, dim: float):
        ebar = self._ebar_class()
        fermion = 1 if np.real(chi[ebar]) < 0.0 else 0
        # Descending characters, so the totally symmetric irrep is first and ``A`` precedes
        # ``B``, ``g`` precedes ``u`` -- the order every published table prints.
        return (fermion, int(np.rint(np.real(dim))),
                tuple(-np.round(np.real(chi), 6)), tuple(-np.round(np.imag(chi), 6)))

    def _orthogonal(self, chi: np.ndarray, sizes: np.ndarray) -> bool:
        gram = (chi * sizes[None, :]) @ np.conj(chi).T / self.order
        return bool(np.allclose(gram, np.eye(chi.shape[0]), atol=1.0e-8))

    def _ebar_class(self) -> int:
        return self.class_of(self._index[_key(np.eye(3), -np.eye(2, dtype=np.complex128))])

    def is_fermion(self, irrep: int) -> bool:
        """``chi(Ebar) = -dim``: a double-valued (spinor) irrep."""
        return bool(np.real(self.characters[irrep, self._ebar_class()]) < 0.0)

    # -- naming ------------------------------------------------------------------------
    def _principal(self) -> Tuple[int, Optional[int]]:
        """``(order of the principal proper axis, class index of that rotation)``, ``z`` only.

        ⚠ The principal axis must be ``z``, for the same reason the abelian labels need it: the
        spinor basis quantizes spin along ``z`` and the molecule is never reoriented.
        """
        best, best_class = 1, None
        for k in range(self.n_irreps):
            e = self.elements[self.class_representative(k)]
            if np.linalg.det(e.cart) < 0.0 or e.bar:
                continue
            axis, angle = canonical_axis_angle(e.cart)
            if angle < 1.0e-8 or not np.allclose(axis, [0.0, 0.0, 1.0], atol=1.0e-8):
                continue
            n = int(np.rint(2.0 * np.pi / angle))
            if abs(2.0 * np.pi / angle - n) < 1.0e-6 and n > best:
                best, best_class = n, k
        return best, best_class

    def _improper_principal(self) -> Tuple[int, Optional[int]]:
        """``(order, class)`` of the highest rotation-reflection ``S_m`` about ``z``.

        Mulliken's rules read ``A`` versus ``B`` off the principal **proper** rotation
        everywhere except the ``S_2n`` and ``D_nd`` families, where the improper axis is the
        higher one and is what the published labels use; :meth:`_name_irreps` is where that
        exception is taken, and only there.
        """
        best, best_class = 0, None
        for k in range(self.n_irreps):
            e = self.elements[self.class_representative(k)]
            if np.linalg.det(e.cart) > 0.0 or e.bar:
                continue
            axis, angle = canonical_axis_angle(-e.cart)
            phi = (angle + np.pi) % (2.0 * np.pi)
            if phi < 1.0e-8 or not np.allclose(axis, [0.0, 0.0, 1.0], atol=1.0e-8):
                continue
            for m in range(3, 2 * MAX_ROTATION_ORDER + 1):
                k_turn = phi * m / (2.0 * np.pi)
                if abs(k_turn - np.rint(k_turn)) < 1.0e-6 and np.rint(k_turn) == 1:
                    if m > best:
                        best, best_class = m, k
                    break
        return best, best_class

    def _find_class(self, predicate) -> Optional[int]:
        for k in range(self.n_irreps):
            e = self.elements[self.class_representative(k)]
            if not e.bar and predicate(e):
                return k
        return None

    def _axis_class(self, axis: Sequence[float]) -> Optional[int]:
        """The class of the twofold **proper** rotation about a given Cartesian axis."""
        target = np.asarray(axis, dtype=float)
        return self._find_class(
            lambda e: np.linalg.det(e.cart) > 0.0
            and abs(canonical_axis_angle(e.cart)[1] - np.pi) < 1.0e-6
            and np.allclose(np.abs(canonical_axis_angle(e.cart)[0]), np.abs(target),
                            atol=1.0e-8))

    def _omega_of(self, irrep: int, n: int) -> Optional[float]:
        """``|omega|`` of a double-valued irrep from its characters on the principal axis.

        The cyclic double subgroup generated by ``C_n(z)`` has order ``2n``, and restricting
        the irrep to it decomposes into characters ``exp(-2 pi i omega / n)``. A discrete
        Fourier transform over that subgroup returns the multiset of ``omega``; a
        double-valued irrep has half-integer ones, and the smallest ``|omega|`` folded into
        ``(0, n/2]`` is what names it.

        ⚠ **The folding can collide, and that is a property of the group rather than a defect
        of the rule.** Two fermion irreps of ``D3h`` restrict to the same ``|omega| = 1/2`` on
        the threefold axis and differ only on ``sigma(xy)``, where their characters are
        ``+-i`` and no parity suffix exists; they come out as ``1E1/2`` and ``2E1/2``, the same
        spelling the abelian vocabulary uses for a conjugate pair it cannot separate either.
        """
        if n < 2:
            return None
        generator = self._find_class(
            lambda e: np.linalg.det(e.cart) > 0.0
            and abs(canonical_axis_angle(e.cart)[1] - 2.0 * np.pi / n) < 1.0e-6
            and np.allclose(canonical_axis_angle(e.cart)[0], [0.0, 0.0, 1.0], atol=1.0e-8))
        if generator is None:
            return None
        g = self.class_representative(generator)
        powers, current = [], self._index[_key(np.eye(3), np.eye(2, dtype=np.complex128))]
        for _ in range(2 * n):
            powers.append(current)
            current = int(self._table[current, g])
        chi = np.array([self.characters[irrep, self.class_of(p)] for p in powers])
        weights = np.array([np.mean(chi * np.exp(2j * np.pi * m * np.arange(2 * n) / (2 * n)))
                            for m in range(2 * n)])
        omegas = []
        for m in range(2 * n):
            if abs(weights[m]) <= 1.0e-6:
                continue
            two_omega = (-m) % (2 * n)
            if two_omega > n:
                two_omega = 2 * n - two_omega
            omegas.append(two_omega / 2.0)
        return min(omegas) if omegas else None

    def _name_irreps(self) -> Tuple[str, ...]:
        n_axis, principal = self._principal()
        improper_order, improper = self._improper_principal()
        z, y, x = [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]
        inversion = self._find_class(lambda e: np.allclose(e.cart, -np.eye(3), atol=1e-8))
        sigma_h = self._find_class(
            lambda e: np.allclose(e.cart, np.diag([1.0, 1.0, -1.0]), atol=1.0e-8))
        # ⚠ The S_2n exception (D2d, S4, ...): with no horizontal plane and an improper axis of
        # twice the proper one's order, the published A/B distinction is read off S_2n.
        grading, grading_order = principal, n_axis
        if (sigma_h is None and improper is not None and improper_order >= 4
                and improper_order == 2 * n_axis):
            grading, grading_order = improper, improper_order
        perpendicular = self._axis_class(x)
        vertical = self._find_class(
            lambda e: np.linalg.det(e.cart) < 0.0
            and np.allclose(e.cart, np.diag([1.0, -1.0, 1.0]), atol=1.0e-8))
        secondary = perpendicular if perpendicular is not None else vertical
        # The D2 family has three equivalent twofold axes and its own published spelling
        # (A, B1, B2, B3 by which axis is symmetric) rather than an A/B plus a subscript.
        d2_axes = None
        if n_axis == 2:
            found = [self._axis_class(a) for a in (z, y, x)]
            if all(k is not None for k in found) and len(set(found)) == 3:
                d2_axes = found
                secondary = None

        names: List[str] = []
        indices: List[Optional[int]] = []
        for r in range(self.n_irreps):
            chi, dim = self.characters[r], int(self.dimensions[r])
            index = None
            if self.is_fermion(r):
                omega = self._omega_of(r, n_axis)
                # Without a principal axis (C1, Ci, Cs) there is nothing to read an omega off,
                # and the single fermion irrep is E1/2 by the convention the abelian
                # vocabulary already uses.
                base = "E{:g}/2".format(2.0 * omega) if omega is not None else "E1/2"
            elif dim == 1 and d2_axes is not None:
                values = [np.real(chi[k]) for k in d2_axes]
                base = "A" if all(v > 0.0 for v in values) else "B{}".format(
                    1 + int(np.argmax(values)))
            elif dim == 1:
                value = chi[grading] if grading is not None else 1.0 + 0.0j
                if abs(np.imag(value)) > 1.0e-6:
                    base = "E"            # a complex-conjugate pair, printed 1E / 2E
                else:
                    base = "A" if np.real(value) > 0.0 else "B"
                    if secondary is not None:
                        base += "1" if np.real(chi[secondary]) > 0.0 else "2"
            elif dim == 2:
                base = "E"
                if grading_order > 2 and grading is not None:
                    m = np.arccos(np.clip(np.real(chi[grading]) / 2.0, -1.0, 1.0)) \
                        * grading_order / (2.0 * np.pi)
                    if abs(m - np.rint(m)) < 1.0e-6:
                        index = int(np.rint(m))
            else:
                base = "T" if dim == 3 else "G{}".format(dim)
            # ⚠ A g/u or '/'' suffix means nothing unless the character is really +-dim: the
            # fermion irreps of a group whose sigma(xy) has order four carry +-i there, and
            # forcing a parity onto them would print a distinction that does not exist.
            base += _parity_suffix(chi, dim, inversion, ("g", "u"))
            if inversion is None:
                base += _parity_suffix(chi, dim, sigma_h, ("\'", "\'\'"))
            names.append(base)
            indices.append(index)
        return tuple(self._disambiguate(names, indices))

    @staticmethod
    def _disambiguate(names: Sequence[str], indices: Sequence[Optional[int]]) -> List[str]:
        """Separate irreps the naming rules give the same spelling.

        Two ways, in order: the ``E1``/``E2`` index of the principal-axis character where
        every colliding irrep has one (``C6v``), and otherwise the ``1E``/``2E`` prefix a
        complex-conjugate pair gets — the same spelling the abelian vocabulary uses, assigned
        in the table's own order so it is reproducible.
        """
        groups: Dict[str, List[int]] = {}
        for r, name in enumerate(names):
            groups.setdefault(name, []).append(r)
        out = list(names)
        for name, members in groups.items():
            if len(members) == 1:
                continue
            picked = [indices[r] for r in members]
            if all(p is not None for p in picked) and len(set(picked)) == len(picked):
                for r in members:
                    head = name[0]
                    out[r] = "{}{}{}".format(head, indices[r], name[1:])
                continue
            for k, r in enumerate(members, start=1):
                out[r] = "{}{}".format(k, name)
        return out

    def irrep(self, name) -> int:
        """Index of an irrep by name, case-insensitively."""
        text = str(name).strip().lower()
        for r, nm in enumerate(self.irrep_names):
            if nm.lower() == text:
                return r
        raise ValueError("{!r} is not an irrep of {}; it has {}".format(
            name, self.name, ", ".join(self.irrep_names)))

    # -- subduction --------------------------------------------------------------------
    def abelian_element(self, group: Group, element: Sequence[int]) -> Optional[int]:
        """Which element of this group an abelian label group's element is, or ``None``.

        The abelian element is ``(spatial bits, Ebar flag)``; its spatial matrix and its spin
        factor are :mod:`kuiva.symm.operators`' own, so the identification is by the matrices
        rather than by a name, and a mismatch of convention between the two modules would
        show up as a missing element rather than as a wrong label.
        """
        spatial, ebar = group.element_spatial(element)
        r, p = spatial
        cart = np.eye(3)
        if r:
            cart = cart @ np.diag([-1.0, -1.0, 1.0])
        if p:
            cart = -cart
        spin = SPIN_FACTOR[spatial] * (-1.0 if ebar else 1.0)
        return self._index.get(_key(cart, spin))

    def subduction(self, group: Group) -> "Dict[int, Dict[Tuple[int, ...], int]]":
        """``{irrep: {abelian label: multiplicity}}`` — the correspondence table, computed.

        Restricting a full-group irrep to the abelian subgroup and projecting onto the
        abelian characters. ⚠ This is the row a user needs to connect a per-irrep
        ``n_states`` request to the physical multiplet it selects: where the two groups
        differ, a physically degenerate manifold carries **several** abelian labels, and the
        abelian count alone cannot see that it is cutting one.
        """
        members = []
        for element in group.elements():
            position = self.abelian_element(group, element)
            if position is None:
                raise ValueError(
                    "{} is not a subgroup of {} as they are represented here: the element {} "
                    "has no counterpart. The two modules must agree on the spin branch and on "
                    "the frame".format(group.name, self.name, group.element_name(element)))
            members.append((element, position))
        out: Dict[int, Dict[Tuple[int, ...], int]] = {}
        for r in range(self.n_irreps):
            decomposition: Dict[Tuple[int, ...], int] = {}
            for label in group.labels():
                total = sum(np.conj(group.character(label, element))
                            * self.characters[r, self.class_of(position)]
                            for element, position in members)
                multiplicity = total / group.order
                if abs(multiplicity.imag) > 1.0e-6 or abs(
                        multiplicity.real - np.rint(multiplicity.real)) > 1.0e-6:
                    raise RuntimeError(
                        "the subduction of {} onto {} is not an integer combination ({}); one "
                        "of the two character tables is wrong"
                        .format(self.irrep_names[r], group.name, multiplicity))
                if int(np.rint(multiplicity.real)):
                    decomposition[label] = int(np.rint(multiplicity.real))
            out[r] = decomposition
        return out

    def __repr__(self) -> str:
        return "DoubleGroup({}, order={}, {} irreps)".format(
            self.name, self.order, self.n_irreps)


_CACHE: Dict[str, DoubleGroup] = {}


def double_group(name: str) -> DoubleGroup:
    """The double group of a Schoenflies point-group name, built once and cached."""
    canonical, generators = _spatial_generators(name)
    if canonical not in _CACHE:
        _CACHE[canonical] = DoubleGroup(canonical, generators)
    return _CACHE[canonical]


def _spatial_closure(generators: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Every **spatial** operation the generators produce — 3x3 matrices, no spin factor.

    Detection only ever asks whether the geometry has an operation, which is a question about
    the 3x3 matrix alone. Closing the point group rather than the double group is what keeps
    the detector from building a character table for every candidate it rejects.
    """
    elements: List[np.ndarray] = [np.eye(3)]
    seen = {_cart_key(np.eye(3))}
    frontier = [np.eye(3)]
    while frontier:
        a = frontier.pop()
        for b in list(elements):
            for product in (a @ b, b @ a):
                key = _cart_key(product)
                if key not in seen:
                    seen.add(key)
                    elements.append(product)
                    frontier.append(product)
        for g in generators:
            for product in (a @ g, g @ a):
                key = _cart_key(product)
                if key not in seen:
                    seen.add(key)
                    elements.append(product)
                    frontier.append(product)
        if len(elements) > 100:
            raise ValueError("a generator is not of finite order")
    return elements


def expected_order(name: str) -> int:
    """Order of a group's **double** group, from its name alone.

    The candidate list is ordered by this rather than by building every group and asking:
    ordering is the only thing the order is needed for before a group is chosen, and building
    a full double group means a multiplication table, a class decomposition and a character
    table for a candidate that is usually rejected on its first operation.
    """
    canonical, _ = _spatial_generators(name)
    if canonical == "C1":
        return 2
    if canonical in ("Ci", "Cs"):
        return 4
    n = int("".join(ch for ch in canonical if ch.isdigit()))
    if canonical.startswith("D"):
        return 8 * n if canonical[-1] in "hd" else 4 * n
    return 4 * n if canonical[-1] in ("h", "v") else 2 * n


def candidate_groups(max_order: int = MAX_ROTATION_ORDER) -> Tuple[str, ...]:
    """Every group the detector tests, largest first."""
    names = ["C1", "Ci", "Cs"]
    for n in range(2, int(max_order) + 1):
        names += ["C{}".format(n), "C{}h".format(n), "C{}v".format(n),
                  "D{}".format(n), "D{}h".format(n), "D{}d".format(n)]
    # ⚠ Ties are broken toward the group with a horizontal mirror. ``D6d`` and ``D6h`` are
    # both of order 48 and a linear molecule has both, but only ``D6h`` contains the
    # inversion -- and a classification group that does not contain the abelian label group
    # has no subduction onto it at all.
    return tuple(sorted(names, key=lambda nm: (-expected_order(nm),
                                               0 if nm.endswith("h") else 1, nm)))


def is_linear(layout, *, tol: float = 1.0e-4) -> bool:
    """Whether every nucleus lies on one line — which is where a finite group is a truncation."""
    coords = np.asarray(layout.coords_bohr, dtype=float)
    if coords.shape[0] < 3:
        return True
    shifted = coords - coords.mean(axis=0)
    return bool(np.linalg.svd(shifted, compute_uv=False)[1] < tol)


def has_group(layout, name: str, *, tol: float = DEFAULT_ATOM_TOL) -> bool:
    """Whether the geometry has every element of ``name`` **in its input frame**."""
    _, generators = _spatial_generators(name)
    return all(atom_permutation_general(layout, cart, tol=tol) is not None
               for cart in _spatial_closure(generators))


def detect_point_group(layout, *, max_order: int = MAX_ROTATION_ORDER,
                       tol: float = DEFAULT_ATOM_TOL, require=()) -> str:
    """The largest supported group the geometry has, tested in the **input frame**.

    ⚠ Frame-dependent by design, exactly as :func:`kuiva.symm.operators.detect_operations` is:
    a molecule whose threefold axis is ``x`` detects less than it has, and the fix is to orient
    the input rather than to let the program move the geometry (which would carry the gauge
    origin and every property operator with it).

    ``require`` is a list of 3x3 matrices the group must contain — the abelian label group's
    own operations, so that the correspondence table between the two exists.
    """
    required = [_cart_key(np.asarray(cart, dtype=float)) for cart in require]
    for name in candidate_groups(max_order):
        if not has_group(layout, name, tol=tol):
            continue
        if required:
            _, generators = _spatial_generators(name)
            present = {_cart_key(cart) for cart in _spatial_closure(generators)}
            if not all(key in present for key in required):
                # ⚠ Not a candidate however large it is: without the abelian label group
                # inside it there is no correspondence table, and the labels of the two
                # groups would be two vocabularies rather than one.
                continue
        return name
    return "C1"


__all__ = ["DEFAULT_PROJECTION_TOL", "DoubleGroup", "Element", "MAX_ROTATION_ORDER",
           "candidate_groups", "detect_point_group", "double_group", "expected_order",
           "has_group", "is_linear"]
