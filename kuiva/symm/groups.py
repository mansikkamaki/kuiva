"""Abelian double groups: the label vocabulary, the groups, and their character tables.

What a label *is*
-----------------
An abelian group's irreducible representations are one-dimensional, so an orbital that is an
eigenvector of every group operation carries one complex number per operation — its character.
Writing the group as a direct product of cyclic factors ``<g_1> x ... x <g_k>`` of orders
``n_1 ... n_k``, every character is fixed by ``k`` integers::

    chi(g_j) = exp(2 pi i m_j / n_j),      m_j in [0, n_j)

so **a label is a tuple of integers with per-component moduli**, composition is componentwise
addition modulo ``n_j``, and complex conjugation — which is what time reversal does to a
character — is componentwise negation. That is the whole vocabulary; :class:`Group` below is
the descriptor that carries it, and nothing else in the program defines a second one.

⚠ **The group here is the DOUBLE group, and its generators are not the spatial operations.**
A spatial ``C2`` squares to the identity, but the ``2 pi`` rotation it squares to acts on a
spinor as ``-1``: in the double group ``C2^2 = Ebar`` and ``C2`` has order **four**, not two.
That is exactly why the label vocabulary needs moduli at all rather than being a string of
signs, and it is why a fermion irrep is recognized by an *odd* exponent on the graded
generator: ``chi(Ebar) = (-1)^m`` there, and ``chi(Ebar) = -1`` is the definition of a
fermion (double-valued, "spinor") irrep.

Which groups are here, and why not more
---------------------------------------
Only groups whose **double** group is abelian have one-dimensional fermion irreps, and only
those can label a spinor with a number. In the D2h chain that is ``C1, Ci, C2, Cs, C2h`` —
the double groups of ``C2v``, ``D2`` and ``D2h`` are non-abelian (they carry a single
two-dimensional fermion irrep), so a larger request is **reduced to the largest subgroup that
does have abelian fermion irreps**, and the reduction is reported rather than hidden.

⚠ **The spin-active generator must be about z**, because the spinor basis quantizes spin
along z: a rotation about x or y mixes the two members of a Kramers pair, so the pair would
have to be re-mixed before its members carried labels at all. Kuiva does not rotate the
molecule (that would move the gauge origin and every property operator fixed with it) and
does not restate the spinor convention, so a symmetry axis that is not z is **reported and
not used** — the fix is to orient the input geometry, and the message says so. The groups
below are therefore ``C2(z)``, ``Cs(xy)``, ``C2h(z)`` and the two axis-free ones.

Spatial operations, and how an element is one
---------------------------------------------
Every element of every group here is ``(C2(z))^r (i)^p (Ebar)^e`` with ``r, p, e`` in
``{0, 1}``, so a spatial operation is two bits and an element is those two bits plus the
``Ebar`` flag. The four spatial operations are ``E``, ``C2(z)``, ``i`` and
``sigma(xy) = i C2(z)``, and their names in this module are **lab-frame geometry**, never a
Schoenflies label whose axis the reader has to guess. That is the point of printing the
character table at all: two programs agreeing on "C2h, Bu" and disagreeing on which axis is
``z`` produce different numbers and no error message.

The spin factor
---------------
For a rotation by ``theta`` about ``n``, ``D^(1/2) = exp(-i theta n.sigma / 2)`` with
``theta`` taken in ``[0, 2 pi)`` — the branch choice that *defines* which of the two
double-group elements above a spatial operation is called ``C2(z)`` rather than
``C2(z) Ebar``. About ``z`` it is ``diag(-i, +i)``: the unbarred (spin-up) partner of a
Kramers pair picks up ``-i`` and the barred (spin-down) one ``+i``, i.e. exponent shifts of
``-1`` and ``+1`` modulo 4. Inversion does not act on spin at all. Those two facts are the
whole of :attr:`Generator.spin_shift`, and they are checked numerically against the AO
operator matrices rather than trusted.

References
----------
* Double-group character tables: G. F. Koster, J. O. Dimmock, R. G. Wheeler, H. Statz,
  "Properties of the Thirty-Two Point Groups", MIT Press (1963); S. L. Altmann, P. Herzig,
  "Point-Group Theory Tables", Clarendon Press (1994).
* The ``exp(-i theta n.sigma / 2)`` spinor representation and the ``E``/``Ebar`` distinction:
  E. P. Wigner, "Group Theory and its Application to the Quantum Mechanics of Atomic
  Spectra", Academic Press (1959), ch. 15.
* Abelian double groups as the working symmetry of two-component/four-component molecular
  codes: T. Saue, H. J. Aa. Jensen, J. Chem. Phys. 111, 6211 (1999),
  doi:10.1063/1.479958.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Spatial operations, as ``(rotation about z, inversion)`` bit pairs, with their lab-frame
#: names. ``sigma(xy)`` is the product of the other two and is named as the reflection it is.
SPATIAL_NAMES: Dict[Tuple[int, int], str] = {
    (0, 0): "E",
    (1, 0): "C2(z)",
    (0, 1): "i",
    (1, 1): "sigma(xy)",
}


@dataclass(frozen=True)
class Generator:
    """One cyclic factor of an abelian double group.

    Attributes
    ----------
    name : str
        Lab-frame name of the generating operation (``"C2(z)"``, ``"i"``, ``"sigma(xy)"``),
        or ``"Ebar"`` for the ``2 pi`` rotation itself, which generates the whole group of
        ``C1``.
    modulus : int
        The generator's order **in the double group** — 4 for an operation that squares to
        ``Ebar`` (every ``C2`` and every reflection), 2 for inversion and for ``Ebar``.
    spin_shift : (int, int)
        Exponent shift the spin-1/2 factor contributes to the ``(unbarred, barred)`` member
        of a Kramers pair, modulo :attr:`modulus`. This is where the spinor convention of
        :mod:`kuiva.spinor.expand` enters the labels and the only place it does.
    graded : bool
        Whether ``chi(Ebar) = (-1) ** m`` reads off this component. Exactly one generator of
        a group is graded, and a label is a **fermion** label when that component is odd.
    spatial : (int, int)
        The ``(C2(z), i)`` bits this generator contributes per unit exponent; ``(0, 0)`` for
        ``Ebar``, which has no spatial action.
    ebar_per_two : bool
        Whether exponent 2 of this generator *is* ``Ebar`` (true for the modulus-4 ones).
    """

    name: str
    modulus: int
    spin_shift: Tuple[int, int]
    graded: bool
    spatial: Tuple[int, int]
    ebar_per_two: bool


#: ``Ebar`` itself: no spatial action, and it multiplies every spinor by ``-1``.
GEN_EBAR = Generator("Ebar", 2, (1, 1), True, (0, 0), False)
#: Inversion: order 2 in the double group (``i^2 = E`` exactly), and inert on spin.
GEN_INVERSION = Generator("i", 2, (0, 0), False, (0, 1), False)
#: A two-fold rotation about ``z``: order 4, and ``diag(-i, +i)`` on ``(alpha, beta)``.
GEN_C2Z = Generator("C2(z)", 4, (-1, 1), True, (1, 0), True)
#: The reflection in the ``xy`` plane, ``i C2(z)``: the same spin factor, spatial bits both set.
GEN_SIGMA_XY = Generator("sigma(xy)", 4, (-1, 1), True, (1, 1), True)


@dataclass(frozen=True)
class Group:
    """An abelian double group: its generators, its irrep names, and its characters.

    The instance is the label descriptor of :mod:`kuiva.symm` — components, moduli,
    composition and conjugation all come from :attr:`generators`, so widening the vocabulary
    (an axial ``2 m_j`` label, say) is a new :class:`Generator`, not a new code path.
    """

    name: str
    generators: Tuple[Generator, ...]
    #: ``{label: irrep name}``. Stored data (see the module references), and checked against
    #: the computed characters by the suite rather than trusted.
    names: Dict[Tuple[int, ...], str]

    # -- the vocabulary ----------------------------------------------------------------
    @property
    def width(self) -> int:
        return len(self.generators)

    @property
    def moduli(self) -> Tuple[int, ...]:
        return tuple(g.modulus for g in self.generators)

    @property
    def graded_component(self) -> int:
        """Index of the component whose parity is ``chi(Ebar)``."""
        for j, g in enumerate(self.generators):
            if g.graded:
                return j
        raise RuntimeError("group {} has no graded generator".format(self.name))

    @property
    def order(self) -> int:
        return int(np.prod(self.moduli))

    def identity(self) -> Tuple[int, ...]:
        return tuple([0] * self.width)

    def compose(self, a: Sequence[int], b: Sequence[int]) -> Tuple[int, ...]:
        """The group product of two labels: componentwise addition modulo the moduli."""
        return tuple(int(x + y) % n for x, y, n in zip(a, b, self.moduli))

    def conjugate(self, a: Sequence[int]) -> Tuple[int, ...]:
        """The complex-conjugate character — what time reversal does to a label."""
        return tuple((-int(x)) % n for x, n in zip(a, self.moduli))

    def is_fermion(self, a: Sequence[int]) -> bool:
        """``chi(Ebar) = -1``: the label of a spinor, not of a spatial orbital."""
        return bool(int(a[self.graded_component]) % 2)

    def labels(self, *, fermion: Optional[bool] = None) -> List[Tuple[int, ...]]:
        """Every irrep label, in the canonical order: **boson rows first, then fermion**.

        One order, used by the character table, by the sector reporting and by every refusal
        that lists what is available, so a reader comparing two of them is comparing the same
        sequence.
        """
        out: List[Tuple[int, ...]] = [()]
        for n in self.moduli:
            out = [t + (m,) for t in out for m in range(n)]
        out.sort(key=lambda t: (self.is_fermion(t), t))
        if fermion is None:
            return out
        return [t for t in out if self.is_fermion(t) == fermion]

    # -- names -------------------------------------------------------------------------
    def irrep_name(self, a: Sequence[int]) -> str:
        """The one spelling of an irrep — the same bytes in the table, in a selection
        request, in a refusal and in the state table."""
        key = tuple(int(x) % n for x, n in zip(a, self.moduli))
        try:
            return self.names[key]
        except KeyError:
            raise KeyError("{} is not a label of {} (moduli {})"
                           .format(tuple(a), self.name, self.moduli))

    def label_of(self, name) -> Tuple[int, ...]:
        """Resolve an irrep name (or an already-resolved label tuple) to a label.

        Names match case-insensitively, because ``"1e1/2g"`` typed at an interactive prompt
        and ``"1E1/2g"`` printed in the character table are the same request.
        """
        if isinstance(name, (tuple, list, np.ndarray)):
            key = tuple(int(x) for x in name)
            if len(key) != self.width or any(not 0 <= x < n for x, n in zip(key, self.moduli)):
                raise ValueError("{} is not a label of {} (moduli {})"
                                 .format(key, self.name, self.moduli))
            return key
        text = str(name).strip().lower()
        for label, nm in self.names.items():
            if nm.lower() == text:
                return label
        raise ValueError("{!r} is not an irrep of {}; it has {}".format(
            name, self.name, ", ".join(self.names[t] for t in self.labels())))

    # -- elements and characters -------------------------------------------------------
    #: Print order of the spatial operations: identity, rotation, inversion, reflection.
    _SPATIAL_ORDER = ((0, 0), (1, 0), (0, 1), (1, 1))

    def elements(self) -> List[Tuple[int, ...]]:
        """Every group element as an exponent tuple, in the order the table prints them.

        Unbarred operations first in the geometric order ``E, C2(z), i, sigma(xy)``, then the
        same four times ``Ebar`` — so the left half of a printed table is the spatial group
        and the right half is what the ``2 pi`` rotation does to it.
        """
        out: List[Tuple[int, ...]] = [()]
        for n in self.moduli:
            out = [t + (e,) for t in out for e in range(n)]
        order = {s: k for k, s in enumerate(self._SPATIAL_ORDER)}
        out.sort(key=lambda e: (self.element_spatial(e)[1], order[self.element_spatial(e)[0]]))
        return out

    def element_spatial(self, element: Sequence[int]) -> Tuple[Tuple[int, int], bool]:
        """``((C2(z) bit, inversion bit), carries Ebar)`` for a group element.

        This is what makes the printed table checkable: the AO operator matrices know only
        the four spatial operations, and every one of the (up to eight) group elements is one
        of them, optionally times ``-1`` on spinors.
        """
        r = p = 0
        ebar = 0
        for e, g in zip(element, self.generators):
            e = int(e)
            if g.ebar_per_two:
                ebar ^= (e // 2) & 1
                e &= 1
            elif g.spatial == (0, 0):        # Ebar itself
                ebar ^= e & 1
                continue
            r ^= (g.spatial[0] * e) & 1
            p ^= (g.spatial[1] * e) & 1
        return (r, p), bool(ebar)

    def element_name(self, element: Sequence[int]) -> str:
        """Lab-frame name of an element: ``"C2(z)"``, ``"sigma(xy)bar"``, ..."""
        spatial, ebar = self.element_spatial(element)
        base = SPATIAL_NAMES[spatial]
        if not ebar:
            return base
        return "Ebar" if base == "E" else base + "bar"

    def character(self, label: Sequence[int], element: Sequence[int]) -> complex:
        """``chi_label(element)`` — a fourth root of unity for every group here."""
        phase = 0.0
        for m, e, n in zip(label, element, self.moduli):
            phase += 2.0 * np.pi * (int(m) * int(e)) / n
        return complex(np.round(np.exp(1j * phase), 12))

    # -- spinor composition ------------------------------------------------------------
    def spinor_labels(self, scalar: np.ndarray) -> np.ndarray:
        """Spinor labels from scalar-orbital labels, in the interleaved Kramers ordering.

        ``scalar`` is ``(n_scalar, width)``; the result is ``(2 * n_scalar, width)`` with row
        ``2p`` the unbarred partner of scalar orbital ``p`` and ``2p+1`` the barred one —
        :mod:`kuiva.spinor.expand`'s convention, consumed here and not restated.

        The two rows of a pair are **conjugate labels** (they must be: time reversal is
        antiunitary and commutes with every spatial operation), which is asserted here rather
        than assumed, because it is the property every downstream Kramers-pair rule rests on.
        """
        scalar = np.atleast_2d(np.asarray(scalar, dtype=int))
        if scalar.shape[1] != self.width:
            raise ValueError("scalar labels are {} wide; {} has {} components"
                             .format(scalar.shape[1], self.name, self.width))
        shift = np.array([g.spin_shift for g in self.generators], dtype=int)   # (width, 2)
        mod = np.asarray(self.moduli, dtype=int)
        out = np.empty((2 * scalar.shape[0], self.width), dtype=int)
        out[0::2] = (scalar + shift[:, 0][None, :]) % mod[None, :]
        out[1::2] = (scalar + shift[:, 1][None, :]) % mod[None, :]
        for row in out:
            if not self.is_fermion(row):
                raise RuntimeError(
                    "the spin factor of {} left a boson label {} on a spinor; the graded "
                    "generator or its spin shift is wrong".format(self.name, tuple(row)))
        conj = np.array([self.conjugate(r) for r in out[0::2]], dtype=int)
        if not np.array_equal(conj, out[1::2]):
            raise RuntimeError(
                "the two members of a Kramers pair must carry conjugate labels in {}; got "
                "{} and {}".format(self.name, out[0], out[1]))
        return out

    def __repr__(self) -> str:
        return "Group({}, generators={}, order={})".format(
            self.name, [g.name for g in self.generators], self.order)


def _names(pairs: Iterable[Tuple[Tuple[int, ...], str]]) -> Dict[Tuple[int, ...], str]:
    return {tuple(k): v for k, v in pairs}


#: The five label groups. Names are Mulliken for the boson irreps and the Koster-style
#: ``E1/2`` spelling for the fermion ones, in ASCII throughout (the output stream is ASCII).
C1 = Group("C1", (GEN_EBAR,), _names([((0,), "A"), ((1,), "E1/2")]))

CI = Group("Ci", (GEN_EBAR, GEN_INVERSION), _names([
    ((0, 0), "Ag"), ((0, 1), "Au"), ((1, 0), "E1/2g"), ((1, 1), "E1/2u")]))

C2Z = Group("C2(z)", (GEN_C2Z,), _names([
    ((0,), "A"), ((1,), "1E1/2"), ((2,), "B"), ((3,), "2E1/2")]))

CS_XY = Group("Cs(xy)", (GEN_SIGMA_XY,), _names([
    ((0,), "A'"), ((1,), "1E1/2"), ((2,), "A''"), ((3,), "2E1/2")]))

C2H_Z = Group("C2h(z)", (GEN_C2Z, GEN_INVERSION), _names([
    ((0, 0), "Ag"), ((0, 1), "Au"), ((2, 0), "Bg"), ((2, 1), "Bu"),
    ((1, 0), "1E1/2g"), ((1, 1), "1E1/2u"), ((3, 0), "2E1/2g"), ((3, 1), "2E1/2u")]))

#: Every group this module can label with, by name.
GROUPS: Dict[str, Group] = {g.name: g for g in (C1, CI, C2Z, CS_XY, C2H_Z)}

#: What a larger request reduces to. ⚠ The value is the **largest subgroup whose double group
#: is abelian**, which is what one-dimensional fermion irreps require; the key's own double
#: group carries a two-dimensional fermion irrep and adapting to it would be non-abelian
#: symmetry adaptation, which is out of scope. The reduction is reported at the point of
#: selection, never applied silently.
REDUCTION: Dict[str, str] = {
    "D2h": "C2h(z)",
    "D2": "C2(z)",
    "C2v": "C2(z)",
    "C2h": "C2h(z)",
    "C2": "C2(z)",
    "Cs": "Cs(xy)",
    "Ci": "Ci",
    "C1": "C1",
}

#: Groups the reduction changes — the ones whose fermion irreps are two-dimensional.
NON_ABELIAN_DOUBLE = ("C2v", "D2", "D2h")


def resolve_group(name: str) -> Tuple[Group, bool]:
    """``(group, reduced)`` for a user-facing point-group name.

    Accepts either a label-group name (``"C2h(z)"``) or a Schoenflies name of the D2h chain
    (``"D2h"``), which is reduced per :data:`REDUCTION`. ``reduced`` says whether the group
    that came back is smaller than the one that was asked for, which is what the caller
    reports.
    """
    text = str(name).strip()
    if text in GROUPS:
        return GROUPS[text], False
    key = text.capitalize() if text[:1].isalpha() else text
    for candidate in (text, key, text.replace(" ", "")):
        for stored in REDUCTION:
            if candidate.lower() == stored.lower():
                return GROUPS[REDUCTION[stored]], stored in NON_ABELIAN_DOUBLE
    raise ValueError(
        "unknown point group {!r}; the D2h chain ({}) and the label groups ({}) are what "
        "this implementation has. A group outside the D2h chain has degenerate spatial "
        "irreps, whose labelling needs the non-abelian classification layer"
        .format(name, ", ".join(sorted(REDUCTION)), ", ".join(sorted(GROUPS))))


__all__ = ["C1", "CI", "C2H_Z", "C2Z", "CS_XY", "GEN_C2Z", "GEN_EBAR", "GEN_INVERSION",
           "GEN_SIGMA_XY", "GROUPS", "Generator", "Group", "NON_ABELIAN_DOUBLE",
           "REDUCTION", "SPATIAL_NAMES", "resolve_group"]
