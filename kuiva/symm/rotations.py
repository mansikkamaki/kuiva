"""General spatial operations: their 3x3 matrices, their spin factor, and their action on AOs.

Why this module exists beside :mod:`kuiva.symm.operators`
--------------------------------------------------------
The abelian label groups need exactly four spatial operations — ``E``, ``C2(z)``, ``i`` and
``sigma(xy)`` — every one of which maps a real solid harmonic to ``+/-`` itself, so
:mod:`kuiva.symm.operators` represents them as an index image and a sign and never builds a
matrix. The **classification** layer needs the rest of a molecule's point group, and a
threefold rotation does not have that form: it mixes the ``2l+1`` functions of a shell into
each other. So the operation has to carry a real ``(2l+1, 2l+1)`` block per shell.

⚠ **The conventions are not restated here, they are reproduced.** For the four operations both
modules can build, the matrices produced here are asserted by the suite to equal
:meth:`kuiva.symm.operators.AOOperation.matrix` element for element, and the spin factors to
equal :data:`kuiva.symm.operators.SPIN_FACTOR`. That is what keeps one convention rather than
two that agree until one is changed.

How the harmonic blocks are built
---------------------------------
By the recursion of Ivanic and Ruedenberg, which expresses the real-spherical-harmonic
rotation matrix of rank ``l`` in terms of rank ``l - 1`` and rank ``1`` — and rank 1 **is** the
Cartesian rotation matrix, written in the basis the AO layout uses for a p shell. That is the
whole reason to prefer it over a Wigner-D evaluation in the complex basis followed by a
real-basis transform: the convention of the real harmonics never has to be written down at all
above ``l = 1``, it is inherited from the ``(x, y, z)`` block, which the AO layout already
pins (:func:`kuiva.basis.layout.shell_m_values`, including the ``px, py, pz`` order of a p
shell that every other ``l`` does not follow).

Three properties are checked rather than trusted, because a wrong harmonic block is a
plausible unitary matrix: the blocks are orthogonal, they reproduce the sign table of
:mod:`kuiva.symm.operators` for the four operations that have one, and they are a
homomorphism, ``D_l(R1 R2) = D_l(R1) D_l(R2)``.

The spin factor and its branch
------------------------------
``D^(1/2)(R) = cos(theta/2) I - i sin(theta/2) (n . sigma)`` for a proper rotation of angle
``theta`` about ``n``; an improper operation is ``i`` times a proper one and inversion is inert
on spin, so an improper operation carries the spin factor of its proper part.

⚠ **The branch is a convention and it is fixed here, once.** ``theta`` is taken in
``[0, pi]`` with the axis fixed by the antisymmetric part of ``R``, and for ``theta = pi``
— where the axis is only defined up to a sign and the two choices differ by ``Ebar`` — the
axis is the one whose first nonzero Cartesian component is positive. About ``z`` that gives
``diag(-i, +i)``, which is :data:`kuiva.symm.operators.SPIN_FACTOR`'s convention and hence
:mod:`kuiva.spinor.expand`'s. Change it and every fermion irrep label swaps with its partner
while every number stays plausible.

References
----------
* J. Ivanic, K. Ruedenberg, "Rotation Matrices for Real Spherical Harmonics. Direct
  Determination by Recursion", J. Phys. Chem. **100**, 6342 (1996), doi:10.1021/jp953350u;
  and the erratum, J. Phys. Chem. A **102**, 9099 (1998), doi:10.1021/jp9833350, whose
  corrected ``P``/``U``/``V``/``W`` definitions are the ones implemented below.
* The ``exp(-i theta n.sigma / 2)`` spinor representation and the ``E``/``Ebar`` distinction:
  E. P. Wigner, "Group Theory and its Application to the Quantum Mechanics of Atomic
  Spectra", Academic Press (1959), ch. 15.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..basis.layout import shell_m_values
from ..util.logging import get_logger
# ⚠ The atom fingerprint is imported rather than re-derived: what makes two atoms images of
# each other under an operation -- same element, same charge, same shell list, so that a
# per-atom basis assignment breaks a symmetry the geometry alone would have -- is one
# definition, and a second copy of it would agree until one of them changed.
from .operators import DEFAULT_ATOM_TOL, _atom_signatures

log = get_logger(__name__)

#: How far two 3x3 operation matrices may differ and still be the same group element. The
#: matrices are products of exact generators, so the spread is roundoff over a handful of
#: multiplications; anything larger means the closure did not close.
ELEMENT_TOL = 1.0e-8


# --- Real solid harmonic rotation blocks ----------------------------------------------------

def _p_block(cart: np.ndarray) -> np.ndarray:
    """Rank-1 rotation matrix in the real-harmonic order ``m = -1, 0, +1``, i.e. ``(y, z, x)``.

    ``g chi_mu = sum_nu D_{nu mu} chi_nu`` with ``chi`` the coordinate functions themselves, so
    ``D`` **is** the Cartesian matrix, permuted into the harmonic order. This is the only place
    a real-harmonic convention is written down; every higher ``l`` inherits it.
    """
    order = [1, 2, 0]                        # (y, z, x) out of (x, y, z)
    return np.asarray(cart, dtype=float)[np.ix_(order, order)]


def _ivanic_ruedenberg(l: int, r1: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """One step of the recursion: ``D_l`` from ``D_1`` and ``D_(l-1)``, both in ``-m..m`` order."""
    size = 2 * l + 1

    def r(i: int, j: int) -> float:
        return float(r1[i + 1, j + 1])

    def d(a: int, b: int) -> float:
        if abs(a) > l - 1 or abs(b) > l - 1:
            return 0.0
        return float(prev[a + l - 1, b + l - 1])

    def p(i: int, a: int, b: int) -> float:
        if b == l:
            return r(i, 1) * d(a, l - 1) - r(i, -1) * d(a, -l + 1)
        if b == -l:
            return r(i, 1) * d(a, -l + 1) + r(i, -1) * d(a, l - 1)
        return r(i, 0) * d(a, b)

    out = np.zeros((size, size))
    for m in range(-l, l + 1):
        for mp in range(-l, l + 1):
            dm0 = 1.0 if m == 0 else 0.0
            denom = (2 * l) * (2 * l - 1) if abs(mp) == l else (l + mp) * (l - mp)
            u = np.sqrt((l + m) * (l - m) / denom)
            v = 0.5 * np.sqrt((1.0 + dm0) * (l + abs(m) - 1) * (l + abs(m)) / denom) \
                * (1.0 - 2.0 * dm0)
            w = -0.5 * np.sqrt(max((l - abs(m) - 1) * (l - abs(m)), 0) / denom) * (1.0 - dm0)

            big_u = p(0, m, mp)
            if m == 0:
                big_v = p(1, 1, mp) + p(-1, -1, mp)
                big_w = 0.0
            elif m > 0:
                dm1 = 1.0 if m == 1 else 0.0
                big_v = p(1, m - 1, mp) * np.sqrt(1.0 + dm1) - p(-1, -m + 1, mp) * (1.0 - dm1)
                big_w = p(1, m + 1, mp) + p(-1, -m - 1, mp)
            else:
                dm1 = 1.0 if m == -1 else 0.0
                big_v = p(1, m + 1, mp) * (1.0 - dm1) + p(-1, -m - 1, mp) * np.sqrt(1.0 + dm1)
                big_w = p(1, m - 1, mp) - p(-1, -m + 1, mp)
            out[m + l, mp + l] = u * big_u + v * big_v + w * big_w
    return out


def harmonic_rotation(l: int, cart: np.ndarray) -> np.ndarray:
    """``D_l`` for a spatial operation, in the AO layout's own ``m`` order for that shell.

    ``cart`` is the 3x3 matrix of the operation, improper included: an improper operation is
    the inversion times a proper rotation, and inversion contributes ``(-1)^l`` to every
    function of the shell, so the block is ``(-1)^l`` times the proper part's.
    """
    cart = np.asarray(cart, dtype=float)
    l = int(l)
    parity = 1.0
    if np.linalg.det(cart) < 0.0:
        cart = -cart
        parity = float((-1) ** l)
    blocks = [np.ones((1, 1)), _p_block(cart)]
    for k in range(2, l + 1):
        blocks.append(_ivanic_ruedenberg(k, blocks[1], blocks[k - 1]))
    full = parity * blocks[l]
    order = [m + l for m in shell_m_values(l)]
    return np.ascontiguousarray(full[np.ix_(order, order)])


# --- Spatial operations ---------------------------------------------------------------------

_AXIS_NAMES = {(1, 0, 0): "x", (0, 1, 0): "y", (0, 0, 1): "z"}
_PLANE_NAMES = {"x": "yz", "y": "xz", "z": "xy"}


def rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    """Proper rotation by ``angle`` about ``axis`` (Rodrigues), as a 3x3 matrix."""
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    k = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def reflection(normal: Sequence[float]) -> np.ndarray:
    """Reflection in the plane whose normal is ``normal``."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return np.eye(3) - 2.0 * np.outer(n, n)


def _axis_angle(proper: np.ndarray) -> Tuple[np.ndarray, float]:
    """``(axis, angle)`` of a proper rotation, with the ``theta = pi`` sign convention fixed."""
    trace = float(np.clip((np.trace(proper) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(trace))
    if angle < 1.0e-10:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if abs(angle - np.pi) < 1.0e-7:
        # A half turn: the antisymmetric part vanishes and the axis is the +1 eigenvector,
        # defined only up to a sign. ⚠ The sign is the branch that separates C2 from C2bar,
        # and it is fixed to "first nonzero component positive" here and nowhere else.
        values, vectors = np.linalg.eigh(0.5 * (proper + proper.T))
        axis = vectors[:, int(np.argmax(values))]
        nonzero = np.nonzero(np.abs(axis) > 1.0e-8)[0]
        if nonzero.size and axis[nonzero[0]] < 0.0:
            axis = -axis
        return axis / np.linalg.norm(axis), np.pi
    axis = np.array([proper[2, 1] - proper[1, 2],
                     proper[0, 2] - proper[2, 0],
                     proper[1, 0] - proper[0, 1]]) / (2.0 * np.sin(angle))
    return axis / np.linalg.norm(axis), angle


def spin_factor(cart: np.ndarray) -> np.ndarray:
    """``D^(1/2)`` of a spatial operation, in the ``(alpha, beta)`` basis.

    Inversion is inert on spin, so an improper operation carries its proper part's factor and
    the sign of the determinant never enters. See the module docstring for the branch.
    """
    cart = np.asarray(cart, dtype=float)
    proper = cart if np.linalg.det(cart) > 0.0 else -cart
    axis, angle = _axis_angle(proper)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    n_sigma = axis[0] * sx + axis[1] * sy + axis[2] * sz
    return (np.cos(angle / 2.0) * np.eye(2, dtype=np.complex128)
            - 1j * np.sin(angle / 2.0) * n_sigma)


def canonical_axis_angle(proper: np.ndarray) -> Tuple[np.ndarray, float]:
    """``(axis, angle)`` of a proper rotation with ``angle`` in ``[0, 2 pi)``.

    ⚠ **This is the NAMING convention and it is deliberately not the branch of**
    :func:`spin_factor`. Naming needs a bijection onto the spatial operations, so the axis is
    the one whose first nonzero component is positive and the angle runs the whole turn —
    which is what distinguishes ``C3(z)`` from ``C3^2(z)`` rather than calling both of them a
    threefold rotation about some axis. Which of ``g`` and ``g Ebar`` an element *is* comes
    from its stored spin factor, never from re-deriving one here.
    """
    axis, angle = _axis_angle(proper)
    nonzero = np.nonzero(np.abs(axis) > 1.0e-8)[0]
    if nonzero.size and axis[nonzero[0]] < 0.0:
        axis, angle = -axis, (2.0 * np.pi - angle) % (2.0 * np.pi)
    return axis, angle


def _axis_text(axis: np.ndarray) -> str:
    for key, name in _AXIS_NAMES.items():
        if np.allclose(axis, key, atol=1.0e-8):
            return name
    scaled = axis / np.max(np.abs(axis))
    small = np.rint(scaled * 2.0).astype(int)
    if np.allclose(scaled * 2.0, small, atol=1.0e-6) and np.all(np.abs(small) <= 4):
        common = int(np.gcd.reduce(np.abs(small[small != 0]))) or 1
        small = small // common
        return "[" + "".join("{:d}".format(v) if v >= 0 else "-{:d}".format(-v)
                             for v in small) + "]"
    return "[{:.2f},{:.2f},{:.2f}]".format(*axis)


def _turn_text(symbol: str, angle: float, axis: np.ndarray) -> Optional[str]:
    """``C3^2(z)``-style name for a turn of ``angle``, or ``None`` if it is not a rational one."""
    if angle < 1.0e-8:
        return None
    for n in range(2, 13):
        k = angle * n / (2.0 * np.pi)
        if abs(k - np.rint(k)) < 1.0e-6:
            k = int(np.rint(k))
            if k == 0:
                return None
            common = np.gcd(k, n)
            k, n = k // common, n // common
            text = "{}{}".format(symbol, n) if k == 1 else "{}{}^{}".format(symbol, n, k)
            return "{}({})".format(text, _axis_text(axis))
    return "{}({:.1f} deg, {})".format(symbol, np.degrees(angle), _axis_text(axis))


def operation_name(cart: np.ndarray) -> str:
    """Lab-frame geometric name of a spatial operation: ``C3^2(z)``, ``sigma(xz)``, ``S4(z)``...

    ⚠ Every operation in every printed table is named this way and never by a Schoenflies
    label whose axis the reader has to guess: two programs agreeing on "C2v, B1" and
    disagreeing on which plane is which produce different numbers and no error message.
    """
    cart = np.asarray(cart, dtype=float)
    if np.linalg.det(cart) > 0.0:
        axis, angle = canonical_axis_angle(cart)
        return _turn_text("C", angle, axis) or "E"
    axis, angle = canonical_axis_angle(-cart)
    # An improper operation is the inversion times a proper rotation, and i = sigma_h C2, so
    # writing it as a rotation-reflection shifts the turn by half: S(phi) = i C(phi - pi).
    phi = (angle + np.pi) % (2.0 * np.pi)
    if phi < 1.0e-8:
        name = _axis_text(axis)
        return "sigma({})".format(_PLANE_NAMES.get(name, "perp " + name))
    if abs(phi - np.pi) < 1.0e-8:
        return "i"
    return _turn_text("S", phi, axis) or "i"


# --- The AO representation ------------------------------------------------------------------

@dataclass(frozen=True)
class AOTransform:
    """``U(g)`` on the AO basis as a shell permutation plus one real block per shell.

    :meth:`apply` is a scatter of small GEMMs and never forms an ``nao x nao`` array;
    :meth:`matrix` materializes one for the tests and the ``tr U(g)`` self-consistency check.
    """

    name: str
    cart: np.ndarray                      # (3, 3)
    spin: np.ndarray                      # (2, 2) complex
    nao: int
    shell_rows: Tuple[np.ndarray, ...]    # AO rows of each shell
    shell_image: np.ndarray               # shell -> image shell
    blocks: Tuple[np.ndarray, ...]        # per shell, (nao_shell, nao_shell)
    atom_image: np.ndarray

    def apply(self, c: np.ndarray) -> np.ndarray:
        """``U(g) c`` for AO-basis coefficient columns ``(nao, n)`` or a vector ``(nao,)``."""
        c = np.asarray(c)
        out = np.zeros_like(c)
        for s, rows in enumerate(self.shell_rows):
            dst = self.shell_rows[int(self.shell_image[s])]
            out[dst] = self.blocks[s] @ c[rows]
        return out

    def matrix(self) -> np.ndarray:
        u = np.zeros((self.nao, self.nao))
        for s, rows in enumerate(self.shell_rows):
            dst = self.shell_rows[int(self.shell_image[s])]
            u[np.ix_(dst, rows)] = self.blocks[s]
        return u

    def two_component(self) -> np.ndarray:
        """``U(g)`` on the spin-blocked two-component AO basis ``[alpha ; beta]``.

        ``kron(D^(1/2), U_spatial)`` — the row ordering of :mod:`kuiva.spinor.expand`, consumed
        and not restated.
        """
        return np.kron(self.spin, self.matrix()).astype(np.complex128)

    def apply_two_component(self, c: np.ndarray) -> np.ndarray:
        """:meth:`apply` on a ``(2*nao, n)`` spin-blocked coefficient array."""
        c = np.asarray(c, dtype=np.complex128)
        nao = self.nao
        upper, lower = self.apply(c[:nao]), self.apply(c[nao:])
        out = np.empty_like(c)
        out[:nao] = self.spin[0, 0] * upper + self.spin[0, 1] * lower
        out[nao:] = self.spin[1, 0] * upper + self.spin[1, 1] * lower
        return out


def _shell_tables(layout):
    rows: List[np.ndarray] = []
    start = 0
    for sh in layout.shells:
        n = sh.nao
        rows.append(np.arange(start, start + n))
        start += n
    if start != layout.nao:
        raise RuntimeError("the shell list covers {} AOs and the layout has {}"
                           .format(start, layout.nao))
    return rows


def atom_permutation_general(layout, cart: np.ndarray, *,
                             tol: float = DEFAULT_ATOM_TOL) -> Optional[np.ndarray]:
    """``atom -> image atom`` under a general 3x3 operation, or ``None`` if it is not one.

    Atoms match only when element, nuclear charge and the whole shell list agree, exactly as
    :func:`kuiva.symm.operators.atom_permutation` requires: a per-atom basis assignment breaks
    a symmetry the geometry alone would have.
    """
    coords = np.asarray(layout.coords_bohr, dtype=float)
    moved = coords @ np.asarray(cart, dtype=float).T
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


def ao_transform(layout, cart: np.ndarray, *, name: Optional[str] = None,
                 tol: float = DEFAULT_ATOM_TOL) -> Optional[AOTransform]:
    """``U(g)`` for a general spatial operation, or ``None`` when the molecule does not have it."""
    cart = np.asarray(cart, dtype=float)
    atom_image = atom_permutation_general(layout, cart, tol=tol)
    if atom_image is None:
        return None
    rows = _shell_tables(layout)
    per_atom: Dict[int, List[int]] = {a: [] for a in range(layout.natm)}
    for s, sh in enumerate(layout.shells):
        per_atom[int(sh.atom)].append(s)
    shell_image = np.empty(len(rows), dtype=int)
    for a in range(layout.natm):
        src, dst = per_atom[a], per_atom[int(atom_image[a])]
        if len(src) != len(dst):
            return None
        for k, s in enumerate(src):
            if layout.shells[s].l != layout.shells[dst[k]].l:
                return None
            shell_image[s] = dst[k]
    cache: Dict[int, np.ndarray] = {}
    blocks = []
    for sh in layout.shells:
        l = int(sh.l)
        if l not in cache:
            cache[l] = harmonic_rotation(l, cart)
        blocks.append(cache[l])
    return AOTransform(name=name or operation_name(cart), cart=cart, spin=spin_factor(cart),
                       nao=int(layout.nao), shell_rows=tuple(rows), shell_image=shell_image,
                       blocks=tuple(blocks), atom_image=atom_image)


__all__ = ["AOTransform", "ELEMENT_TOL", "ao_transform", "atom_permutation_general",
           "canonical_axis_angle", "harmonic_rotation", "operation_name", "reflection",
           "rotation", "spin_factor"]
