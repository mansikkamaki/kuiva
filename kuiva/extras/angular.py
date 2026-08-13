"""Angular algebra of the atomic two-electron integrals: 3j, Gaunt, and the ``A^k`` tensors.

What this module is for
-----------------------
A two-electron integral between orbitals of a **spherical** atom factorizes into a radial part
that carries all the chemistry and an angular part that is a pure number. Writing the chemists'
integral over real-harmonic orbitals as

.. math::

    (p q | r s) \\;=\\; \\sum_k A^k(p q; r s)\\, R^k(p r; q s)

is the whole content: the ``R^k`` are the Slater-Condon radial parameters and the ``A^k`` are
what this module computes. Everything here is exact rational arithmetic up to a single square
root, depends on nothing but angular momenta, and has no free convention left once the four
statements below are fixed.

Conventions, all four of them, fixed here and nowhere else
----------------------------------------------------------
**1. The radial integral is Condon-Shortley ordered.** With ``P_a(r) = r R_a(r)``,

.. math::

    R^k(ab; cd) = \\int\\!\\!\\int P_a(r_1) P_c(r_1)\\,
                  \\frac{r_<^k}{r_>^{k+1}}\\, P_b(r_2) P_d(r_2)\\, dr_1 dr_2 ,

so the **first and third** labels sit on electron 1 and the **second and fourth** on electron
2 — ``a, b`` are the bra orbitals of electrons 1 and 2, ``c, d`` the ket orbitals of the same
two. The direct and exchange parameters are the two special cases ``F^k(a,b) = R^k(ab;ab)``
and ``G^k(a,b) = R^k(ab;ba)``.

**2. The chemists' integral is** ``(p q | r s) = \\int\\!\\!\\int \\phi_p^*(1) \\phi_q(1)
r_{12}^{-1} \\phi_r^*(2) \\phi_s(2)``, so ``p, q`` are on electron 1 and ``r, s`` on electron 2.
Matching the two conventions gives the pairing in the expansion above: the radial parameter
belonging to ``(p q | r s)`` is ``R^k(p r; q s)``. ⚠ **The reindexing is the step that is easy
to get wrong and impossible to notice**, because a wrong assignment still produces a symmetric,
plausible, and entirely incorrect set of parameters. Worked example, and the one to check any
future change against: the chemists' integral ``(4f 4f | 6s 5d)`` is
``\\sum_k A^k R^k(4f\\,6s; 4f\\,5d)``, and only ``k = 2`` survives the selection rules.

**3. Complex spherical harmonics carry the Condon-Shortley phase**, so that
``Y_l^{-m} = (-1)^m (Y_l^m)^*`` and the raising and lowering operators have real positive
matrix elements.

**4. Real spherical harmonics are the integral library's**, i.e. the standard real solid
harmonics

.. math::

    Y_{lm}^{\\mathrm{real}} = \\sqrt{2}\\,(-1)^m \\operatorname{Re} Y_l^{m}\\ (m > 0),\\quad
    Y_{l0}^{\\mathrm{real}} = Y_l^{0},\\quad
    Y_{lm}^{\\mathrm{real}} = \\sqrt{2}\\,(-1)^m \\operatorname{Im} Y_l^{|m|}\\ (m < 0),

which give ``p`` functions proportional to ``x, y, z`` with positive constants.
:func:`complex_to_real` is the single definition and every tensor here is expressed in it.

⚠ **Every index of every array in this module runs over ascending ``m``, from ``-l`` to
``+l``.** That is *not* the order the integral library lays a shell out in — a ``p`` shell is
stored ``px, py, pz``, i.e. ``m = +1, -1, 0`` — so anything contracting these tensors against
AO-basis quantities must reorder, and the shell orbitals this module is written for
(:mod:`kuiva.extras.shells`) already come in ascending ``m`` for exactly this reason.

Why the arithmetic is exact
---------------------------
A 3j symbol is the square root of a rational times a rational. Computing it in floating point
through factorials loses digits to cancellation in the Racah sum — badly, for the ``f`` shells
this feature exists for, where the alternating sum has terms many orders above its value. So
the sum and the prefactor are built with :class:`fractions.Fraction` over Python's arbitrary
precision integers and **one** square root is taken at the end. The cost is irrelevant (a few
hundred symbols per run, memoized) and the result is correct to the last bit for any angular
momentum a basis set will ever hold.

Half-integer angular momenta are **refused rather than supported**: nothing here needs them —
the spin enters separately and analytically in :func:`spin_orbit_matrix` — and accepting them
would mean carrying half-integers through the exact arithmetic for a case with no caller.

References
----------
* E. U. Condon, G. H. Shortley, *The Theory of Atomic Spectra*, Cambridge University Press
  (1935), Chapter VI — the ``c^k``, ``F^k``, ``G^k`` and ``R^k`` definitions and the phase
  conventions used throughout.
* G. Racah, "Theory of Complex Spectra. II", Phys. Rev. **62**, 438 (1942) — the closed-form
  (single-sum) expression for the 3j symbol implemented in :func:`wigner_3j`.
* R. D. Cowan, *The Theory of Atomic Structure and Spectra*, University of California Press
  (1981), Chapters 6 and 11 — the Laplace expansion of ``1/r_12``, the ``c^k`` tables and the
  configuration-average energy expressions these tensors reproduce.
* H. B. Schlegel, M. J. Frisch, "Transformation between Cartesian and pure spherical harmonic
  Gaussians", Int. J. Quantum Chem. **54**, 83 (1995), doi:10.1002/qua.560540202 — the real
  solid-harmonic convention of the integral library.
"""
from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache
from typing import Tuple

import numpy as np

#: Largest imaginary part tolerated in a quantity that is real by construction, relative to
#: the size of the array. The real-harmonic tensors below are real because the functions are;
#: a nonzero imaginary part is a mistake in the transformation, not a small number.
_REAL_TOL = 1e-12


# --- Exact 3j symbols ----------------------------------------------------------------------

def _sqrt_fraction(value: Fraction) -> float:
    """``sqrt`` of a non-negative :class:`~fractions.Fraction`, without overflowing.

    ``numerator / denominator`` on Python integers is a correctly rounded float whenever the
    *quotient* is representable, which it is here even when the factorials themselves are far
    past the float range.
    """
    return math.sqrt(value.numerator / value.denominator)


@lru_cache(maxsize=None)
def wigner_3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """The 3j symbol, exact to the last bit for integer angular momenta.

    Racah's single-sum formula (Racah 1942), evaluated in exact rational arithmetic with one
    square root at the end — see the module docstring for why floating-point factorials are
    not good enough here.

    Returns ``0.0`` for every combination the selection rules forbid (``m1 + m2 + m3 != 0``,
    a violated triangle condition, ``|m| > j``), so a caller may loop over full ranges without
    testing anything.

    ⚠ Half-integer arguments **raise**. This module's angular momenta are orbital ones; spin
    is handled separately and analytically.
    """
    js = (j1, j2, j3, m1, m2, m3)
    for value in js:
        if value != int(value):
            raise ValueError(
                "the 3j symbols here are for integer (orbital) angular momenta only, and {} "
                "is not one; spin enters analytically in spin_orbit_matrix() instead."
                .format(value))
    j1, j2, j3, m1, m2, m3 = (int(v) for v in js)
    if m1 + m2 + m3 != 0:
        return 0.0
    if j1 < 0 or j2 < 0 or j3 < 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if not abs(j1 - j2) <= j3 <= j1 + j2:
        return 0.0

    f = math.factorial
    triangle = Fraction(f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3),
                        f(j1 + j2 + j3 + 1))
    prefactor = triangle * (f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2) *
                            f(j3 + m3) * f(j3 - m3))
    total = Fraction(0)
    t_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
    t_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    for t in range(t_min, t_max + 1):
        denominator = (f(t) * f(j3 - j2 + t + m1) * f(j3 - j1 + t - m2) *
                       f(j1 + j2 - j3 - t) * f(j1 - t - m1) * f(j2 - t + m2))
        total += Fraction((-1) ** t, denominator)
    return (-1) ** (j1 - j2 - m3) * _sqrt_fraction(prefactor) * float(total)


@lru_cache(maxsize=None)
def gaunt(l1: int, m1: int, l2: int, m2: int, l3: int, m3: int) -> float:
    """``Int Y_{l1 m1}^* Y_{l2 m2} Y_{l3 m3} dOmega`` over **complex** harmonics.

    Note the conjugation on the *first* pair only — the asymmetry is what makes this the
    coefficient that appears in the Laplace expansion of ``1/r_12`` rather than the fully
    symmetric Gaunt coefficient of the three-``Y`` product.
    """
    l1, m1, l2, m2, l3, m3 = (int(v) for v in (l1, m1, l2, m2, l3, m3))
    if m1 != m2 + m3:
        return 0.0
    norm = math.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1) / (4.0 * math.pi))
    return ((-1) ** (m1 % 2) * norm * wigner_3j(l1, l2, l3, 0, 0, 0)
            * wigner_3j(l1, l2, l3, -m1, m2, m3))


@lru_cache(maxsize=None)
def condon_shortley_ck(k: int, l: int, m: int, lp: int, mp: int) -> float:
    """The Condon-Shortley coefficient ``c^k(l m; l' m')``.

    .. math::

        c^k(lm; l'm') = \\sqrt{\\frac{4\\pi}{2k+1}}
                        \\int Y_{lm}^* \\, Y_{k, m-m'} \\, Y_{l'm'} \\, d\\Omega

    (Condon & Shortley 1935; Cowan 1981, Eq. 6.29). ``c^0(lm; lm) = 1`` for every ``l`` and
    ``m``, which is the cheapest check that the normalization here is the tabulated one.
    """
    return math.sqrt(4.0 * math.pi / (2 * int(k) + 1)) * gaunt(l, m, k, int(m) - int(mp),
                                                               lp, mp)


# --- Real spherical harmonics ---------------------------------------------------------------

@lru_cache(maxsize=None)
def complex_to_real(l: int) -> np.ndarray:
    """The unitary ``U`` with ``Y^real_{lm} = sum_mu U[m, mu] Y_l^mu``.

    Both indices run over ascending ``m`` (row ``i`` is ``m = i - l``). The convention is
    stated in the module docstring; the two properties that pin it down and are asserted by
    the tests are that ``U`` is unitary and that the ``l = 1`` rows come out as ``x, y, z``
    with **positive** coefficients — the sign that decides whether every odd-``l`` parameter
    in the output has the right sign.
    """
    l = int(l)
    n = 2 * l + 1
    u = np.zeros((n, n), dtype=np.complex128)
    root_half = 1.0 / math.sqrt(2.0)
    u[l, l] = 1.0
    for m in range(1, l + 1):
        phase = (-1) ** m
        u[l + m, l + m] = phase * root_half            # m > 0: cosine-like combination
        u[l + m, l - m] = root_half
        u[l - m, l + m] = -1j * phase * root_half      # m < 0: sine-like combination
        u[l - m, l - m] = 1j * root_half
    u.flags.writeable = False
    return u


def _as_real(a: np.ndarray, what: str) -> np.ndarray:
    """Drop an imaginary part that is zero by construction, or raise if it is not."""
    scale = max(float(np.max(np.abs(a))), 1e-300)
    largest = float(np.max(np.abs(a.imag)))
    if largest > _REAL_TOL * scale:
        raise RuntimeError(
            "{} came out with an imaginary part {:.2e} of its own scale; it is real by "
            "construction, so this is an error in the complex-to-real transformation rather "
            "than a small number".format(what, largest / scale))
    return np.ascontiguousarray(a.real)


# --- Selection rules and the A^k tensors ----------------------------------------------------

def couples(l1: int, l2: int, k: int) -> bool:
    """Whether ``k`` couples ``l1`` and ``l2``: triangle condition **and** even parity.

    Both halves matter and only one of them is obvious. The triangle condition
    ``|l1 - l2| <= k <= l1 + l2`` bounds the range; the parity condition ``l1 + l2 + k`` even
    is what makes ``F^1``, ``F^3``, ... vanish identically and is why a shell's direct
    parameters are ``F^0, F^2, F^4, F^6`` rather than every ``k`` up to ``2l``.
    """
    return abs(l1 - l2) <= k <= l1 + l2 and (l1 + l2 + k) % 2 == 0


def admissible_k(l_p: int, l_q: int, l_r: int, l_s: int) -> Tuple[int, ...]:
    """The ``k`` values for which ``(p q | r s)`` has a nonvanishing angular coefficient.

    ``k`` must couple the **electron-1** pair ``(p, q)`` and the **electron-2** pair
    ``(r, s)`` at once. An empty result means the whole parameter class is identically zero by
    symmetry and is not a parameter at all — ``(4f 5d | 6s 6s)`` is such a case, and
    ``(4f 4f | 6s 5d)`` is the case that survives at ``k = 2`` alone.
    """
    lo = max(abs(l_p - l_q), abs(l_r - l_s))
    hi = min(l_p + l_q, l_r + l_s)
    return tuple(k for k in range(lo, hi + 1)
                 if couples(l_p, l_q, k) and couples(l_r, l_s, k))


@lru_cache(maxsize=None)
def angular_tensor(l_p: int, l_q: int, l_r: int, l_s: int, k: int) -> np.ndarray:
    """``A^k(p q; r s)``: the coefficient of ``R^k(p r; q s)`` in ``(p q | r s)``.

    Returns a **read-only** ``(2l_p+1, 2l_q+1, 2l_r+1, 2l_s+1)`` real array over **real**
    spherical harmonics, every index in ascending ``m``. The defining identity, and the one
    the extraction inverts, is

    .. math::

        (p m_p, q m_q | r m_r, s m_s) = \\sum_k A^k[m_p, m_q, m_r, m_s]\\; R^k(pr; qs) .

    Construction, in two steps that are each simple and neither of which may be skipped:

    * over **complex** harmonics the tensor factorizes into two Condon-Shortley coefficients
      and a conservation delta,
      ``A^k = c^k(l_p mu_p; l_q mu_q)\\, c^k(l_s mu_s; l_r mu_r)\\,
      \\delta_{mu_p + mu_r,\\, mu_q + mu_s}``.
      ⚠ **The delta is not implied by the two coefficients** — each is separately nonzero for
      magnetic quantum numbers that do not conserve the total, and dropping it produces a
      dense, symmetric, wrong tensor;
    * the four indices are then rotated to real harmonics with :func:`complex_to_real`,
      **conjugated on the two bra indices** ``p`` and ``r`` and not on the ket indices.

    The result is real by construction and is checked to be.
    """
    l_p, l_q, l_r, l_s, k = (int(v) for v in (l_p, l_q, l_r, l_s, k))
    m_p = np.arange(-l_p, l_p + 1)
    m_q = np.arange(-l_q, l_q + 1)
    m_r = np.arange(-l_r, l_r + 1)
    m_s = np.arange(-l_s, l_s + 1)

    c_one = np.array([[condon_shortley_ck(k, l_p, a, l_q, b) for b in m_q] for a in m_p])
    c_two = np.array([[condon_shortley_ck(k, l_s, b, l_r, a) for a in m_r] for b in m_s])
    conserved = (m_p[:, None, None, None] + m_r[None, None, :, None]
                 == m_q[None, :, None, None] + m_s[None, None, None, :])
    a_complex = (c_one[:, :, None, None] * c_two.T[None, None, :, :]) * conserved

    u_p, u_q = complex_to_real(l_p), complex_to_real(l_q)
    u_r, u_s = complex_to_real(l_r), complex_to_real(l_s)
    # Tiny arrays (7^4 at the largest shell any basis carries), evaluated a few dozen times
    # per run: einsum is orchestration here, not a kernel.
    a_real = np.einsum("pP,qQ,rR,sS,PQRS->pqrs", u_p.conj(), u_q, u_r.conj(), u_s,
                       a_complex, optimize=True)
    result = _as_real(a_real, "the angular tensor A^{}({}{};{}{})".format(
        k, l_p, l_q, l_r, l_s))
    result.flags.writeable = False
    return result


# --- Orbital angular momentum and the spin-orbit operator -----------------------------------

@lru_cache(maxsize=None)
def angular_momentum_matrices(l: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(Lx, Ly, Lz)`` in the **real** harmonic basis of one shell, in units of hbar.

    Each is ``(2l+1, 2l+1)`` complex with rows and columns in ascending ``m``. They are built
    in the complex basis, where ``Lz`` is diagonal and ``L±`` has the textbook real matrix
    elements, and rotated by :func:`complex_to_real` — an operator transforms as
    ``O_real = U^* O U^T``, which is *not* ``U O U^†``.

    ⚠ In the real basis all three are **purely imaginary and antisymmetric**, so an
    implementation that quietly takes a real part somewhere returns zero and looks like a
    system with no orbital angular momentum.
    """
    l = int(l)
    n = 2 * l + 1
    m = np.arange(-l, l + 1, dtype=float)
    lz = np.diag(m).astype(np.complex128)
    raise_op = np.zeros((n, n), dtype=np.complex128)
    for i in range(n - 1):                              # L+ |l, m> = c |l, m+1>
        raise_op[i + 1, i] = math.sqrt(l * (l + 1) - m[i] * (m[i] + 1))
    lower_op = raise_op.conj().T
    lx = 0.5 * (raise_op + lower_op)
    ly = (raise_op - lower_op) / 2j

    u = complex_to_real(l)
    return tuple(np.ascontiguousarray(u.conj() @ op @ u.T) for op in (lx, ly, lz))


@lru_cache(maxsize=None)
def spin_orbit_matrix(l: int) -> np.ndarray:
    """``l . s`` for one shell, ``(2(2l+1), 2(2l+1))`` complex, in units of hbar^2.

    The basis is the real harmonics of the shell times spin, **spin-blocked**: the first
    ``2l+1`` rows are the alpha component and the last ``2l+1`` the beta component, each in
    ascending ``m``. That is the row convention the two-component code uses throughout, so a
    shell block of a spinor operator can be contracted against this with no reordering.

    The spectrum is the closed form the fit of a spin-orbit constant is checked against:
    ``l/2`` with multiplicity ``2l+2`` (the ``j = l + 1/2`` level) and ``-(l+1)/2`` with
    multiplicity ``2l`` (the ``j = l - 1/2`` level), so the splitting between them is
    ``(2l+1)/2`` in units of the constant.
    """
    lx, ly, lz = angular_momentum_matrices(l)
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = 0.5 * np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    ls = np.kron(sx, lx) + np.kron(sy, ly) + np.kron(sz, lz)
    ls = np.ascontiguousarray(ls)
    ls.flags.writeable = False
    return ls


__all__ = ["admissible_k", "angular_momentum_matrices", "angular_tensor", "complex_to_real",
           "condon_shortley_ck", "couples", "gaunt", "spin_orbit_matrix", "wigner_3j"]
