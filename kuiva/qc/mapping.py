"""Fermion-to-qubit mapping: the active-space Hamiltonian as a sum of Pauli strings.

**This module is orchestration and Pauli algebra, not a registered kernel.** It runs once per set
of integrals, its cost is dwarfed by everything around it, and nothing here is registered in
``ci/kernels.py``. What it *is* is the one place Kuiva's second-quantized conventions are
translated into qubit operators, and that translation is the single most dangerous step in
:mod:`kuiva.qc`: a sign or ordering error here is Hermitian, of plausible magnitude, and
wrong — the defects that cost the most. Hence the structure below, in
which the convention is written once and checked against a genuinely independent
implementation (``ci/strings.hamiltonian_matrix``) rather than by inspection.

What is mapped
--------------
Exactly the operator ``ci/strings.py`` and ``ci/sigma.py`` diagonalize::

    H = sum_pq h_pq a_p^dag a_q + 1/2 sum_pqrs (pq|rs) a_p^dag a_r^dag a_s a_q

with ``(pq|rs)`` in chemists' notation and only **4-fold** permutational symmetry:
``(pq|rs) = (rs|pq)`` and ``(pq|rs)* = (qp|sr)``. Both ``h`` and the ERI are complex.

⚠ **``e_core`` is excluded**, exactly as ``SigmaOperator`` and the DMRG's TTNO exclude it
. Drivers add it to the energy. An identity Pauli string does appear here — it comes
from normal-ordering the active-space operator itself — and it is *not* ``e_core``.

The representation: symplectic (X-mask, Z-mask), real coefficient
-----------------------------------------------------------------
A term is one real coefficient plus two ``uint64`` bitmasks. Bit ``k`` of ``x`` says qubit
``k`` carries an ``X`` factor, bit ``k`` of ``z`` a ``Z``; both set means ``Y``. The operator
a term denotes is the **Hermitian tensor product** of single-qubit Paulis::

    P(x, z) = i^popcount(x & z) * X^x Z^z

so every coefficient of a Hermitian ``H`` is real *by construction* — a complex one would
break Hermiticity, since the Pauli strings are themselves Hermitian. That fact is used as a
check, not assumed: :func:`jordan_wigner` accumulates in the ``X^x Z^z`` convention where
coefficients are complex and cancellation across the 4-fold-symmetric integral set is what
makes them real, and refuses if the imaginary part survives.

Three reasons for this encoding rather than a framework's operator type:

* it is framework-independent, so the mapping's output crosses the backend boundary
  untranslated and an out-of-process adapter stays possible;
* it reuses the same ``<= 64`` mode masking convention ``ci/strings.py`` already lives with,
  imported from there rather than re-derived (see that module's public bit helpers);
* a Pauli string acts on a computational basis state by one XOR and two popcounts, which is
  what makes :func:`pauli_expectation` and :meth:`PauliSum.to_dense` short enough to trust.

Jordan-Wigner, and why it needs no index reshuffling here
----------------------------------------------------------
``ci/strings.py`` addresses the CI space with a single occupation-string bitmask over
**spinors** — chosen because SOC forbids the usual alpha/beta factorization — which is
already the JW computational basis, mode ``p`` to qubit ``p``, bit for bit. So the default
ordering is the identity, ``|I>`` is the computational basis state whose integer *is* the
determinant mask, and a JW-mapped matrix element can be compared to a CI matrix element with
no permutation in between. That is what makes the Stage-1 test exact rather than approximate.

The convention, spelled out because everything downstream inherits it::

    a_p     = (prod_{k<p} Z_k) (X_p + i Y_p)/2
    a_p^dag = (prod_{k<p} Z_k) (X_p - i Y_p)/2

with ``(X + iY)/2 = |0><1|``, i.e. occupied is ``|1>``. The parity string ``prod_{k<p} Z_k``
reproduces ``ci/strings.py``'s ``(-1)^(occupied below p)`` sign for ascending-ordered
determinants; :func:`kuiva.ci.strings.below_mask` is where the "below p" index set is defined,
and it is imported rather than rewritten.

Other mappings
--------------
:func:`resolve_mapping` is a name-to-implementation registry, the same extension pattern as
``ci/kernels.py``, ``amf/backend.py`` and ``qc/backend.py``. ⚠ Only ``"jordan_wigner"`` is
registered. Parity and Bravyi-Kitaev are *not* — following ``amf/backend.py``'s decision that
a name resolving to something non-functional is worse than a name that does not resolve. They
would slot in here unchanged, and a Bravyi-Kitaev implementation wanting unbarred/barred
spinors grouped instead of interleaved should reuse
:func:`kuiva.spinor.expand.kramers_block_permutation` rather than re-derive the permutation.

References
----------------------------
* P. Jordan, E. Wigner, "Ueber das Paulische Aequivalenzverbot", *Z. Phys.* **47**, 631
  (1928), doi:10.1007/BF01331938 — the transformation itself.
* S. B. Bravyi, A. Y. Kitaev, "Fermionic quantum computation", *Ann. Phys.* **298**, 210
  (2002), doi:10.1006/aphy.2002.6254 — the alternative encodings the registry is open to.
* J. T. Seeley, M. J. Richard, P. J. Love, "The Bravyi-Kitaev transformation for quantum
  computation of electronic structure", *J. Chem. Phys.* **137**, 224109 (2012),
  doi:10.1063/1.4768229 — the standard statement of the JW/BK electronic-structure mapping
  and of the symplectic bookkeeping used here.
* The two-electron operator ordering and the ``Gamma_pqrs`` pairing follow T. Helgaker,
  P. Jorgensen, J. Olsen, "Molecular Electronic-Structure Theory", Wiley (2000), ch. 1-2, as
  ``ci/strings.py`` does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from ..ci.strings import DEFAULT_MAX_SPINORS, below_mask, mode_bit, popcount
from ..util import resources as res
from ..util.logging import get_logger
from ..util.timing import timer

log = get_logger(__name__)

_U64 = np.uint64

#: Qubits addressable by a single ``uint64`` mask — the same limit, for the same reason, as
#: ``ci/strings.DEFAULT_MAX_SPINORS``. Going further needs multi-word masks there first.
MAX_QUBITS = DEFAULT_MAX_SPINORS

#: Integrals below this magnitude contribute no Pauli term. Twelve orders below the 1e-8 Eh
#: suite tolerance and six below the tightest Cholesky threshold, so it removes the
#: structural zeros of a sparse active-space Hamiltonian and nothing else. It is a screening
#: parameter, not an accuracy knob: pass ``tol=0.0`` for a bit-exact unscreened mapping.
INTEGRAL_SCREEN_TOL = 1e-14

#: Relative size at which a surviving imaginary Pauli coefficient stops being rounding.
#: ⚠ Reaching it means the *input* was not Hermitian (or the ERI not 4-fold symmetric), which
#: is refused rather than warned about: every coefficient of a Hermitian operator in the Pauli
#: basis is real, so a complex one is not a small error but a different operator.
HERMITICITY_TOL = 1e-10

#: ``(-i)^k``, the factor converting a ``X^x Z^z`` coefficient to the Hermitian-tensor-product
#: convention: ``c X^x Z^z = c (-i)^popcount(x&z) P(x, z)``.
_MINUS_I_POW = np.array([1.0 + 0.0j, -1.0j, -1.0 + 0.0j, 1.0j], dtype=np.complex128)

#: ``i^k``, its inverse — the phase a Hermitian Pauli string carries when applied as X then Z.
_I_POW = np.array([1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j], dtype=np.complex128)

#: Bytes one stored Pauli term costs: a ``float64`` coefficient and two ``uint64`` masks.
_BYTES_PER_TERM = 24


# --- Sizing --------------------------------------------------------------

def pauli_terms_gb(n_terms: int) -> float:
    """Size [GB] of ``n_terms`` stored Pauli terms. Exact, and never padded."""
    return (res.array_gb((int(n_terms),), np.float64)
            + 2.0 * res.array_gb((int(n_terms),), np.uint64))


def jordan_wigner_terms(n_one: int, n_two: int) -> int:
    """Pauli terms a JW mapping generates **before** collapsing duplicates.

    Each ``a_p^dag a_q`` is a product of two two-term factors (4 terms); each
    ``a_p^dag a_r^dag a_s a_q`` a product of four (16). ``n_one``/``n_two`` are the counts of
    integrals that survived screening, so this is an exact statement about the buffer, not an
    estimate of it — and, since collapsing can only shrink the list, an exact upper bound on
    the resident result. :func:`jordan_wigner` reports the collapsed count it actually got.
    """
    return 4 * int(n_one) + 16 * int(n_two)


def dense_matrix_gb(n_qubit: int) -> float:
    """Size [GB] of the ``2^n x 2^n`` dense matrix :meth:`PauliSum.to_dense` returns."""
    dim = 1 << int(n_qubit)
    return res.array_gb((dim, dim), np.complex128)


def dense_matrix_workspace_gb(n_qubit: int) -> float:
    """Size [GB] of the accumulators :meth:`PauliSum.to_dense` holds *beside* its result.

    Two real ``2^n x 2^n`` arrays: scattered accumulation is done with :func:`numpy.bincount`,
    which is real-valued, so the real and imaginary parts are summed separately and combined
    once at the end. ⚠ Counted rather than ignored because it is the *larger* half of the peak
    — the workspace and the result coexist only at the moment the result is formed, and 32
    bytes per element is what the caller has to have. NumPy's own per-call ``bincount`` output
    is a genuine transient on top and is not counted, the same ``external`` status given to PySCF's
    allocation.
    """
    dim = 1 << int(n_qubit)
    return 2.0 * res.array_gb((dim, dim), np.float64)


# --- The operator -------------------------------------------------------------------------

def pauli_label(x: int, z: int, n_qubit: int) -> str:
    """``"IXYZ..."`` for one Pauli string, **qubit 0 first**.

    ⚠ Kuiva's order, matching the bit order of the masks and of a determinant. Qiskit's label
    convention is the reverse (qubit 0 last); the adapter reverses it, and that reversal is
    the adapter's business and appears nowhere else.
    """
    chars = []
    for k in range(int(n_qubit)):
        bit = 1 << k
        has_x, has_z = bool(int(x) & bit), bool(int(z) & bit)
        chars.append("Y" if (has_x and has_z) else "X" if has_x else "Z" if has_z else "I")
    return "".join(chars)


@dataclass(frozen=True)
class PauliSum:
    """A Hermitian operator as ``sum_t c_t P(x_t, z_t)``: plain arrays, nothing else.

    Terms are unique and sorted by ``(x, z)``, so two mappings of the same Hamiltonian
    compare element by element and a stored operator has a canonical form.

    Attributes
    ----------
    coeffs : ndarray (n_terms,) float64
        Real by construction — see the module docstring.
    x_masks, z_masks : ndarray (n_terms,) uint64
    n_qubit : int
    """

    coeffs: np.ndarray
    x_masks: np.ndarray
    z_masks: np.ndarray
    n_qubit: int

    def __post_init__(self) -> None:
        if not 0 <= int(self.n_qubit) <= MAX_QUBITS:
            raise ValueError("n_qubit must be in [0, {}], got {}".format(
                MAX_QUBITS, self.n_qubit))
        object.__setattr__(self, "n_qubit", int(self.n_qubit))
        object.__setattr__(self, "coeffs",
                           np.ascontiguousarray(self.coeffs, dtype=np.float64))
        object.__setattr__(self, "x_masks", np.ascontiguousarray(self.x_masks, dtype=_U64))
        object.__setattr__(self, "z_masks", np.ascontiguousarray(self.z_masks, dtype=_U64))
        if not (self.coeffs.shape == self.x_masks.shape == self.z_masks.shape):
            raise ValueError("coeffs, x_masks and z_masks must have the same shape; got {}, "
                             "{}, {}".format(self.coeffs.shape, self.x_masks.shape,
                                             self.z_masks.shape))
        if self.coeffs.ndim != 1:
            raise ValueError("a PauliSum is a flat list of terms; got {} dimensions"
                             .format(self.coeffs.ndim))
        limit = _U64((1 << self.n_qubit) - 1) if self.n_qubit < 64 else _U64(0xFFFFFFFFFFFFFFFF)
        if self.n_terms and (np.any(self.x_masks & ~limit) or np.any(self.z_masks & ~limit)):
            raise ValueError("a Pauli string acts on a qubit outside [0, {})"
                             .format(self.n_qubit))

    @property
    def n_terms(self) -> int:
        return int(self.coeffs.size)

    def __len__(self) -> int:
        return self.n_terms

    @property
    def identity_coefficient(self) -> float:
        """Coefficient of the identity string — the constant part of the operator.

        ⚠ Not ``e_core``: this is what normal-ordering the *active-space* operator leaves
        behind, and the inactive energy is added by the driver (the TTNO convention).
        """
        if not self.n_terms:
            return 0.0
        hit = (self.x_masks == _U64(0)) & (self.z_masks == _U64(0))
        return float(self.coeffs[hit].sum())

    @property
    def one_norm(self) -> float:
        """``sum_t |c_t|`` — the standard bound on the operator norm, and the figure shot
        budgets for a sampling or estimation algorithm are quoted against."""
        return float(np.abs(self.coeffs).sum())

    def weights(self) -> np.ndarray:
        """Pauli weight (number of non-identity factors) of each term."""
        return popcount(self.x_masks | self.z_masks)

    def labels(self) -> Tuple[str, ...]:
        """Every term as a ``"IXYZ"`` label, qubit 0 first. Diagnostics and adapters."""
        return tuple(pauli_label(int(x), int(z), self.n_qubit)
                     for x, z in zip(self.x_masks, self.z_masks))

    def drop_below(self, tol: float) -> "PauliSum":
        """Discard terms with ``|c| < tol``. ⚠ A truncation of the operator, not a tidy-up:
        the discarded weight is ``sum |c|`` over what went, and no caller may treat the result
        as the same Hamiltonian."""
        keep = np.abs(self.coeffs) >= float(tol)
        return PauliSum(self.coeffs[keep], self.x_masks[keep], self.z_masks[keep],
                        self.n_qubit)

    def scaled(self, factor: float) -> "PauliSum":
        """``factor * self``. Real factor only — a complex one would break Hermiticity."""
        return PauliSum(self.coeffs * float(factor), self.x_masks, self.z_masks, self.n_qubit)

    def plus(self, other: "PauliSum") -> "PauliSum":
        """``self + other``, duplicate strings summed and the result re-canonicalized.

        ⚠ **The sum is taken in the Hermitian convention**, where coefficients are already
        real, so there is no cancellation check to redo — both operands passed
        :func:`_to_hermitian` when they were built. Terms that cancel to zero are *kept*, at
        their zero coefficient, rather than dropped: a caller comparing two operator sums term
        by term needs the same string list, and :meth:`drop_below` is the explicit way to
        shorten one.
        """
        if other.n_qubit != self.n_qubit:
            raise ValueError("cannot add a {}-qubit operator to a {}-qubit one"
                             .format(other.n_qubit, self.n_qubit))
        c, x, z = _collapse_real(np.concatenate([self.coeffs, other.coeffs]),
                                 np.concatenate([self.x_masks, other.x_masks]),
                                 np.concatenate([self.z_masks, other.z_masks]))
        return PauliSum(c, x, z, self.n_qubit)

    def commuting_pairs(self) -> np.ndarray:
        """``(n_terms, n_terms)`` boolean: whether each pair of strings commutes.

        Full operator commutation, not the qubit-wise kind :func:`qwc_groups` uses:
        ``P(x1,z1)`` and ``P(x2,z2)`` commute iff ``<x1,z2> + <z1,x2>`` is even. ⚠ Quadratic in
        the term count and meant for *validation* — the property it establishes (that a
        Trotter product is exact, :mod:`kuiva.qc.fermionic`) is exactly the kind of claim that
        must be checked rather than argued.
        """
        x, z = self.x_masks, self.z_masks
        anti = (popcount(x[:, None] & z[None, :]) + popcount(z[:, None] & x[None, :])) & 1
        return anti == 0

    def all_commute(self) -> bool:
        """Whether every pair of strings commutes — i.e. whether ``exp`` of this operator
        factorizes **exactly** into per-term exponentials with no Trotter error."""
        return bool(self.commuting_pairs().all())

    def to_dense(self) -> np.ndarray:
        """The ``2^n x 2^n`` matrix, in the computational basis whose index **is** the
        occupation mask (the determinant convention: bit ``k`` = mode ``k``).

        For validation only, and it refuses rather than thrashes: the budgeting requirement is
        raised before the allocation, and the whole point of the qubit representation is that
        this matrix never has to exist.
        """
        dim = 1 << self.n_qubit
        res.require("dense qubit Hamiltonian",
                    dense_matrix_gb(self.n_qubit) + dense_matrix_workspace_gb(self.n_qubit),
                    note="{} qubits, {} Pauli terms (result plus real/imaginary accumulators)"
                         .format(self.n_qubit, self.n_terms),
                    advice=["this is a validation path only; a solver never forms it",
                            "compare on a smaller active space (4-8 spinors is enough to "
                            "pin every sign in the mapping)"])
        basis = np.arange(dim, dtype=_U64)
        flat_re = np.zeros(dim * dim, dtype=np.float64)
        flat_im = np.zeros(dim * dim, dtype=np.float64)
        block = _term_block(dim)
        for lo in range(0, self.n_terms, block):
            sl = slice(lo, min(lo + block, self.n_terms))
            x, z = self.x_masks[sl], self.z_masks[sl]
            rows = basis[None, :] ^ x[:, None]
            sign = 1.0 - 2.0 * (popcount(basis[None, :] & z[:, None]) & 1)
            vals = (self.coeffs[sl] * _I_POW[popcount(x & z) & 3])[:, None] * sign
            flat = (rows.astype(np.int64) * dim + basis.astype(np.int64)[None, :]).ravel()
            flat_re += np.bincount(flat, weights=vals.real.ravel(), minlength=dim * dim)
            flat_im += np.bincount(flat, weights=vals.imag.ravel(), minlength=dim * dim)
        return (flat_re + 1j * flat_im).reshape(dim, dim)

    def __repr__(self) -> str:
        return "PauliSum(n_qubit={}, n_terms={}, |c|_1={:.6g})".format(
            self.n_qubit, self.n_terms, self.one_norm)


def _term_block(dim: int) -> int:
    """How many Pauli terms to process against a ``dim``-long basis at once.

    Asked **once, outside the loop**: the budget is a number the kernel blocks against,
    never a check performed per iteration.
    """
    budget_bytes = res.transient_gb() * (1 << 30)
    per_term = max(1, dim) * 32                   # rows + signs + values, all dim-long
    return int(max(1, min(1 << 20, budget_bytes // per_term)))


# --- Acting with Pauli strings ------------------------------------------------------------
#
# One definition of the phase rule, three consumers (the dense matrix above, the expectation
# value below, and the stub backend's `estimate`). Writing it twice is exactly how the two
# halves of a check come to share a defect.

def pauli_apply(x_masks: np.ndarray, z_masks: np.ndarray, coeffs: np.ndarray,
                psi: np.ndarray) -> np.ndarray:
    """``sum_t c_t P(x_t, z_t) |psi>`` for a statevector of length ``2^n``.

    ``P(x, z)|j> = i^popcount(x&z) (-1)^popcount(j&z) |j ^ x>`` — the whole content of the
    symplectic encoding, and the only place it is written.
    """
    psi = np.ascontiguousarray(psi, dtype=np.complex128)
    dim = psi.size
    x = np.ascontiguousarray(x_masks, dtype=_U64)
    z = np.ascontiguousarray(z_masks, dtype=_U64)
    c = np.ascontiguousarray(coeffs, dtype=np.complex128)
    basis = np.arange(dim, dtype=_U64)
    out = np.zeros(dim, dtype=np.complex128)
    block = _term_block(dim)
    for lo in range(0, x.size, block):
        sl = slice(lo, min(lo + block, x.size))
        sign = 1.0 - 2.0 * (popcount(basis[None, :] & z[sl, None]) & 1)
        amp = (c[sl] * _I_POW[popcount(x[sl] & z[sl]) & 3])[:, None] * sign * psi[None, :]
        rows = (basis[None, :] ^ x[sl, None]).astype(np.int64)
        out += (np.bincount(rows.ravel(), weights=amp.real.ravel(), minlength=dim)
                + 1j * np.bincount(rows.ravel(), weights=amp.imag.ravel(), minlength=dim))
    return out


def pauli_expectation(x_masks: np.ndarray, z_masks: np.ndarray,
                      psi: np.ndarray) -> np.ndarray:
    """``<psi| P(x_t, z_t) |psi>`` for every term, as a real array.

    Real because each ``P`` is Hermitian; the imaginary part is discarded after being checked
    against :data:`HERMITICITY_TOL`, so a broken phase convention cannot pass silently.
    """
    psi = np.ascontiguousarray(psi, dtype=np.complex128)
    dim = psi.size
    x = np.ascontiguousarray(x_masks, dtype=_U64)
    z = np.ascontiguousarray(z_masks, dtype=_U64)
    basis = np.arange(dim, dtype=_U64)
    values = np.empty(x.size, dtype=np.complex128)
    block = _term_block(dim)
    for lo in range(0, x.size, block):
        sl = slice(lo, min(lo + block, x.size))
        sign = 1.0 - 2.0 * (popcount(basis[None, :] & z[sl, None]) & 1)
        rows = (basis[None, :] ^ x[sl, None]).astype(np.int64)
        braket = np.conj(psi[rows]) * (sign * psi[None, :])
        values[sl] = _I_POW[popcount(x[sl] & z[sl]) & 3] * braket.sum(axis=1)
    scale = max(1.0, float(np.abs(values).max()) if values.size else 1.0)
    worst = float(np.abs(values.imag).max()) if values.size else 0.0
    if worst > HERMITICITY_TOL * scale:
        raise ValueError(
            "Pauli expectation values came out complex (max |Im| = {:.3e}); a Pauli string is "
            "Hermitian, so this is a broken phase convention, not a tolerance question"
            .format(worst))
    return np.ascontiguousarray(values.real)


# --- Symplectic algebra used to build the mapping ------------------------------------------
#
# Accumulation happens in the ``X^x Z^z`` convention, where multiplication is one XOR pair and
# one sign, and coefficients are complex. The conversion to the Hermitian convention — and the
# check that the imaginary parts have cancelled — happens once, at the end.

def _multiply(c1, x1, z1, c2, x2, z2):
    """``(c1 X^x1 Z^z1)(c2 X^x2 Z^z2)``, elementwise over broadcastable arrays.

    ``(X^x1 Z^z1)(X^x2 Z^z2) = (-1)^<z1, x2> X^(x1^x2) Z^(z1^z2)``, with ``<z1, x2>`` the
    parity of ``popcount(z1 & x2)``: commuting ``Z^z1`` past ``X^x2`` costs one sign per qubit
    where both act.
    """
    sign = 1.0 - 2.0 * (popcount(z1 & x2) & 1)
    return c1 * c2 * sign, x1 ^ x2, z1 ^ z2


def _ladder_terms(modes: np.ndarray, dagger: bool):
    """``a_p`` or ``a_p^dag`` for a batch of modes, as ``(len(modes), 2)`` arrays.

    From the module docstring's convention, in the ``X^x Z^z`` form::

        a_p     = 1/2 (Zstring_p X_p) - 1/2 (Zstring_p X_p Z_p)
        a_p^dag = 1/2 (Zstring_p X_p) + 1/2 (Zstring_p X_p Z_p)

    using ``Y = i X Z``. ``Zstring_p`` is ``below_mask(p)`` — imported from
    ``ci/strings.py``, where the "below p" index set of the fermionic sign rule is defined.
    """
    p = np.asarray(modes, dtype=np.int64)
    bit = mode_bit(p)
    string = below_mask(p)
    x = np.stack([bit, bit], axis=1)
    z = np.stack([string, string | bit], axis=1)
    second = 0.5 if dagger else -0.5
    c = np.stack([np.full(p.shape, 0.5), np.full(p.shape, second)],
                 axis=1).astype(np.complex128)
    return c, x, z


def _product(factors):
    """Product of ordered ladder factors, each ``(B, 2)``, giving ``(B, 2^k)``."""
    c, x, z = factors[0]
    for c2, x2, z2 in factors[1:]:
        c, x, z = _multiply(c[:, :, None], x[:, :, None], z[:, :, None],
                            c2[:, None, :], x2[:, None, :], z2[:, None, :])
        c = c.reshape(c.shape[0], -1)
        x = x.reshape(x.shape[0], -1)
        z = z.reshape(z.shape[0], -1)
    return c, x, z


def _collapse(coeffs, x, z):
    """Sum duplicate Pauli strings; return terms sorted by ``(x, z)``.

    ⚠ **Reduction order note.** The sum over duplicates is ``np.add.reduceat`` over a
    lexicographically sorted list, so it is deterministic and reproducible run to run, but it
    is *not* the order the terms were generated in. A future implementation that reorders it
    (a threaded or compiled one) changes the last bits of a coefficient. That is stated rather
    than assumed because the Hermiticity check below is a cancellation test, and cancellation
    is where reduction order shows.
    """
    if coeffs.size == 0:
        return (np.zeros(0, dtype=np.complex128), np.zeros(0, dtype=_U64),
                np.zeros(0, dtype=_U64))
    order = np.lexsort((z, x))
    x, z, c = x[order], z[order], coeffs[order]
    boundary = np.empty(x.size, dtype=bool)
    boundary[0] = True
    boundary[1:] = (x[1:] != x[:-1]) | (z[1:] != z[:-1])
    starts = np.flatnonzero(boundary)
    return np.add.reduceat(c, starts), x[starts], z[starts]


def _collapse_real(coeffs, x, z):
    """:func:`_collapse` for coefficients already in the real Hermitian convention."""
    if coeffs.size == 0:
        return (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=_U64),
                np.zeros(0, dtype=_U64))
    order = np.lexsort((z, x))
    x, z, c = x[order], z[order], coeffs[order]
    boundary = np.empty(x.size, dtype=bool)
    boundary[0] = True
    boundary[1:] = (x[1:] != x[:-1]) | (z[1:] != z[:-1])
    starts = np.flatnonzero(boundary)
    return np.add.reduceat(c, starts), x[starts], z[starts]


def _to_hermitian(coeffs, x, z, tol: float):
    """Convert ``X^x Z^z`` coefficients to the Hermitian convention and refuse a complex one.

    ``c X^x Z^z = c (-i)^popcount(x&z) P(x, z)``. The imaginary part of the result is the
    cancellation the 4-fold symmetry of the ERI and the Hermiticity of ``h`` are supposed to
    produce; if it survives, the *input* was not what this function was told it was.
    """
    herm = coeffs * _MINUS_I_POW[popcount(x & z) & 3]
    scale = max(1.0, float(np.abs(herm).max()) if herm.size else 1.0)
    worst = float(np.abs(herm.imag).max()) if herm.size else 0.0
    if worst > HERMITICITY_TOL * scale:
        raise ValueError(
            "the qubit Hamiltonian has complex Pauli coefficients (max |Im| = {:.3e}, "
            "relative {:.3e}). Every coefficient of a Hermitian operator in the Pauli basis "
            "is real, so this means the input was not Hermitian: check that h_pq = h_qp^* and "
            "that the ERI has the 4-fold symmetry (pq|rs) = (rs|pq), (pq|rs)^* = (qp|sr) "
            "(kuiva.integrals.transform.check_permutational_symmetry)."
            .format(worst, worst / scale))
    keep = np.abs(herm.real) > float(tol)
    return np.ascontiguousarray(herm.real[keep]), x[keep], z[keep]


# --- Arbitrary ladder-operator strings ------------------------------------------------------
#
# The Hamiltonian is one consumer of this. The others are the excitation *generators* an ansatz
# exponentiates (:mod:`kuiva.qc.fermionic`) and the RDM operators a VQE has to measure
# (:func:`rdm_measurement`) — three consumers of one convention, which is the only arrangement
# in which a sign error cannot hide in two of them agreeing.

def _ladder_products(modes: np.ndarray, daggers: Sequence[bool]):
    """``(B, 2^k)`` coefficient/x/z arrays for the ordered ladder product of each row.

    ``modes[b, j]`` is the mode of the ``j``-th factor of term ``b`` and ``daggers[j]`` says
    whether that factor is a creation operator. Factors are applied **left to right in the
    order given**, which is the order the operator is written in — ``a_p^dag a_r^dag a_s a_q``
    is ``modes=(p, r, s, q)``, ``daggers=(True, True, False, False)``.
    """
    modes = np.atleast_2d(np.asarray(modes, dtype=np.int64))
    if modes.shape[1] != len(daggers):
        raise ValueError("{} mode columns but {} dagger flags".format(modes.shape[1],
                                                                      len(daggers)))
    return _product([_ladder_terms(modes[:, j], bool(daggers[j]))
                     for j in range(len(daggers))])


def _row_block(n_factors: int) -> int:
    """Rows of a ladder-product batch to process at once, asked **once**, outside the loop.

    A product of ``n_factors`` ladder operators expands into ``2^n_factors`` Pauli strings, so
    that — not the rank — is what the buffer scales with (the budget is a number the
    kernel blocks against, never a check inside the loop).
    """
    per_row = (1 << int(n_factors)) * (_BYTES_PER_TERM + 16)
    return int(max(1, (res.transient_gb() * (1 << 30)) // max(1, per_row)))


def _mapped_parts(coeffs: np.ndarray, modes: np.ndarray, daggers: Sequence[bool]):
    """Collapsed ``X^x Z^z`` parts of ``sum_b coeffs[b] * (ladder product of row b)``."""
    modes = np.atleast_2d(np.asarray(modes, dtype=np.int64))
    coeffs = np.ascontiguousarray(coeffs, dtype=np.complex128)
    if coeffs.shape != (modes.shape[0],):
        raise ValueError("{} coefficients for {} mode rows".format(coeffs.size,
                                                                   modes.shape[0]))
    parts = []
    block = _row_block(modes.shape[1])
    for lo in range(0, modes.shape[0], block):
        sl = slice(lo, min(lo + block, modes.shape[0]))
        c, x, z = _ladder_products(modes[sl], daggers)
        parts.append(_collapse((c * coeffs[sl][:, None]).ravel(), x.ravel(), z.ravel()))
    return parts


def fermionic_operator(coeffs, modes, daggers, n_qubit: int, *,
                       tol: float = INTEGRAL_SCREEN_TOL) -> PauliSum:
    """Jordan-Wigner image of an arbitrary sum of ladder-operator strings.

    ``sum_b c_b a^{d_0}_{m_b0} ... a^{d_{k-1}}_{m_b,k-1}``, all rows sharing the dagger
    pattern ``daggers``. Several patterns are combined by :meth:`PauliSum.plus`.

    ⚠ **The result must be Hermitian and is refused if it is not** — same check, same reason
    and same message as :func:`jordan_wigner`'s. A generator like ``T - T^dag`` is
    anti-Hermitian, so pass ``i(T - T^dag)``, which is Hermitian, and exponentiate it as
    ``exp(-i A)``; :mod:`kuiva.qc.fermionic` does exactly that and is the reason this function
    is public.
    """
    n_qubit = int(n_qubit)
    if n_qubit > MAX_QUBITS:
        raise ValueError("at most {} qubits, got {}".format(MAX_QUBITS, n_qubit))
    modes = np.atleast_2d(np.asarray(modes, dtype=np.int64))
    if modes.size and (modes.min() < 0 or modes.max() >= n_qubit):
        raise ValueError("mode indices must lie in [0, {}), got [{}, {}]".format(
            n_qubit, int(modes.min()), int(modes.max())))
    parts = _mapped_parts(np.asarray(coeffs, dtype=np.complex128), modes, daggers)
    if parts:
        c, x, z = _collapse(np.concatenate([p[0] for p in parts]),
                            np.concatenate([p[1] for p in parts]),
                            np.concatenate([p[2] for p in parts]))
    else:
        c, x, z = (np.zeros(0, dtype=np.complex128), np.zeros(0, dtype=_U64),
                   np.zeros(0, dtype=_U64))
    real, x, z = _to_hermitian(c, x, z, tol)
    return PauliSum(real, x, z, n_qubit)


# --- Qubit-wise commuting groups -------------------------------------------------------------

def pauli_commute(x1, z1, x2, z2) -> np.ndarray:
    """Whether ``P(x1,z1)`` and ``P(x2,z2)`` commute, elementwise over broadcastable arrays."""
    return ((popcount(np.asarray(x1, _U64) & np.asarray(z2, _U64))
             + popcount(np.asarray(z1, _U64) & np.asarray(x2, _U64))) & 1) == 0


def qwc_groups(operator: PauliSum) -> Tuple[np.ndarray, ...]:
    """Partition the terms into **qubit-wise commuting** sets, one measurement basis each.

    Two strings are qubit-wise commuting when, on every qubit where both act, they carry the
    *same* Pauli. A whole such group is then measured with **one** circuit: rotate each qubit
    into the group's basis, measure in the computational basis once, and read every term's
    parity off the same bitstrings.

    ⚠ **Qubit-wise commuting is strictly stronger than commuting**, and deliberately so: a
    generally commuting group needs a Clifford diagonalization circuit, which is depth on
    hardware and a second thing to get wrong. This is the cheap, safe grouping every framework
    starts with, and the reduction it buys is measured rather than assumed —
    ``tests/test_qc_mapping.py`` reports it.

    ⚠ **Grouping breaks the independence assumption of**
    :meth:`~kuiva.qc.backend.EstimateResult.combine`. Terms measured on the same circuit are
    correlated: their covariance is exactly what the grouping buys (and it can help or hurt the
    variance of a *sum*), so a combined variance computed term by term is no longer exact. The
    result records its groups so a caller can see this rather than inherit it silently.

    Greedy largest-weight-first assignment against a running per-group basis. Not optimal —
    minimum clique cover is NP-hard — and the greedy result is what every implementation in the
    field uses.

    Returns
    -------
    tuple of ndarray
        Term indices per group, each ascending; the concatenation is a permutation of
        ``range(operator.n_terms)``.
    """
    return qwc_groups_from_masks(operator.x_masks, operator.z_masks)


def qwc_groups_from_masks(x_masks: np.ndarray, z_masks: np.ndarray) -> Tuple[np.ndarray, ...]:
    """:func:`qwc_groups` on raw symplectic masks — what a backend adapter has in hand."""
    x = np.ascontiguousarray(x_masks, dtype=_U64)
    z = np.ascontiguousarray(z_masks, dtype=_U64)
    n = int(x.size)
    if n == 0:
        return ()
    support = x | z
    order = np.argsort(-popcount(support), kind="stable")
    basis_x: list = []
    basis_z: list = []
    basis_s: list = []
    groups: list = []
    for t in order.tolist():
        xt, zt, st = int(x[t]), int(z[t]), int(support[t])
        for g in range(len(groups)):
            overlap = st & basis_s[g]
            if (xt & overlap) == (basis_x[g] & overlap) and \
               (zt & overlap) == (basis_z[g] & overlap):
                groups[g].append(t)
                basis_x[g] |= xt
                basis_z[g] |= zt
                basis_s[g] |= st
                break
        else:
            groups.append([t])
            basis_x.append(xt)
            basis_z.append(zt)
            basis_s.append(st)
    return tuple(np.array(sorted(g), dtype=np.int64) for g in groups)


# --- RDM operators, for a solver that has to *measure* its densities --------------------------

def rdm_operator_gb(n_qubit: int, rank: int) -> float:
    """Size [GB] of the sparse coefficient matrix :func:`rdm_measurement` builds.

    ``n^(2*rank)`` rows with ``4^rank`` stored entries each — a ``complex128`` value, an
    ``int32`` column index and the CSR row pointer. Exact, never padded.
    """
    rows = int(n_qubit) ** (2 * int(rank))
    nnz = rows * (1 << (2 * int(rank)))
    return (res.array_gb((nnz,), np.complex128) + res.array_gb((nnz,), np.int32)
            + res.array_gb((rows + 1,), np.int32))


@dataclass(frozen=True)
class RDMMeasurement:
    """A plan for obtaining ``gamma`` (and ``Gamma``) from Pauli **expectation values**.

    Why this exists at all: SQD gets its RDMs from the classical subspace diagonalization and
    never measures an operator. A VQE does not have that luxury — the state lives in the
    device, so every density-matrix element is a measurement, and *this* is the NISQ bottleneck
    SQD was chosen to sidestep. Making the cost explicit and countable is the point:
    :attr:`n_strings` and ``len(qwc_groups(...))`` are the honest figures.

    ``gamma_pq = <a_p^dag a_q>`` and ``Gamma_pqrs = <a_p^dag a_r^dag a_s a_q>`` — Kuiva's
    convention (``ci/strings.py``), so the result plugs into the ci_solver contract unchanged.

    ⚠ **Individual RDM elements are complex, and that is not a contradiction.** Each Pauli
    string is Hermitian with a real expectation value; ``a_p^dag a_q`` is not Hermitian, so its
    *coefficients* over those strings are complex. Hermiticity of the assembled ``gamma`` is a
    property of the whole set, and :meth:`gamma` checks it rather than symmetrizing silently.

    Attributes
    ----------
    x_masks, z_masks : ndarray (n_strings,) uint64
        The unique Pauli strings that must be measured — the union over all elements.
    one_body : scipy.sparse matrix ``(n^2, n_strings)`` complex
    two_body : scipy.sparse matrix ``(n^4, n_strings)`` complex, or ``None``
    """

    x_masks: np.ndarray
    z_masks: np.ndarray
    one_body: Any
    two_body: Optional[Any]
    n_qubit: int

    @property
    def n_strings(self) -> int:
        return int(self.x_masks.size)

    def gamma(self, values: np.ndarray) -> np.ndarray:
        """Assemble ``gamma`` from the expectation values of :attr:`x_masks`/:attr:`z_masks`.

        ⚠ Hermiticity is **checked and reported, never imposed.** A shot-noisy set of
        expectation values gives a ``gamma`` that is Hermitian only to the noise level, and
        symmetrizing it silently would hide exactly the quantity a caller needs in order to
        know whether the measurement budget was enough.
        """
        n = self.n_qubit
        out = np.ascontiguousarray(
            np.asarray(self.one_body @ np.asarray(values, dtype=np.complex128)).reshape(n, n))
        defect = float(np.abs(out - out.conj().T).max()) if n else 0.0
        if defect > 1e-10:
            log.warning("the measured 1-RDM is Hermitian only to %.3e; with shot noise this "
                        "is the measurement error, not a defect", defect)
        return out

    def gamma2(self, values: np.ndarray) -> np.ndarray:
        """Assemble ``Gamma``. Raises if the plan was built with ``rank=1``."""
        if self.two_body is None:
            raise ValueError("this RDMMeasurement was built for the 1-RDM only (rank=1)")
        n = self.n_qubit
        out = np.asarray(self.two_body @ np.asarray(values, dtype=np.complex128))
        return np.ascontiguousarray(out.reshape(n, n, n, n))

    def __repr__(self) -> str:
        return "RDMMeasurement(n_qubit={}, n_strings={}, rank={})".format(
            self.n_qubit, self.n_strings, 1 if self.two_body is None else 2)


def _triplets(coeffs, x, z, index):
    """``(rows, cols, data)`` for one row-labelled batch of Pauli products.

    ⚠ Returns triplets rather than a matrix because the *column count is not known yet*: the
    two-body block interns strings the one-body block never saw, so both matrices are assembled
    once, at the end, against the final string list. Building the first one early gives two
    matrices of different widths that multiply the same expectation-value vector — which is a
    shape error on a good day and a silent misread on a bad one.
    """
    n_rows = coeffs.shape[0]
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), coeffs.shape[1])
    herm = (coeffs * _MINUS_I_POW[popcount(x & z) & 3]).ravel()
    cols = index(np.stack([x.ravel(), z.ravel()], axis=1))
    keep = np.abs(herm) > 0.0
    return rows[keep], cols[keep], herm[keep]


class _StringIndex:
    """Interns ``(x, z)`` pairs to column indices — one shared string list for both ranks."""

    def __init__(self) -> None:
        self._map: Dict[Tuple[int, int], int] = {}
        self.x: list = []
        self.z: list = []

    @property
    def size(self) -> int:
        return len(self.x)

    def __call__(self, keys: np.ndarray) -> np.ndarray:
        out = np.empty(keys.shape[0], dtype=np.int64)
        table = self._map
        for i, (xi, zi) in enumerate(keys.tolist()):
            key = (int(xi), int(zi))
            hit = table.get(key)
            if hit is None:
                hit = len(self.x)
                table[key] = hit
                self.x.append(key[0])
                self.z.append(key[1])
            out[i] = hit
        return out


def rdm_measurement(n_qubit: int, *, rank: int = 2) -> RDMMeasurement:
    """Build the measurement plan for the 1-RDM (``rank=1``) or the 1- and 2-RDMs.

    ⚠ **The cost is the message.** The 2-RDM plan enumerates ``n^4`` operators of 16 Pauli
    strings each; the *distinct* strings are far fewer, and qubit-wise grouping cuts the
    circuit count further, but the scaling is what it is. Compare with SQD, which measures
    ``sample`` and nothing else.
    """
    n_qubit, rank = int(n_qubit), int(rank)
    if rank not in (1, 2):
        raise ValueError("rank must be 1 or 2, got {}".format(rank))
    res.require("RDM measurement plan", rdm_operator_gb(n_qubit, rank),
                note="{} qubits, rank {}".format(n_qubit, rank),
                advice=["rank=1 if only the 1-RDM is needed",
                        "a smaller active space; this is the tomography cost a sampled-"
                        "subspace solver does not pay at all"])
    from scipy.sparse import csr_matrix

    index = _StringIndex()
    pq = np.stack(np.meshgrid(np.arange(n_qubit), np.arange(n_qubit), indexing="ij"),
                  axis=-1).reshape(-1, 2)
    c, x, z = _ladder_products(pq, (True, False))
    one_t = _triplets(c, x, z, index)
    two_t = None
    if rank == 2:
        grid = np.meshgrid(*(np.arange(n_qubit),) * 4, indexing="ij")
        p, q, r, s = (g.reshape(-1) for g in grid)
        c, x, z = _ladder_products(np.stack([p, r, s, q], axis=1), (True, True, False, False))
        two_t = _triplets(c, x, z, index)
    x_masks = np.array(index.x, dtype=_U64)
    z_masks = np.array(index.z, dtype=_U64)
    width = int(x_masks.size)
    one = csr_matrix((one_t[2], (one_t[0], one_t[1])), shape=(n_qubit ** 2, width),
                     dtype=np.complex128)
    two = None if two_t is None else csr_matrix(
        (two_t[2], (two_t[0], two_t[1])), shape=(n_qubit ** 4, width), dtype=np.complex128)
    log.debug("RDM measurement plan: %d qubits, rank %d -> %d distinct Pauli strings",
              n_qubit, rank, width)
    return RDMMeasurement(x_masks=x_masks, z_masks=z_masks, one_body=one,
                          two_body=two, n_qubit=n_qubit)


# --- The mapping --------------------------------------------------------------------------

def jordan_wigner(h: np.ndarray, eri: np.ndarray, *,
                  tol: float = INTEGRAL_SCREEN_TOL) -> PauliSum:
    """Map the active-space Hamiltonian to qubits under Jordan-Wigner.

    Parameters
    ----------
    h : ndarray (n, n) complex
        The one-electron active-space matrix — ``CASIntegrals.h_active_effective()``, i.e. the
        inactive Fock restricted to the active spinors. Must be Hermitian.
    eri : ndarray (n, n, n, n) complex
        ``(pq|rs)`` in chemists' notation over the same spinors —
        ``CASIntegrals.active_eri()`` — with 4-fold permutational symmetry only.
    tol : float
        Integrals below this magnitude are skipped and final coefficients below it dropped.
        See :data:`INTEGRAL_SCREEN_TOL`. ``0.0`` maps every integral.

    Returns
    -------
    PauliSum
        Real coefficients, unique strings sorted by ``(x, z)``. ⚠ Excludes ``e_core``.
    """
    h = np.ascontiguousarray(h, dtype=np.complex128)
    eri = np.ascontiguousarray(eri, dtype=np.complex128)
    n = int(h.shape[0])
    if h.shape != (n, n):
        raise ValueError("h must be square, got {}".format(h.shape))
    if eri.shape != (n, n, n, n):
        raise ValueError("eri must be ({0}, {0}, {0}, {0}) for a {0}-spinor active space, got "
                         "{1}".format(n, eri.shape))
    if n > MAX_QUBITS:
        raise ValueError(
            "active spaces beyond {} spinors need multi-word masks in ci/strings.py before "
            "they can be mapped to qubits; got {}".format(MAX_QUBITS, n))

    one_idx = np.array(np.nonzero(np.abs(h) > tol)).T                   # (n1, 2)
    two_idx = np.array(np.nonzero(np.abs(eri) > tol)).T                 # (n2, 4)
    n_pre = jordan_wigner_terms(len(one_idx), len(two_idx))
    res.require("Jordan-Wigner Pauli sum", pauli_terms_gb(n_pre),
                note="{} qubits, {} one-electron and {} two-electron integrals retained"
                     .format(n, len(one_idx), len(two_idx)),
                advice=["a smaller active space",
                        "a larger screening tolerance (tol=) — but it is a truncation of the "
                        "Hamiltonian, not a tidy-up"])

    with timer("Jordan-Wigner mapping"):
        # Both blocks go through the shared ladder-product path of :func:`fermionic_operator`,
        # which is also what an excitation generator and an RDM operator are built from — one
        # convention with three consumers rather than three copies of it.
        parts = []
        if len(one_idx):
            p, q = one_idx[:, 0], one_idx[:, 1]
            parts.extend(_mapped_parts(h[p, q], one_idx, (True, False)))
        if len(two_idx):
            p, q, r, s = (two_idx[:, k] for k in range(4))
            parts.extend(_mapped_parts(0.5 * eri[p, q, r, s],
                                       np.stack([p, r, s, q], axis=1),
                                       (True, True, False, False)))
        if parts:
            c, x, z = _collapse(np.concatenate([p_[0] for p_ in parts]),
                                np.concatenate([p_[1] for p_ in parts]),
                                np.concatenate([p_[2] for p_ in parts]))
        else:
            c, x, z = _collapse(np.zeros(0, dtype=np.complex128), np.zeros(0, dtype=_U64),
                                np.zeros(0, dtype=_U64))
        coeffs, x, z = _to_hermitian(c, x, z, tol)

    result = PauliSum(coeffs, x, z, n)
    log.debug("Jordan-Wigner: %d qubits, %d integrals -> %d Pauli terms (from %d before "
              "collapsing), |c|_1 = %.6g, max weight %d", n,
              len(one_idx) + len(two_idx), result.n_terms, n_pre, result.one_norm,
              int(result.weights().max()) if result.n_terms else 0)
    return result


# --- The registry -------------------------------------------------------------------------

_MAPPINGS: Dict[str, Callable[..., PauliSum]] = {}


def register_mapping(name: str, impl: Callable[..., PauliSum]) -> None:
    """Register a fermion-to-qubit mapping under ``name``."""
    if not callable(impl):
        raise TypeError("mapping {!r} is not callable".format(name))
    _MAPPINGS[name.lower()] = impl


def available_mappings() -> Tuple[str, ...]:
    return tuple(sorted(_MAPPINGS))


def resolve_mapping(name: str = "jordan_wigner") -> Callable[..., PauliSum]:
    """The mapping registered under ``name``; refuse, naming what is available."""
    key = str(name).lower()
    if key not in _MAPPINGS:
        raise ValueError(
            "unknown fermion-to-qubit mapping {!r}; registered: {}. Parity and Bravyi-Kitaev "
            "are deliberately not registered until they are implemented — a name that "
            "resolves to something non-functional fails further from its cause."
            .format(name, ", ".join(available_mappings()) or "(none)"))
    return _MAPPINGS[key]


register_mapping("jordan_wigner", jordan_wigner)


def qubit_hamiltonian(ints, *, mapping: str = "jordan_wigner",
                      tol: float = INTEGRAL_SCREEN_TOL) -> PauliSum:
    """Map a ``CASIntegrals``-shaped object's active space to a qubit Hamiltonian.

    ``ints`` is **duck-typed** on ``h_active_effective()`` and ``active_eri()``, exactly as
    ``dmrg.ttno.ttno_from_cas_integrals`` is, so this module needs no import from
    :mod:`kuiva.mcscf` and a caller may pass any object presenting those two methods.

    ⚠ ``ints.e_core`` is **not** included, matching ``SigmaOperator`` and the TTNO: the
    qubit Hamiltonian is the active-space operator alone and the driver adds the inactive
    energy to whatever eigenvalue comes back.
    """
    return resolve_mapping(mapping)(ints.h_active_effective(), ints.active_eri(), tol=tol)


__all__ = ["HERMITICITY_TOL", "INTEGRAL_SCREEN_TOL", "MAX_QUBITS", "PauliSum",
           "RDMMeasurement", "available_mappings", "dense_matrix_gb",
           "dense_matrix_workspace_gb", "fermionic_operator",
           "jordan_wigner", "jordan_wigner_terms",
           "pauli_apply", "pauli_commute", "pauli_expectation", "pauli_label",
           "pauli_terms_gb", "qubit_hamiltonian", "qwc_groups", "qwc_groups_from_masks",
           "rdm_measurement",
           "rdm_operator_gb", "register_mapping", "resolve_mapping"]
