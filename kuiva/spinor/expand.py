"""Scalar MO -> Kramers-paired two-component spinor guess.

The reference orbitals are real, scalar-relativistic X2C MOs: spin-free, so each spatial
orbital is doubly degenerate under spin. The two-component wavefunction is built on
**spinors**, and the guess for them is the trivial one — each scalar spatial orbital ``phi_p``
becomes the Kramers pair ``(psi_p, T psi_p)``. There is no spin-orbit coupling in the guess;
SOC enters when the two-component wavefunction is built and optimized.

Conventions fixed here (part of the interface; CI addressing depends on them)
-----------------------------------------------------------------------------------
**1. Row layout: spin-blocked.** A spinor coefficient array has shape ``(2*nbas, nspinor)``
with the first ``nbas`` rows the alpha (spin-up) component and the last ``nbas`` the beta
component of the *same* underlying scalar basis (AO, or the orthonormal working basis —
the expansion is basis-agnostic and is normally applied after orthogonalization)::

    C = [ C_alpha ]      C_alpha, C_beta : (nbas, nspinor)
        [ C_beta  ]

Blocked, not interleaved, because the integral transformation contracts the two spin
components separately against a **spin-free** AO integral (only ``sum_sigma C^sigma* ...
C^sigma`` appears; see :mod:`kuiva.integrals.transform`), and blocked rows make each spin
component a contiguous slice — one GEMM operand, no gather.

**2. Column layout: interleaved Kramers pairs.** Spinor ``2p`` is the unbarred partner of
scalar orbital ``p`` and spinor ``2p+1`` its barred partner ``T(2p)``::

    columns:  0    1    2    3   ...        (p, pbar, q, qbar, ...)
    from:     phi_0     phi_1    ...

Interleaved, not blocked, because every orbital space in the code is defined on *spatial*
orbitals — an active space of scalar orbitals ``[m, m+1, ..., n)`` must map to a contiguous
range of spinors ``[2m, 2n)``, and a Kramers pair must never be split across a space boundary
(state averaging over Kramers pairs and the planned Kramers-restricted CI both depend
on it). :func:`kramers_block_permutation` converts to the blocked ordering for the algorithms
that prefer it; use it, do not re-derive the permutation.

**3. Time reversal.** ``T = -i sigma_y K`` acting on the two-component coefficient blocks::

    T (C_alpha, C_beta) = (-conj(C_beta), conj(C_alpha))

``K`` is complex conjugation of the *coefficients* only, which presumes the underlying scalar
basis functions are **real**. That holds for the real solid-harmonic Gaussians PySCF gives us
and for any real orthogonalization of them, and it is checked where it can be. If a
complex basis is ever introduced, this is the line that breaks. ``T^2 = -1`` on any single
spinor, which is the origin of Kramers degeneracy and is asserted as a Tier-0 test.

With real ``phi_p``, the guess is ``psi_{2p} = (phi_p, 0)`` and ``psi_{2p+1} = (0, phi_p)``:
the barred partner carries the *unconjugated* coefficients, and the sign convention above is
what makes ``T^2 = -1`` rather than ``+1``. Nothing physical depends on the sign, but the CI
phase bookkeeping of the Kramers-restricted path does, so it is fixed here.

Storage
-------
``complex128``, C-contiguous, (complex arithmetic is first-class in the
multireference layer). The guess is real and extremely sparse — each column has one nonzero
spin block — but it is stored densely and complex: after the first CASSCF rotation it is
neither, and a fast path for a structure that survives one iteration would buy little and
would have to be maintained forever. (The transform does exploit the one part of that
structure that survives: the AO integrals are spin-free.)

References
----------
* Time reversal and Kramers pairs: H. A. Kramers, Proc. Amsterdam Acad. 33, 959 (1930);
  E. P. Wigner, "Gruppentheorie und ihre Anwendung...", Vieweg (1931), ch. 26.
* Kramers-paired spinor bases, the ``-i sigma_y K`` convention and the barred/unbarred
  notation: K. G. Dyall, K. Faegri, "Introduction to Relativistic Quantum Chemistry", Oxford
  University Press (2007), ch. 6 and 10; T. Saue, "Relativistic Hamiltonians for Chemistry",
  ChemPhysChem 12, 3077 (2011), doi:10.1002/cphc.201100682.
* Time-reversal-adapted (Kramers-restricted) two-component MCSCF/CI, where this ordering is
  the standard one: J. Thyssen, T. Fleig, H. J. Aa. Jensen, J. Chem. Phys. 129, 034109 (2008),
  doi:10.1063/1.2943670; T. Fleig, J. Olsen, L. Visscher, J. Chem. Phys. 119, 2963 (2003),
  doi:10.1063/1.1590636.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..util import output as out
from ..util.logging import get_logger

log = get_logger(__name__)


# --- Index algebra of the interleaved Kramers ordering -----------------------------------

def unbarred(p: int) -> int:
    """Spinor index of the unbarred partner of scalar orbital ``p``."""
    return 2 * p


def barred(p: int) -> int:
    """Spinor index of the barred (time-reversed) partner of scalar orbital ``p``."""
    return 2 * p + 1


def spatial_index(i: int) -> int:
    """Scalar orbital a spinor index came from."""
    return i // 2


def is_barred(i: int) -> bool:
    return bool(i % 2)


def spinor_indices(spatial: Sequence[int]) -> np.ndarray:
    """Spinor indices of a set of scalar orbitals, both partners, in order.

    This is how a scalar active-space definition ("lowest orbitals of a given angular
    momentum character") becomes a spinor active space, and it is the only sanctioned way to
    do it — a Kramers pair must never be split (see the module docstring).
    """
    spatial = np.asarray(spatial, dtype=int).ravel()
    return np.stack([2 * spatial, 2 * spatial + 1], axis=1).ravel()


def kramers_block_permutation(nspinor: int) -> np.ndarray:
    """Permutation from the interleaved ordering to blocked ``[all unbarred | all barred]``.

    ``c[:, perm]`` is the blocked form. Kramers-restricted algorithms want the blocked
    layout because time reversal is then a single 2x2 block operation on the whole array.
    """
    if nspinor % 2:
        raise ValueError("a Kramers-paired spinor set has an even number of spinors, "
                         "got {}".format(nspinor))
    npair = nspinor // 2
    return np.concatenate([2 * np.arange(npair), 2 * np.arange(npair) + 1])


# --- Time reversal ------------------------------------------------------------------------

def time_reverse(c: np.ndarray) -> np.ndarray:
    """Apply ``T = -i sigma_y K`` to spinor coefficient columns ``(2*nbas, n)``.

    ``T (C_alpha, C_beta) = (-conj(C_beta), conj(C_alpha))``. Applying it twice returns
    ``-C``; that is the content of Kramers degeneracy and a Tier-0 test.
    """
    c = np.asarray(c)
    if c.shape[0] % 2:
        raise ValueError("spinor coefficients must have an even number of rows "
                         "(alpha block then beta block), got {}".format(c.shape[0]))
    nbas = c.shape[0] // 2
    out_c = np.empty_like(c, dtype=np.complex128)
    out_c[:nbas] = -np.conj(c[nbas:])
    out_c[nbas:] = np.conj(c[:nbas])
    return out_c


def nearest_kramers_paired(c: np.ndarray, blocks) -> np.ndarray:
    """The closest Kramers-paired orbital set to ``c``, restored block by block.

    A Kramers-unrestricted optimization is entitled to drift off exact pairing — a truncated
    (selected-CI) state is not closed under time reversal, so the orbitals it optimizes are
    not either — but every consumer of the pairing convention (the state-averaging gate,
    a contiguous-pair active space) assumes pairs. This restores them:

    1. per block, the span is replaced by the **nearest time-reversal-closed subspace** of
       the same dimension — the dominant eigenspace of the symmetrized projector
       ``(P + T P T^-1) / 2``, which commutes with ``T`` by construction;
    2. inside that subspace, columns are rebuilt as explicit ``(u, T u)`` pairs in the
       interleaved layout (``<u|T u> = 0`` identically, since ``T^2 = -1``);
    3. blocks are processed in the given order and each is projected off the ones already
       rebuilt, so the result is one orthonormal set with every block span moved as little
       as possible.

    Parameters
    ----------
    c : ``(2*nbas, n)`` complex, over an **orthonormal real** scalar basis (the working
        basis; identity metric — transform first when holding AO-basis coefficients).
    blocks : sequence of index arrays partitioning the columns (e.g. inactive, active,
        virtual), each of even size and pair-aligned.

    Raises ``ValueError`` when a block's span is so far from time-reversal closed that the
    nearest closed subspace is ambiguous (a symmetrized-projector eigenvalue at or below
    1/2) — at that point the orbitals do not *have* a meaningful paired form.
    """
    c = np.ascontiguousarray(c, dtype=np.complex128)
    out = np.array(c, copy=True)
    finished: list = []
    for idx in blocks:
        idx = np.asarray(idx, dtype=int).ravel()
        if idx.size == 0:
            continue
        if idx.size % 2:
            raise ValueError("a Kramers-paired block has an even number of columns; block "
                             "{} has {}".format(idx.tolist(), idx.size))
        b = c[:, idx]
        if finished:
            q = np.concatenate(finished, axis=1)
            b = b - q @ (q.conj().T @ b)
        b, _ = np.linalg.qr(b)
        tb = time_reverse(b)
        d = 0.5 * (b @ b.conj().T + tb @ tb.conj().T)
        w, v = np.linalg.eigh(d)
        keep = np.argsort(w)[::-1][:idx.size]
        if w[keep[-1]] <= 0.5 + 1e-6:
            raise ValueError(
                "the block on columns {}.. is too far from time-reversal closed to repair: "
                "the symmetrized projector's smallest kept eigenvalue is {:.3f} (a closed "
                "span has 1.0, and at 0.5 the nearest closed subspace is ambiguous)"
                .format(int(idx[0]), float(w[keep[-1]])))
        span = np.ascontiguousarray(v[:, keep])              # (2*nbas, 2m), T-closed

        # Pairing inside the closed span, in its own 2m coordinates: for v = span @ z the
        # time reverse is span @ (m2 @ conj(z)) with m2 unitary and antisymmetric.
        m2 = span.conj().T @ time_reverse(span)
        basis = np.eye(idx.size, dtype=np.complex128)
        coords = []
        while basis.shape[1]:
            u = basis[:, 0] / np.linalg.norm(basis[:, 0])
            tu = m2 @ np.conj(u)
            tu = tu - u * np.vdot(u, tu)                     # numerical guard; exactly 0
            tu = tu / np.linalg.norm(tu)
            coords.extend([u, tu])
            rest = basis - np.outer(u, np.conj(u) @ basis) \
                - np.outer(tu, np.conj(tu) @ basis)
            uu, ss, _ = np.linalg.svd(rest, full_matrices=False)
            basis = np.ascontiguousarray(uu[:, ss > 0.5])
        pairs = span @ np.stack(coords, axis=1)
        out[:, idx] = pairs
        finished.append(pairs)
    return out


def time_reversal_index_signs(n: int) -> "tuple":
    """``(swap, t)`` for the interleaved Kramers ordering: ``T a+_k T^-1 = t_k a+_{swap[k]}``.

    ``swap[k] = k ^ 1`` exchanges the partners of a pair and ``t_k = (-1)^k`` is ``+1`` on an
    unbarred spinor and ``-1`` on a barred one -- the sign that makes ``T^2 = -1``. Together
    they are how time reversal acts on a *spinor index*, as opposed to on the scalar-basis
    rows (:func:`time_reverse`), and they are the whole content of the convention for anything
    matrix-valued: an operator or a rotation generator over Kramers-paired spinors commutes
    with time reversal exactly when ``A_pq = t_p t_q conj(A[swap[p], swap[q]])``
    (:func:`time_reversal_even_part`).

    ⚠ Both statements are about a **Kramers-paired** orbital set. On an unrestricted one
    (:attr:`SpinorBasis.kramers_paired` false) the index ``k ^ 1`` is not the time-reversed
    partner of ``k`` at all, and every relation built on it is meaningless rather than merely
    inaccurate -- measure :func:`kramers_pairing_defect` before relying on one.
    """
    if int(n) % 2:
        raise ValueError("a Kramers-paired spinor set has an even number of spinors, "
                         "got {}".format(n))
    idx = np.arange(int(n))
    return idx ^ 1, np.where(idx % 2 == 0, 1.0, -1.0)


def time_reversal_even_part(a: np.ndarray) -> np.ndarray:
    """The time-reversal-**even** part of a matrix over Kramers-paired spinors.

    ``(A + Theta A) / 2`` with ``(Theta A)_pq = t_p t_q conj(A[pbar, qbar])``
    (:func:`time_reversal_index_signs`). ``Theta`` is an **antilinear** involution -- so the
    even part is a real-linear projection, not a complex-linear one -- and it preserves both
    hermiticity and anti-hermiticity, since transposition and the index swap commute.

    Two uses, and they are the same statement about different objects. On an *operator* it is
    the projection whose residual :func:`kuiva.ci.sigma.time_reversal_violation` measures. On
    an anti-Hermitian **rotation generator** it is the constraint that makes the rotation keep
    a Kramers-paired orbital set paired: ``exp`` is a real power series and ``Theta`` is
    multiplicative, so ``Theta(kappa) = kappa`` implies ``Theta(exp kappa) = exp(kappa)``,
    which written out is ``Omega conj(U) = U Omega`` -- exactly "the time reverse of the new
    unbarred orbital is the new barred one". That is what the orbital optimizer imposes
    (:func:`kuiva.mcscf.orbopt.optimize_orbitals`).

    ⚠ **Exact in floating point.** ``t_p t_q`` is ``+-1`` and ``(x + y) / 2`` is symmetric, so
    the result satisfies the relation to the last bit rather than to a tolerance; a projection
    that only nearly projected would leave exactly the drift it exists to remove.
    """
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("expected a square spinor-basis matrix, got shape {}"
                         .format(a.shape))
    swap, t = time_reversal_index_signs(a.shape[0])
    return 0.5 * (a + np.outer(t, t) * np.conj(a[np.ix_(swap, swap)]))


def time_reversal_odd_norm(a: np.ndarray) -> float:
    """``max |A - Theta A| / 2``, relative to ``max |A|``: how time-reversal-odd a matrix is.

    Zero exactly when :func:`time_reversal_even_part` returns the input unchanged. Relative,
    so it is comparable between a Hamiltonian of order 1e4 Eh and a rotation generator of
    order 1e-2.
    """
    a = np.asarray(a)
    if a.size == 0:
        return 0.0
    odd = float(np.max(np.abs(a - time_reversal_even_part(a))))
    return odd / (float(np.max(np.abs(a))) or 1.0)


def kramers_pairing_defect(c: np.ndarray) -> float:
    """``max_p |T c_2p - c_2p+1|`` over ``max |c|``: how far the columns are from exact pairs.

    The *column* statement, which is stronger than the *span* one
    (:func:`time_reversal_closure_defect`) and is the one anything indexed by ``2p`` /
    ``2p+1`` actually depends on: a set can span time-reversal-closed spaces while no column
    is any particular column's partner, which is precisely what a converged
    Kramers-unrestricted CASSCF returns.

    Relative and basis-agnostic: ``c`` may be over the AO basis or the working basis, both
    being real (``T`` conjugates coefficients only), and the scale divides out.
    """
    c = np.ascontiguousarray(c, dtype=np.complex128)
    if c.shape[1] % 2:
        raise ValueError("a Kramers-paired spinor set has an even number of columns, "
                         "got {}".format(c.shape[1]))
    if c.size == 0:
        return 0.0
    defect = float(np.max(np.abs(time_reverse(c[:, 0::2]) - c[:, 1::2])))
    return defect / (float(np.max(np.abs(c))) or 1.0)


def time_reversal_closure_defect(c: np.ndarray, blocks) -> float:
    """How far the worst block's **span** is from closed under time reversal.

    ``max_b || T b - b (b^dag T b) ||`` over the blocks: the part of the time-reversed block
    that does not lie back inside the block, which is zero exactly when the span is closed
    and is basis-independent within a block (an internal rotation moves neither term).

    ⚠ **Measuring this is what lets a repair be conditional, and conditional is what makes
    it safe to switch on by default.** An unconditional repair rewrites the orbitals at every
    step, and even a 1e-13 rewrite per step accumulates into a visibly different optimizer
    trajectory — measured: an example's CASSCF moved between 6 and 12 macro-iterations
    depending only on whether an inert-looking repair ran. Gated on this, a healthy step is
    returned untouched and the trajectory is the one the optimizer would have taken anyway.
    """
    c = np.ascontiguousarray(c, dtype=np.complex128)
    worst = 0.0
    for idx in blocks:
        idx = np.asarray(idx, dtype=int).ravel()
        if idx.size == 0:
            continue
        b = c[:, idx]
        tb = time_reverse(b)
        worst = max(worst, float(np.max(np.abs(tb - b @ (b.conj().T @ tb)))))
    return worst


def time_reversal_closed_span(c: np.ndarray, blocks) -> np.ndarray:
    """Move each block's **span** onto the nearest time-reversal-closed one, minimally.

    The same first step as :func:`nearest_kramers_paired` — per block, the dominant
    eigenspace of the symmetrized projector ``(P + T P T^-1) / 2``, which commutes with
    ``T`` by construction — but **without rebuilding the columns as ``(u, T u)`` pairs**.
    What comes back is the orthonormal set *closest to the input* that spans the closed
    subspace (the polar factor of the projected block), so a block that is already closed
    is returned unchanged to roundoff.

    ⚠ **The difference from :func:`nearest_kramers_paired` is the whole point, and it is not
    stylistic.** Re-pairing the columns is a rotation *within* each block: it changes no
    energy, no density and no CI spectrum, and it is exactly what a consumer of the pairing
    convention wants. But it is O(1) even on a set that is already perfectly closed —
    measured at 1.0 on a UF3 spinor set whose time-reversal breach was 7e-21 — and an
    orbital optimizer cannot survive that. Its curvature memory and its pending rotation are
    expressed in the current orbital frame, so an O(1) redundant rotation injected between
    steps makes both meaningless and the trust region collapses. An optimizer needs the
    span corrected and the frame left alone; that is this function.

    ⚠ **"Nearest" is not "almost the same", and that is a trap this function cannot check
    for you.** The correction it applies is the size of the defect only while the defect is
    small. Once a block has genuinely drifted, the nearest closed span is a *materially
    different subspace*, and applying this to an active space then **re-selects it** — a
    different calculation wearing the same name. Measured: driving an N2 CAS(6,8) CASSCF
    with this at every step converged 0.6 Eh **above its own SCF energy**, which a CAS
    containing the reference determinant cannot do. Use it to re-symmetrise something that
    is already nearly closed; do not use it to rescue something that is not.

    Parameters and failure mode are :func:`nearest_kramers_paired`'s: ``c`` is
    ``(2*nbas, n)`` over an **orthonormal real** scalar basis, ``blocks`` partition the
    columns into even-sized orbital spaces (so no Kramers pair straddles a space boundary),
    and a block too far from closed to repair unambiguously raises rather than being
    silently rounded onto some nearby span.
    """
    c = np.ascontiguousarray(c, dtype=np.complex128)
    out = np.array(c, copy=True)
    for idx in blocks:
        idx = np.asarray(idx, dtype=int).ravel()
        if idx.size == 0:
            continue
        if idx.size % 2:
            raise ValueError("a Kramers-paired block has an even number of columns; block "
                             "{} has {}".format(idx.tolist(), idx.size))
        b = np.ascontiguousarray(out[:, idx])
        tb = time_reverse(b)
        d = 0.5 * (b @ b.conj().T + tb @ tb.conj().T)
        w, v = np.linalg.eigh(d)
        keep = np.argsort(w)[::-1][:idx.size]
        if w[keep[-1]] <= 0.5 + 1e-6:
            raise ValueError(
                "the block on columns {}.. is too far from time-reversal closed to repair: "
                "the symmetrized projector's smallest kept eigenvalue is {:.3f} (a closed "
                "span has 1.0, and at 0.5 the nearest closed subspace is ambiguous)"
                .format(int(idx[0]), float(w[keep[-1]])))
        span = np.ascontiguousarray(v[:, keep])              # (2*nbas, 2m), T-closed
        # The orthonormal basis of `span` closest to the incoming columns: the polar factor
        # of the projection, U V^dag from its SVD. Identity when the block already lies in
        # the closed span, which is what keeps this a no-op on a healthy step.
        u, _s, vh = np.linalg.svd(span @ (span.conj().T @ b), full_matrices=False)
        out[:, idx] = u @ vh
    return out


def fold_to_kramers_pairs(a: np.ndarray, columns=None) -> "tuple":
    """Fold a spinor-basis operator matrix onto its Kramers-pair space.

    A **spin-free, time-even** operator ``O`` obeys ``<T p|O|T q> = <p|O|q>*``, so in the
    interleaved layout its unbarred-unbarred and barred-barred blocks are complex conjugates
    of each other and its unbarred-barred block vanishes. Everything such an operator says
    about a Kramers-paired orbital set is therefore carried by one ``(npair, npair)``
    Hermitian matrix::

        A_pair[m, n] = (O[2m, 2n] + conj(O[2m+1, 2n+1])) / 2

    Returns ``(a_pair, residual)`` with ``residual`` the largest of: any element of the two
    off-diagonal (unbarred-barred) blocks, and any departure from the conjugation relation
    between the two diagonal ones — the measure of how far the input is from being what this
    fold assumes. ⚠ **Both** off-diagonal blocks are looked at, not just one: they are each
    other's conjugate transpose only for a Hermitian operator, and this takes any square
    matrix. ⚠ It is returned
    rather than checked here: whether a given residual matters is the caller's decision, and
    every caller of this must make it (an operator folded despite a large residual is
    Hermitian, plausible and wrong).

    ``columns`` restricts the fold to a subset of Kramers pairs, given as **spinor** indices
    (whole pairs, ``2m`` and ``2m+1`` together); the pair ordering of the result follows the
    ascending pair index.
    """
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("expected a square spinor-basis matrix, got {}".format(a.shape))
    if columns is None:
        idx = np.arange(a.shape[0])
    else:
        idx = np.asarray(columns, dtype=int).ravel()
    if idx.size % 2:
        raise ValueError("a Kramers-paired selection has an even number of spinors, got {}"
                         .format(idx.size))
    unbarred, barred = idx[0::2], idx[1::2]
    if np.any(unbarred % 2 != 0) or np.any(barred != unbarred + 1):
        raise ValueError("the columns must be whole Kramers pairs (2m, 2m+1) in order; got "
                         "{}".format(idx.tolist()))
    uu = a[np.ix_(unbarred, unbarred)]
    bb = a[np.ix_(barred, barred)]
    off = (a[np.ix_(unbarred, barred)], a[np.ix_(barred, unbarred)])
    residual = max([float(np.max(np.abs(x))) for x in off if x.size]
                   + ([float(np.max(np.abs(uu - np.conj(bb))))] if uu.size else [0.0]))
    return 0.5 * (uu + np.conj(bb)), residual


def rotate_kramers_pairs(c: np.ndarray, v: np.ndarray, columns) -> np.ndarray:
    """Rotate whole Kramers pairs of ``c`` by the pair-space unitary ``v``, preserving pairing.

    ⚠ **The barred partners transform with the conjugate of ``v``, not with ``v``.** The
    pairing convention is that column ``2m+1`` is ``T`` applied to column ``2m``, and ``T``
    is antiunitary: ``T (sum_m v_mn |m>) = sum_m conj(v_mn) T|m>``. Rotating both sublattices
    with ``v`` gives an orthonormal set of the right shape whose barred columns are no longer
    the time reverses of the unbarred ones — which every consumer of the convention assumes
    and none of them checks, so it would survive to produce a plausible wrong answer.

    Parameters
    ----------
    c : ``(2*nbas, nspinor)`` complex — spinor coefficients in the interleaved layout.
    v : ``(npair, npair)`` complex unitary over the selected pairs, in their given order.
    columns : spinor indices of the pairs to rotate (whole pairs, ascending).

    Returns a copy of ``c`` with those columns replaced.
    """
    c = np.ascontiguousarray(c, dtype=np.complex128)
    idx = np.asarray(columns, dtype=int).ravel()
    if idx.size % 2:
        raise ValueError("a Kramers-paired selection has an even number of spinors, got {}"
                         .format(idx.size))
    unbarred, barred = idx[0::2], idx[1::2]
    if np.any(unbarred % 2 != 0) or np.any(barred != unbarred + 1):
        raise ValueError("the columns must be whole Kramers pairs (2m, 2m+1) in order; got "
                         "{}".format(idx.tolist()))
    v = np.ascontiguousarray(v, dtype=np.complex128)
    if v.shape != (unbarred.size, unbarred.size):
        raise ValueError("the pair rotation must be ({0}, {0}) for {0} Kramers pairs; got {1}"
                         .format(unbarred.size, v.shape))
    out_c = np.array(c, copy=True)
    out_c[:, unbarred] = c[:, unbarred] @ v
    out_c[:, barred] = c[:, barred] @ np.conj(v)
    return out_c


def spin_block_diagonal(a: np.ndarray) -> np.ndarray:
    """Lift a spin-free ``(m, n)`` matrix to the two-component basis: ``1_2 (x) A``.

    Used for the overlap and for the *spin-free* part of the one-electron Hamiltonian. The
    spin-dependent (spin-orbit) part is not of this form — see :func:`two_component_operator`.

    ``A`` need not be square. A **rectangular** one is a spin-free map between two scalar
    bases rather than an operator — a basis contraction, for instance
    (:meth:`kuiva.interface.pyscf_bridge.MolecularFourComponent.contract`) — and it lifts by
    the identical formula, because contraction mixes basis functions and does nothing to spin.
    Keeping that here rather than at the call site is the convention-lives-here rule: the ``[alpha; beta]`` row
    layout is defined in this module and assumed nowhere else.
    """
    a = np.asarray(a)
    m, n = a.shape
    big = np.zeros((2 * m, 2 * n), dtype=np.result_type(a.dtype, np.complex128))
    big[:m, :n] = a
    big[m:, n:] = a
    return big


def two_component_operator(a_sf: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Assemble a two-component one-electron operator ``A_sf * 1_2 + sigma . W``.

    Parameters
    ----------
    a_sf : ndarray (nbas, nbas)
        The spin-free part (e.g. the scalar X2C one-electron Hamiltonian), real symmetric.
    w : ndarray (3, nbas, nbas), optional
        The spatial factors of the spin-dependent part, in the convention ``W_k = i * w_k``
        with ``w_k`` **real antisymmetric** — which is how spin-orbit integrals come out of an
        integral library over real Gaussians. Omitted (``None``) gives the spin-free operator,
        i.e. the SOC-free reference.

    In the row layout of this module (``[alpha; beta]``)::

        sigma . W = [[  W_z      , W_x - i W_y ],     with W_k = i w_k
                     [ W_x + i W_y,   -W_z     ]]
                  = [[ i w_z      , i w_x + w_y ]
                     [ i w_x - w_y, -i w_z      ]]

    .. warning::
       This function fixes the *assembly*, not the provenance of ``w``. The sign and
       normalization of spin-orbit integrals differ between integral libraries and between
       X2C formulations (whether the picture-change transformation has been applied, and
       whether the factor of 1/2 from ``sigma = 2 s`` is inside or outside). When the X2C
       spin-orbit ingestion is added to the front-end, its convention **must** be
       validated against a known atomic splitting before anything is believed — a sign error
       here inverts multiplets and is invisible in every norm-based check.
       :func:`is_time_reversal_even` catches structural errors but not a global sign.
    """
    a_sf = np.asarray(a_sf)
    h = spin_block_diagonal(a_sf)
    if w is None:
        return h
    w = np.asarray(w)
    if w.shape[0] != 3 or w.shape[1:] != a_sf.shape:
        raise ValueError("spin-orbit factors must have shape (3, nbas, nbas) matching the "
                         "spin-free part {}, got {}".format(a_sf.shape, w.shape))
    # ⚠ **Relative, not absolute**, for the same reason :func:`time_reversal_residual` is: the
    # operators this function assembles span from an AMF correction of order 1e-3 Eh to the
    # raw ``W = <sigma.p V sigma.p>`` integrals of a heavy element at order 1e+6 Eh. An
    # absolute threshold is wrong at both ends — it fired on every heavy-element ``W``, whose
    # antisymmetry is exact to 1.5e-16 relative but 9.3e-10 absolute, while a genuinely broken
    # 1e-8-relative asymmetry on a small correction would have passed it silently.
    asym = max(float(np.max(np.abs(wk + wk.T))) for wk in w) if w.size else 0.0
    scale = float(np.max(np.abs(w))) if w.size else 0.0
    if asym > 1e-12 * (scale or 1.0):
        log.warning("spin-orbit factors w_k are not antisymmetric (max |w + w^T| = %.2e, "
                    "%.1e relative); the assembled operator will not be Hermitian",
                    asym, asym / (scale or 1.0))
    return h + sigma_dot(1j * w)


def sigma_dot(w: np.ndarray) -> np.ndarray:
    """``sigma . W`` for a **general** ``(3, n, n)`` complex ``W``, in the row layout of this
    module (``[alpha; beta]``)::

        sigma . W = [[  W_z       , W_x - i W_y ]
                     [ W_x + i W_y,   -W_z      ]]

    :func:`two_component_operator` is the common case of this — its ``W_k = i w_k`` with
    ``w_k`` real antisymmetric, which is how spin-orbit integrals come out of an integral
    library over real Gaussians, and which makes the result Hermitian and time-reversal even.
    This function assumes **neither**, because not every spin-dependent operator is of that
    form: the small-component block of the anomalous magnetic moment,
    ``<(sigma.p) sigma_k (sigma.p)>``, expands over a ``W`` that is real rather than
    imaginary and non-antisymmetric, and assembling it through the antisymmetric route would
    silently drop it.

    ⚠ **This is the single definition of the ``sigma . W`` layout** and
    :func:`two_component_operator` is a caller of it, not a second copy — the conventions of
    this module are defined here and nowhere else. A second spelling of these four blocks
    passes every norm and hermiticity test and is still a different convention.
    """
    w = np.asarray(w)
    if w.ndim != 3 or w.shape[0] != 3 or w.shape[1] != w.shape[2]:
        raise ValueError("sigma . W needs W as (3, n, n), got {}".format(w.shape))
    n = w.shape[1]
    wx, wy, wz = w
    h = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    h[:n, :n] = wz
    h[n:, n:] = -wz
    h[:n, n:] = wx - 1j * wy
    h[n:, :n] = wx + 1j * wy
    return h


def spin_operator(s_scalar: np.ndarray) -> np.ndarray:
    """The three spin matrices ``S_k = (1/2) sigma_k (x) S`` in the spin-blocked row layout.

    Parameters
    ----------
    s_scalar : ndarray (nbas, nbas)
        The **overlap** of the underlying scalar basis. It is the metric, not an operator:
        spin acts only on the spin index, so the spatial factor of ``S_k`` is whatever makes
        ``<mu sigma| S_k |nu tau>`` an integral over the scalar functions — the overlap.
        In an orthonormal basis pass the identity.

    Returns
    -------
    ndarray (3, 2*nbas, 2*nbas), complex128 — Hermitian, in units of hbar::

        S_x = 1/2 [[0, S], [S, 0]]    S_y = 1/2 [[0, -iS], [iS, 0]]
        S_z = 1/2 [[S, 0], [0, -S]]

    Notes
    -----
    ⚠ **This is deliberately not routed through** :func:`two_component_operator`, although the
    structures look alike. That function takes ``W_k = i w_k`` with ``w_k`` real
    **antisymmetric** — the form a spin-orbit integral over real Gaussians comes in — and the
    spatial factor here is a real **symmetric** overlap. Forcing it through would mean handing
    it an imaginary ``w``, tripping its antisymmetry warning on every call and burying a real
    diagnostic under a false one. The convention requires only that the ``[alpha; beta]`` layout be
    defined in this module, which is why the assembly lives here rather than at the call site.

    ⚠ **Spin is not the magnetic moment.** ``mu = -(L + g_e S) mu_B`` (see
    :func:`kuiva.props.multiplet.magnetic_moment_matrices`), and a state-averaged Kramers pair
    carries exactly zero spin density while its magnetic-moment *matrix* does not
    vanish — the off-diagonal elements are the whole content of a g tensor.
    """
    s = np.asarray(s_scalar)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("the scalar overlap must be square, got shape {}".format(s.shape))
    n = s.shape[0]
    ops = np.zeros((3, 2 * n, 2 * n), dtype=np.complex128)
    ops[0, :n, n:] = 0.5 * s                     # S_x
    ops[0, n:, :n] = 0.5 * s
    ops[1, :n, n:] = -0.5j * s                   # S_y
    ops[1, n:, :n] = 0.5j * s
    ops[2, :n, :n] = 0.5 * s                     # S_z
    ops[2, n:, n:] = -0.5 * s
    return ops


def decompose_two_component(h: np.ndarray) -> "tuple":
    """Inverse of :func:`two_component_operator`: ``H -> (A, w)`` in the fixed spinor conventions (kuiva/spinor/expand.py).

    Given a two-component operator in the spin-blocked ``[alpha; beta]`` row layout, return
    the spin-free part ``A`` (real symmetric) and the spin-orbit factors ``w`` (real
    antisymmetric, ``W_k = i w_k``)::

        A   = Re (H_aa + H_bb) / 2
        w_x = Im (H_ab + H_ba) / 2
        w_y = Re (H_ab - H_ba) / 2
        w_z = Im (H_aa - H_bb) / 2

    **This is a projection, not just a change of variables**, and that is the point. Taking
    ``A`` real symmetric and ``w_k`` real antisymmetric *is* the projection onto the
    time-reversal-even part of ``H``; everything else is discarded. For an operator that is
    exactly time-reversal even the discarded part is zero, and the round trip
    ``two_component_operator(*decompose_two_component(H)) == H`` holds to machine precision.
    For one that is not — the X2C decoupling involves a matrix square root whose rounding
    error is time-odd — the residual is the size of the symmetry breaking, and callers
    are expected to measure and report it rather than let it through: a time-odd part shows up
    downstream as a spurious Kramers splitting sitting right in the 1e-8..1e-6 Eh band
    reserves for genuine numerical splitting.

    This function is the single definition of the decomposition. It is used both by the
    front-end (``pyscf_bridge.ingest_spin_orbit``, on the one-electron X2C Hamiltonian) and by
    the atomic mean-field correction (:mod:`kuiva.amf.decouple`, on the two-electron picture
    change); the two must not drift apart, or the correction would be decomposed in a
    different convention from the operator it corrects.

    Returns
    -------
    (a_sf, w) : ndarray (nbas, nbas) real, ndarray (3, nbas, nbas) real
    """
    h = np.asarray(h)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] % 2:
        raise ValueError("a two-component operator must be square with an even dimension "
                         "(alpha block then beta block), got shape {}".format(h.shape))
    n = h.shape[0] // 2
    haa, hab, hba, hbb = h[:n, :n], h[:n, n:], h[n:, :n], h[n:, n:]
    a_sf = np.real(0.5 * (haa + hbb))
    a_sf = 0.5 * (a_sf + a_sf.T)
    w = np.stack([np.imag(0.5 * (hab + hba)),       # w_x
                  np.real(0.5 * (hab - hba)),       # w_y
                  np.imag(0.5 * (haa - hbb))])      # w_z
    w = 0.5 * (w - np.transpose(w, (0, 2, 1)))      # enforce exact antisymmetry
    return np.ascontiguousarray(a_sf), np.ascontiguousarray(w)


def time_reversal_residual(h: np.ndarray) -> "tuple":
    """``(absolute, relative)`` size of the part of ``h`` that :func:`decompose_two_component`
    discards, i.e. of its time-reversal-**odd** component.

    The relative figure is against ``max |h|``, so it is comparable between a one-electron
    Hamiltonian of order 1e4 Eh and a correction of order 1e-3 Eh.
    """
    h = np.asarray(h)
    residual = float(np.max(np.abs(two_component_operator(*decompose_two_component(h)) - h)))
    scale = float(np.max(np.abs(h))) or 1.0
    return residual, residual / scale


def is_time_reversal_even(h: np.ndarray, tol: float = 1e-10) -> bool:
    """True if ``T H T^-1 = H``, i.e. ``H`` commutes with time reversal.

    In the blocked row layout this reads ``H_aa = conj(H_bb)`` and ``H_ab = -conj(H_ba)``.
    Every term of a spin-orbit Hamiltonian for a spinless external field is time-reversal
    even (spin and orbital angular momentum are both time-odd, so ``sigma . W`` is even), so
    this is a genuine structural check on an assembled two-component operator: it catches a
    swapped or transposed spin block, which no norm or hermiticity test sees.
    """
    h = np.asarray(h)
    n = h.shape[0] // 2
    haa, hab, hba, hbb = h[:n, :n], h[:n, n:], h[n:, :n], h[n:, n:]
    return bool(np.max(np.abs(haa - np.conj(hbb))) < tol and
                np.max(np.abs(hab + np.conj(hba))) < tol)


# --- The spinor basis ----------------------------------------------------------------------

@dataclass(frozen=True)
class SpinorBasis:
    """A two-component spinor orbital set in the interleaved Kramers ordering.

    Attributes
    ----------
    c : ndarray (2*nbas, nspinor), complex128
        Coefficients over the underlying scalar basis; rows blocked ``[alpha; beta]``.
    energy : ndarray (nspinor,)
        Orbital energies, each scalar energy repeated over its Kramers pair. Meaningful for
        the guess only; after CASSCF the spinors are not eigenfunctions of anything diagonal.
    occ : ndarray (nspinor,)
        Guess occupations, ``mo_occ / 2`` on both partners (so a closed shell gives 1.0 and a
        singly occupied scalar orbital gives the Kramers-averaged 0.5). Bookkeeping only —
        the CI determines the actual occupation.
    basis : str
        Which scalar basis ``c`` is expressed in: ``"ao"`` or ``"working"``.
    kramers_paired : bool
        Whether ``c`` is known to be exactly Kramers paired (true for the guess; a
        general-complex CASSCF breaks it and the flag becomes False).
    """

    c: np.ndarray
    energy: np.ndarray
    occ: np.ndarray
    basis: str = "working"
    kramers_paired: bool = True
    #: ``|<phi^a_p|phi^b_p>|`` per spatial index, for an unrestricted expansion only. The
    #: phase-invariant measure of how well the two spin sets correspond; ``None`` otherwise.
    pair_overlap: Optional[np.ndarray] = None

    @property
    def nbas(self) -> int:
        """Size of the underlying *scalar* basis."""
        return int(self.c.shape[0] // 2)

    @property
    def nspinor(self) -> int:
        return int(self.c.shape[1])

    @property
    def alpha(self) -> np.ndarray:
        """The alpha spin block ``(nbas, nspinor)`` — a contiguous view, not a copy."""
        return self.c[:self.nbas]

    @property
    def beta(self) -> np.ndarray:
        """The beta spin block ``(nbas, nspinor)`` — a contiguous view, not a copy."""
        return self.c[self.nbas:]

    def transform_scalar_basis(self, x: np.ndarray, basis: str = "ao") -> "SpinorBasis":
        """Re-express the spinors over a different scalar basis: ``C -> (1_2 (x) X) C``.

        The normal use is going back to the AO basis for the integral transformation, with
        ``X`` the working-basis transformation (``OrthonormalBasis.x``): the three-index
        factors live in the AO basis, and folding ``X`` into the (small) coefficient matrix
        once is far cheaper than transforming the (large) factors. The spin blocking makes
        this two GEMMs on contiguous blocks.
        """
        x = np.asarray(x)
        if x.shape[1] != self.nbas:
            raise ValueError("basis transformation of shape {} does not match a scalar basis "
                             "of {} functions".format(x.shape, self.nbas))
        c = np.empty((2 * x.shape[0], self.nspinor), dtype=np.complex128)
        c[:x.shape[0]] = x @ self.alpha
        c[x.shape[0]:] = x @ self.beta
        return SpinorBasis(c=c, energy=self.energy, occ=self.occ, basis=basis,
                           kramers_paired=self.kramers_paired,
                           pair_overlap=self.pair_overlap)

    def take(self, columns) -> np.ndarray:
        """Coefficient block for a set of spinor indices (e.g. an active space)."""
        return np.ascontiguousarray(self.c[:, np.asarray(columns, dtype=int)])

    def partner_deviation(self) -> float:
        """Max deviation of each barred column from the time reverse of its unbarred partner.

        Zero for a Kramers-paired guess. Under a general-complex (Kramers-unrestricted)
        optimization it grows to the 1e-8..1e-6 range of the general-complex path — this is the number that
        measures it.

        ⚠ For an **unrestricted** expansion this is *not* a useful diagnostic: it is dominated
        by the arbitrary sign of each orbital (an alpha/beta pair differing only by a sign
        gives a deviation of 2). Use :attr:`pair_overlap`, which is phase invariant.
        """
        tc = time_reverse(self.c[:, ::2])
        return float(np.max(np.abs(self.c[:, 1::2] - tc))) if self.nspinor else 0.0

    def orthonormality_error(self, s_scalar: Optional[np.ndarray] = None) -> float:
        """``max |C^dag S_2c C - 1|``. With ``s_scalar=None`` the scalar basis is orthonormal
        (the working basis), which is the normal case."""
        if s_scalar is None:
            g = self.c.conj().T @ self.c
        else:
            g = self.alpha.conj().T @ s_scalar @ self.alpha + \
                self.beta.conj().T @ s_scalar @ self.beta
        return float(np.max(np.abs(g - np.eye(self.nspinor))))

    def report(self, logger=None, *, s_scalar: Optional[np.ndarray] = None) -> None:
        """Log the standard spinor-basis block (the standard output grammar)."""
        logger = logger or log
        out.entry(logger, "scalar basis functions", self.nbas,
                  note="{} basis".format(self.basis))
        out.entry(logger, "spinors", self.nspinor)
        out.entry(logger, "Kramers pairs", self.nspinor // 2)
        out.entry(logger, "ordering", "interleaved (p, pbar); rows [alpha|beta]")
        if self.kramers_paired:
            out.entry(logger, "Kramers pairing deviation", self.partner_deviation(),
                      fmt="{:.2e}")
        else:
            out.entry(logger, "Kramers paired", False,
                      note="unrestricted: barred partner is the beta orbital")
            if self.pair_overlap is not None:
                occupied = self.pair_overlap[self.occ[0::2] + self.occ[1::2] > 1e-8]
                out.entry(logger, "min |<phi_a|phi_b>| over occupied",
                          float(occupied.min()) if occupied.size else 1.0, fmt="{:.4f}",
                          note="0 is normal for a degenerate shell")
        out.entry(logger, "orthonormality error", self.orthonormality_error(s_scalar),
                  fmt="{:.2e}")

    def __repr__(self) -> str:
        return "SpinorBasis(nbas={}, nspinor={}, basis={}, kramers_paired={})".format(
            self.nbas, self.nspinor, self.basis, self.kramers_paired)


def expand_scalar_mos(mo_coeff: np.ndarray, mo_energy: Optional[np.ndarray] = None,
                      mo_occ: Optional[np.ndarray] = None, *, basis: str = "working",
                      report: bool = False) -> SpinorBasis:
    """Expand real scalar MOs into a Kramers-paired two-component spinor set.

    Parameters
    ----------
    mo_coeff : ndarray (nbas, nmo)
        Real scalar-relativistic MO coefficients, in the AO basis or (preferably) in the
        orthonormal working basis.
    mo_energy, mo_occ : ndarray (nmo,), optional
        Copied onto both Kramers partners; ``occ`` is halved (see :class:`SpinorBasis`).
    basis : str
        Label recording which scalar basis the coefficients are in.

    Returns
    -------
    SpinorBasis with ``nspinor = 2 * nmo``.
    """
    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim != 2:
        raise ValueError("mo_coeff must be (nbas, nmo), got shape {}".format(mo_coeff.shape))
    if np.iscomplexobj(mo_coeff) and np.max(np.abs(mo_coeff.imag)) > 1e-12:
        # The scalar X2C reference is real by construction; a complex input means the
        # caller is expanding something that is not a scalar reference.
        log.warning("scalar MOs have a non-negligible imaginary part (max %.2e); the Kramers "
                    "pairing below assumes real scalar orbitals",
                    float(np.max(np.abs(mo_coeff.imag))))
    nbas, nmo = mo_coeff.shape

    c = np.zeros((2 * nbas, 2 * nmo), dtype=np.complex128)
    c[:nbas, 0::2] = mo_coeff          # unbarred: (phi, 0)
    c[nbas:, 1::2] = mo_coeff          # barred:   (0, phi) = T (phi, 0) for real phi

    energy = (np.repeat(np.asarray(mo_energy, dtype=float), 2) if mo_energy is not None
              else np.zeros(2 * nmo))
    occ = (np.repeat(np.asarray(mo_occ, dtype=float), 2) * 0.5 if mo_occ is not None
           else np.zeros(2 * nmo))

    sb = SpinorBasis(c=c, energy=energy, occ=occ, basis=basis, kramers_paired=True)
    dev = sb.partner_deviation()
    if dev > 1e-12:                    # cannot happen by construction; guards a future edit
        log.error("spinor guess is not Kramers paired (deviation %.2e)", dev)
    log.debug("expanded %d scalar MOs into %d spinors (%d Kramers pairs)",
              nmo, 2 * nmo, nmo)
    if report:
        sb.report()
    return sb


def expand_unrestricted_mos(mo_alpha: np.ndarray, mo_beta: np.ndarray,
                            mo_energy=None, mo_occ=None, *, basis: str = "working",
                            s_scalar: Optional[np.ndarray] = None,
                            report: bool = False) -> SpinorBasis:
    """Expand an **unrestricted** scalar reference (UHF/UKS) into a spinor set.

    Spinor ``2p`` takes the alpha orbital ``phi^a_p`` in its alpha component, spinor ``2p+1``
    the beta orbital ``phi^b_p`` in its beta component::

        psi_{2p}   = (phi^a_p, 0)
        psi_{2p+1} = (0, phi^b_p)

    This set is **orthonormal** — the two spin components are orthogonal by spin, so the
    cross overlaps vanish identically whatever the alpha/beta orbitals do — but it is **not
    Kramers paired**: ``T psi_{2p} = (0, phi^a_p)``, which equals ``psi_{2p+1}`` only if the
    two spin sets coincide. That is exactly the point of an unrestricted reference, and
    :meth:`SpinorBasis.partner_deviation` then measures something physical: the spin
    polarization of the orbitals. ``kramers_paired`` is set ``False`` accordingly, so nothing
    downstream can assume a symmetry the guess does not have.

    ⚠ **The index pairing carries no meaning here, and the actionable consequence is that an
    active space may not be chosen as a contiguous spinor range.** For a restricted reference
    spinors ``2p`` and ``2p+1`` are the same spatial orbital, so a contiguous range is a
    well-defined set of spatial orbitals. For an unrestricted one they are the
    *p*-th alpha and the *p*-th beta orbital, which need not describe the same thing: within a
    degenerate shell the alpha and beta orbitals are arbitrarily oriented, so
    ``|<phi^a_p|phi^b_p>|`` is routinely **zero** for perfectly good orbitals (an O atom's
    singly occupied 2p is the standard example). That is not an error and is not warned about.
    Select the active space by orbital *character*, per spin set, and build the column
    list explicitly.

    The overlaps are recorded in :attr:`SpinorBasis.pair_overlap` for inspection. They are
    absolute values by necessity: the sign of each SCF orbital is arbitrary, and a sign flip
    alone makes a naive coefficient comparison look catastrophic.
    A corresponding-orbital (Amos-Hall) transformation is the standard fix if it ever matters;
    the alternative, and usually the better route to a CASSCF guess from an unrestricted
    reference, is to use the UHF natural orbitals — a single set, which then goes through
:func:`expand_scalar_mos` instead. That choice belongs to `mcscf/preopt.py`, not
    here.

    References
    ----------
    * Corresponding orbitals for comparing two non-orthogonal orbital sets: A. T. Amos,
      G. G. Hall, Proc. R. Soc. London A 263, 483 (1961), doi:10.1098/rspa.1961.0175.
    * UHF natural orbitals as an active-space guess: P. Pulay, T. P. Hamilton, J. Chem. Phys.
      88, 4926 (1988), doi:10.1063/1.454704.
    """
    mo_alpha = np.asarray(mo_alpha)
    mo_beta = np.asarray(mo_beta)
    if mo_alpha.shape != mo_beta.shape:
        raise ValueError("alpha and beta MO sets must have the same shape, got {} and {}"
                         .format(mo_alpha.shape, mo_beta.shape))
    nbas, nmo = mo_alpha.shape

    c = np.zeros((2 * nbas, 2 * nmo), dtype=np.complex128)
    c[:nbas, 0::2] = mo_alpha
    c[nbas:, 1::2] = mo_beta

    if mo_energy is not None:
        ea, eb = np.asarray(mo_energy, dtype=float)
        energy = np.empty(2 * nmo)
        energy[0::2], energy[1::2] = ea, eb
    else:
        energy = np.zeros(2 * nmo)
    if mo_occ is not None:
        oa, ob = np.asarray(mo_occ, dtype=float)
        occ = np.empty(2 * nmo)
        occ[0::2], occ[1::2] = oa, ob        # already per spin: no halving (cf. restricted)
    else:
        occ = np.zeros(2 * nmo)

    # Phase-invariant correspondence measure (see the warning above).
    gram = mo_alpha.T @ mo_beta if s_scalar is None else mo_alpha.T @ s_scalar @ mo_beta
    pair_overlap = np.abs(np.diag(gram))

    sb = SpinorBasis(c=c, energy=energy, occ=occ, basis=basis, kramers_paired=False,
                     pair_overlap=pair_overlap)
    log.debug("expanded %d unrestricted MO pairs into %d spinors; min |<a|b>| over occupied "
              "%.4f", nmo, 2 * nmo, float(pair_overlap.min()) if pair_overlap.size else 1.0)
    # The one thing the caller must act on (see the docstring): index ranges are not orbital
    # sets here. Warned once per expansion, not per orbital — a small pair overlap is normal.
    log.warning("unrestricted reference: the spinor set is not Kramers paired, and spinors "
                "2p / 2p+1 are the p-th alpha and p-th beta orbital rather than one spatial "
                "orbital. Choose the active space by orbital character per spin set, not as "
                "a contiguous spinor range.")
    if report:
        sb.report()
    return sb


__all__ = ["SpinorBasis", "expand_scalar_mos", "expand_unrestricted_mos",
           "time_reverse", "spin_block_diagonal",
           "two_component_operator", "sigma_dot", "spin_operator",
           "decompose_two_component", "time_reversal_residual",
           "is_time_reversal_even", "kramers_block_permutation",
           "spinor_indices", "spatial_index", "barred", "unbarred", "is_barred",
           "fold_to_kramers_pairs", "rotate_kramers_pairs", "nearest_kramers_paired",
           "time_reversal_closed_span", "time_reversal_closure_defect",
           "time_reversal_index_signs", "time_reversal_even_part",
           "time_reversal_odd_norm", "kramers_pairing_defect"]
