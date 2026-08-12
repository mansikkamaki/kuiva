"""Integral blocks for the perturbation classes, assembled on demand.

**Orchestration, not a registered kernel.** The arithmetic here is one call to
:func:`kuiva.integrals.transform.transform_3c` per requested block plus the ``tensordot`` the
classes do themselves; everything in this module is caching, sizing and batching policy.

What this exists to prevent
---------------------------
SC-NEVPT2 wants ``(ai|bj)``, ``(ai|bt)``, ``(ti|uj)``, ``(at|bu)`` and friends — four-index
objects over *external* label spaces that are the two large ones. Materializing any of them
over the full orbital range is the thing the factorization design forbids: ``n_virtual^2 n_inactive^2`` is
gigabytes on the first real system. So:

* the **three-index** MO factors ``B^P_{pq}`` are what is stored (``naux * n_bra * n_ket``, the
  same order as ``CASIntegrals.b_act``), one per requested space pair, cached because a class
  loop asks for the same pair once per state;
* the **four-index** blocks are assembled by the classes, per batch, from those factors, and
  released before the next batch. :func:`batch_slices` is the one place a batch size is
  decided, and it asks :func:`kuiva.util.resources.transient_gb` **once, outside the loop**
  (kernel rule B7).

⚠ A three-index block is a *resident* allocation and is reserved as one; a four-index batch is
a *transient* buffer and is deliberately not. That is the resident/transient two-category split, and the
reason this module never wraps ``np.empty``.

⚠ **The bra index is the conjugated one, and no class may add a second conjugation.**
``transform_3c`` builds ``B^P_{pq} = sum_{mu nu} C*_{mu p} L^P_{mu nu} C_{nu q}``, so
``assemble_4c(B_pq, B_rs)`` is a plain contraction over ``P`` with no ``conj`` — see that
function's warning. Every block name below is ordered ``(bra, ket)`` for this reason.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..integrals.transform import ThreeIndexAO, mo_block_memory_gb, transform_3c
from ..mcscf.orbopt import OrbitalSpaces
from ..util import resources as res
from ..util.logging import get_logger

log = get_logger(__name__)

#: The orbital spaces a block may be requested over. ``"all"`` is the full spinor range, which
#: some classes need on one side (the general index of a generalized Fock contraction).
SPACE_NAMES = ("inactive", "active", "virtual", "all")


class IntegralBlocks:
    """Three-index MO factors over pairs of orbital spaces, cached and budgeted.

    Parameters
    ----------
    factors : :class:`~kuiva.integrals.transform.ThreeIndexAO`
        AO factors, DF or Cholesky — the caller never learns which.
    c_spinor : ``(2*nao, n_orb)`` complex
        Spinor coefficients **in the AO basis**, at the orbitals the perturbation is built on.
        ⚠ For NEVPT2 these are the *pseudo-canonical* orbitals
        (:func:`kuiva.pt.nevpt2.pseudo_canonicalize`), not the raw converged CASSCF set.
    spaces : :class:`~kuiva.mcscf.orbopt.OrbitalSpaces`
    inactive_keep, virtual_keep : integer arrays, optional
        Positions **within** ``spaces.inactive`` / ``spaces.virtual`` that the perturbation is
        allowed to use as external labels — the frozen-core and deleted-virtual selection of
        the frozen-core option, resolved by :func:`kuiva.pt.nevpt2.select_correlated`.

        ⚠ **After this, ``"inactive"`` MEANS "correlated inactive" for every block and every
        class, and that is exactly the frozen-core semantics**: a frozen spinor keeps its mean field —
        it stays in ``F^I`` and in ``e_core``, which are built by ``CASIntegrals`` from the
        *whole* inactive space and are untouched here — and only disappears from the ``i, j``
        label ranges. Nothing else about the calculation changes, which is why the option is
        cheap and safe. A block that wanted the frozen orbitals would be asking for something
        no class needs.
    """

    def __init__(self, factors: ThreeIndexAO, c_spinor: np.ndarray,
                 spaces: OrbitalSpaces, *,
                 inactive_keep: Optional[np.ndarray] = None,
                 virtual_keep: Optional[np.ndarray] = None) -> None:
        c_spinor = np.ascontiguousarray(c_spinor)
        if c_spinor.shape[1] != spaces.n_orb:
            raise ValueError("coefficients carry {} spinors but the spaces partition {}"
                             .format(c_spinor.shape[1], spaces.n_orb))
        self.factors = factors
        self.coeff = c_spinor
        self.spaces = spaces
        self._keep = {
            "inactive": _resolve_keep("inactive", inactive_keep, spaces.n_inactive),
            "virtual": _resolve_keep("virtual", virtual_keep, spaces.n_virtual),
        }
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._allocs: Dict[Tuple[str, str], object] = {}

    # -- spaces ---------------------------------------------------------------------------
    @property
    def naux(self) -> int:
        return int(self.factors.naux)

    def indices(self, space: str) -> np.ndarray:
        """The spinor indices of ``space``, in the order the blocks are laid out in.

        ⚠ ``"inactive"`` and ``"virtual"`` are the **correlated** subsets when a frozen-core or
        deleted-virtual selection was given; ``"all"`` is always the full spinor range, because
        the only thing that asks for it is a generalized index of a Fock-like contraction.
        """
        if space == "all":
            return np.arange(self.spaces.n_orb)
        if space not in SPACE_NAMES:
            raise ValueError("unknown orbital space {!r}; expected one of {}"
                             .format(space, list(SPACE_NAMES)))
        full = getattr(self.spaces, space)
        keep = self._keep.get(space)
        return full if keep is None else np.ascontiguousarray(full[keep])

    def size(self, space: str) -> int:
        return int(self.indices(space).size)

    def orbitals(self, space: str) -> np.ndarray:
        """The coefficient block of ``space``, C-contiguous, ready for ``transform_3c``."""
        return np.ascontiguousarray(self.coeff[:, self.indices(space)])

    # -- blocks ---------------------------------------------------------------------------
    def block_gb(self, bra: str, ket: str) -> float:
        """Size [GB] of the three-index block ``(bra, ket)`` (exact sizing function)."""
        return mo_block_memory_gb(self.naux, self.size(bra), self.size(ket), np.complex128)

    def three_index(self, bra: str, ket: str) -> np.ndarray:
        """``B^P_{pq}`` with ``p`` in ``bra`` and ``q`` in ``ket``; cached.

        Returns ``(naux, n_bra, n_ket)`` ``complex128``. The cache is keyed by the *pair*, so
        ``("virtual", "inactive")`` and ``("inactive", "virtual")`` are two blocks — they are
        related by ``B^P_{qp} = conj(B^P_{pq})`` and a class that wants the other order should
        conjugate rather than ask for a second transform.
        """
        key = (str(bra), str(ket))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        gb = self.block_gb(*key)
        self._allocs[key] = res.reserve(
            "NEVPT2 three-index block B^P_({}|{})".format(*key), gb,
            note="naux={} x {} x {} spinors".format(self.naux, self.size(bra), self.size(ket)),
            advice=["freeze core spinors, which removes them from every class label range",
                    "reduce the Cholesky/auxiliary dimension",
                    "restrict the state average: the blocks are shared across states, but "
                    "the per-state batches are not"])
        block = transform_3c(self.factors, self.orbitals(bra), self.orbitals(ket))
        self._cache[key] = block
        log.debug("NEVPT2 block B^P_(%s|%s): %s, %.4f GB", key[0], key[1], block.shape, gb)
        return block

    def release(self) -> None:
        """Drop every cached block and its reservation."""
        for key, alloc in self._allocs.items():
            res.BUDGET.release(alloc)
        self._allocs.clear()
        self._cache.clear()

    def __enter__(self) -> "IntegralBlocks":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def __repr__(self) -> str:
        return "IntegralBlocks(naux={}, cached={})".format(
            self.naux, sorted("({}|{})".format(*k) for k in self._cache))


def _resolve_keep(space: str, keep, size: int) -> Optional[np.ndarray]:
    """Validate a correlated-subset selection: sorted, unique, in range, or ``None``."""
    if keep is None:
        return None
    idx = np.asarray(keep, dtype=int).ravel()
    if idx.size and (idx.min() < 0 or idx.max() >= size):
        raise ValueError("the {} selection indexes position {}..{} of a space of {}"
                         .format(space, int(idx.min()), int(idx.max()), size))
    if np.unique(idx).size != idx.size:
        raise ValueError("the {} selection repeats a position".format(space))
    if idx.size == size:
        return None
    return np.sort(idx)


def batch_slices(n_items: int, gb_per_item: float, *,
                 budget_gb: Optional[float] = None) -> List[slice]:
    """Slices over ``n_items`` whose per-batch working set fits the transient budget.

    ``gb_per_item`` is the caller's own exact estimate of what **one** leading index costs in
    temporaries — the classes compute it from their own array shapes, because only they know
    them. One call to :func:`kuiva.util.resources.transient_gb` happens here and nowhere
    inside a loop (B7).

    Always returns at least one slice, and never an empty one: a single item that does not fit
    the budget is still attempted, because refusing it here would report a shortfall against a
    *share* of the limit rather than against the limit, which is not the same number and not
    the caller's fault. The resident checks are what refuse.
    """
    n_items = int(n_items)
    if n_items <= 0:
        return []
    budget = res.transient_gb() if budget_gb is None else float(budget_gb)
    per = max(float(gb_per_item), 1e-12)
    step = int(max(1, min(n_items, budget / per)))
    return [slice(lo, min(lo + step, n_items)) for lo in range(0, n_items, step)]


def batch_gb(shape: Sequence[int], count: int = 1) -> float:
    """[GB] of ``count`` ``complex128`` arrays of ``shape`` — the classes' per-item estimate."""
    return count * res.array_gb(tuple(int(s) for s in shape), np.complex128)


__all__ = ["IntegralBlocks", "SPACE_NAMES", "batch_gb", "batch_slices"]
