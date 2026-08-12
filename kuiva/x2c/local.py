"""Local (atom-blocked) X2C decoupling: the DLU approximation.

Primary reference: D. Peng, M. Reiher, "Local relativistic exact decoupling",
J. Chem. Phys. **136**, 244108 (2012), doi:10.1063/1.4729788.

What DLU is
-----------
Exact X2C builds one decoupling matrix ``X`` and one renormalization ``R`` for the whole
molecule, which costs a dense four-component eigenproblem of dimension ``4 nao``. The
**diagonal local approximation to the unitary decoupling transformation (DLU)** replaces that
single global transformation by a *block-diagonal* one, ``U ~ (+) U_A``, whose blocks come
from local — normally atomic — problems. In the ``X``/``R`` parametrization the two-component
operator is then::

    A^DLU_AB = R_A^dag ( A_LL,AB + A_LS,AB X_B + X_A^dag A_SL,AB + X_A^dag A_SS,AB X_B ) R_B

⚠ **Every molecular block is still transformed, including the off-diagonal ones.** That is the
whole difference between DLU and the much cruder DLH ("diagonal local approximation to the
Hamiltonian"), which keeps only the diagonal blocks relativistic and leaves the off-diagonal
ones non-relativistic. DLH is not implemented here and is not wanted: it costs the same as DLU
and is substantially worse, and it breaks translational invariance, which DLU does not.

⚠ **DLU is also not PySCF's** ``approx="atom1e"``. That builds ``X`` block-diagonally from
isolated-atom problems but then computes ``R`` from the **full molecular** overlap, so the
transformation is not local and the ``O(nao^3)`` work remains. Both are available in Kuiva and
they are different approximations; see the README's Hamiltonian table.

Why this module only builds ``X`` and ``R``
-------------------------------------------
Once ``X`` and ``R`` are block-diagonal matrices, the expression above **is**
:func:`kuiva.x2c.decouple.picture_change` applied to them — block-diagonal ``X`` makes
``(A_LS X)_AB = A_LS,AB X_B`` and ``(X^dag A_SS X)_AB = X_A^dag A_SS,AB X_B`` automatically.
So there is no separate DLU transformation routine and no second code path that could drift
from the exact one: DLU differs from exact X2C *only* in how ``X`` and ``R`` are obtained.
That is a deliberate structural choice, and it is what makes the "single atom must reproduce
exact X2C bitwise" test meaningful rather than tautological.

⚠ **The first implementation forms dense block-diagonal matrices and uses dense GEMMs**
(correctness before speed). The saving that matters is already realized — the
``O((4 nao)^3)`` dense eigenproblem is replaced by one small eigenproblem per fragment, and
the ``(4 nao)^2`` complex workspace by per-fragment ones. Exploiting the sparsity of ``X`` and
``R`` in the contraction is a further, unmeasured win and belongs behind a profile.

Where the local problem comes from — a real convention, not a detail
--------------------------------------------------------------------
``U_A`` has to be defined by *some* four-component problem restricted to fragment ``A``, and
there are two defensible choices. Kuiva implements both and records which was used, because
they are not the same approximation:

``source="diagonal"`` (**the default**)
    The diagonal block of the *molecular* matrices, ``h_AA`` and ``S_AA``. The fragment's own
    nuclear attraction **and that of every other nucleus** is present, so the local
    decoupling sees the molecular environment. Needs no extra integrals and no basis
    bookkeeping — it is a slice — which also makes it impossible to get the AO ordering wrong.

``source="isolated"``
    A separate one-fragment four-component problem, with only that fragment's nuclei. The
    resulting ``U_A`` depends on nothing but ``(element, basis)``, so it is transferable and
    cacheable across an entire potential-energy surface, and it is the ``X`` PySCF's
    ``atom1e`` uses. ⚠ It also has a failure mode the diagonal source cannot have: the
    isolated fragment's AO ordering must match the molecular block's, and a permuted block
    would be Hermitian, of the right magnitude, and wrong. :func:`check_local_blocks` refuses
    it by comparing the *overlap*, which depends on the basis alone and must agree exactly.

Neither is "the" DLU; the paper's own numerical work uses local atomic problems, and which one
is better for the systems this program targets is an empirical question that the molecular
measurements will answer. Until they do, the default is the one that cannot be silently
misassembled.

What is exact, and therefore testable
--------------------------------------
* **A single fragment.** With one block covering the whole basis the "approximation" is the
  exact transformation, and both sources reduce to it identically. DLU is not an approximation
  at all for an atom or a monatomic ion.
* **Non-interacting fragments.** As the separation grows the off-diagonal molecular blocks
  vanish and the exact ``U`` becomes block-diagonal, so DLU becomes exact. The error is
  therefore a *bonding-region* error, which is what makes a potential-energy scan the
  informative measurement rather than a single geometry.
* **Structure.** ``X`` and ``R`` block-diagonal preserve Hermiticity and time-reversal
  evenness of whatever is transformed, because each block does.
* **The non-relativistic limit.** ``X -> 0`` and ``R -> 1`` block by block, so DLU reduces to
  the non-relativistic operator exactly, as the exact transformation does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..util import resources as res
from ..util.logging import get_logger
from .decouple import FourComponentBlocks, decoupling_matrices, decoupling_memory_gb

log = get_logger(__name__)

#: Sources for the local four-component problem that defines ``U_A``. See the module
#: docstring: they are different approximations, not implementation choices.
LOCAL_SOURCES = ("diagonal", "isolated")

#: Largest difference tolerated between an isolated fragment's overlap and the molecular
#: diagonal block of it (:func:`check_local_blocks`). Both are built by the same integral
#: engine from the same parsed basis, so the measured difference is **exactly** zero; the
#: tolerance exists so that a future normalization change fails with a diagnosis rather than
#: at the last digit. Matches :data:`kuiva.amf.correction.AO_ORDERING_TOLERANCE`, which guards
#: the same failure mode for the atomic mean field.
OVERLAP_MATCH_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Partition:
    """A grouping of a spin-blocked basis into the fragments DLU decouples separately.

    ``indices[i]`` holds the row/column indices of fragment ``i`` in a ``(2*nao, 2*nao)``
    block. ⚠ They are **not** contiguous: the conventions order rows spin-blocked
    ``[alpha; beta]``, so a fragment owns one range in each half. Build these with
    :meth:`kuiva.interface.pyscf_bridge.MolecularFourComponent.spin_blocked_indices` rather
    than by hand.

    ``labels[i]`` names the fragment (an atom label such as ``"Ti1"``). It is what a
    ``source="isolated"`` local-block mapping is keyed by, and what appears in diagnostics.
    """

    indices: Tuple[np.ndarray, ...]
    labels: Tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.labels):
            raise ValueError("a partition needs one label per fragment, got {} indices and "
                             "{} labels".format(len(self.indices), len(self.labels)))
        if not self.indices:
            raise ValueError("a partition must contain at least one fragment")

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def dimension(self) -> int:
        """Total number of indices covered, i.e. the ``2*nao`` this partition is of."""
        return int(sum(idx.size for idx in self.indices))

    def validate(self, n2c: int) -> None:
        """Refuse a partition that is not an exact cover of ``range(n2c)``.

        ⚠ An overlap or a gap is the one error here that no downstream check can see. A
        repeated index means a fragment's ``X`` block is silently overwritten by its
        neighbour's; a missing one leaves ``X`` and ``R`` zero there, which makes the
        transformation **singular on that direction** while every block that was written stays
        perfectly well formed. Both produce a Hermitian, plausible, wrong Hamiltonian.
        """
        seen = np.concatenate([np.asarray(idx, dtype=np.intp) for idx in self.indices])
        if seen.size != n2c:
            raise ValueError(
                "the partition covers {} indices but the basis has {}. A DLU partition must "
                "be an exact cover: every basis function belongs to exactly one fragment."
                .format(seen.size, n2c))
        if not np.array_equal(np.sort(seen), np.arange(n2c)):
            duplicated = sorted({int(i) for i in seen[np.bincount(seen, minlength=n2c)[seen] > 1]})
            missing = sorted(set(range(n2c)) - set(int(i) for i in seen))
            raise ValueError(
                "the partition is not an exact cover of the basis: {} index/indices appear "
                "more than once ({}...), {} are missing ({}...). An overlap silently "
                "overwrites one fragment's decoupling with another's; a gap leaves the "
                "transformation singular there.".format(
                    len(duplicated), duplicated[:5], len(missing), missing[:5]))

    @classmethod
    def single(cls, n2c: int, label: str = "molecule") -> "Partition":
        """The trivial one-fragment partition, for which DLU **is** exact X2C.

        Not a curiosity: it is what makes the exactness test a real test of this module rather
        than of a special case written into it.
        """
        return cls(indices=(np.arange(n2c),), labels=(label,))


def sub_blocks(blocks: FourComponentBlocks, indices: np.ndarray) -> FourComponentBlocks:
    """The four-component sub-operator on ``indices``, block by block."""
    idx = np.asarray(indices, dtype=np.intp)
    grid = np.ix_(idx, idx)
    return FourComponentBlocks(
        ll=np.ascontiguousarray(blocks.ll[grid]), ls=np.ascontiguousarray(blocks.ls[grid]),
        sl=np.ascontiguousarray(blocks.sl[grid]), ss=np.ascontiguousarray(blocks.ss[grid]))


def check_local_blocks(overlap: FourComponentBlocks, partition: Partition,
                       local: Dict[str, Tuple[FourComponentBlocks, FourComponentBlocks]],
                       tolerance: float = OVERLAP_MATCH_TOLERANCE) -> None:
    """Assert that supplied isolated-fragment blocks really are this molecule's fragments.

    ⚠ **Checked on the overlap, not on a shape.** Two bases can agree on how many functions a
    fragment has and disagree on their order, and that difference is invisible to every
    norm-based test while being fatal. The overlap depends on the basis alone — not on the
    nuclei, not on the geometry — so the isolated fragment's ``S`` and the molecular diagonal
    block of it must agree to the last digits. ``h`` must **not** be compared this way: it
    legitimately differs, since the molecular block carries every other nucleus's attraction.

    This is the same guard, for the same reason, as
    :func:`kuiva.amf.correction._check_atom_ordering`.
    """
    for label, idx in zip(partition.labels, partition.indices):
        if label not in local:
            raise KeyError(
                "no isolated-fragment blocks supplied for {!r}; the partition names {}. With "
                "source='isolated' every fragment needs its own local problem.".format(
                    label, ", ".join(sorted(set(partition.labels)))))
        _, local_overlap = local[label]
        expected = sub_blocks(overlap, idx)
        if local_overlap.ll.shape != expected.ll.shape:
            raise ValueError(
                "the isolated blocks for {} span {} spin-orbital functions where the molecule "
                "gives it {}.".format(label, local_overlap.ll.shape[0], expected.ll.shape[0]))
        deviation = float(np.max(np.abs(local_overlap.ll - expected.ll)))
        if deviation > tolerance:
            raise ValueError(
                "the isolated-fragment basis for {} is not the one the molecule uses for it: "
                "their overlap matrices differ by {:.2e} (tolerance {:.0e}). The two are built "
                "from the same parsed basis, so this is an AO **ordering** or normalization "
                "mismatch, not a numerical one — and a permuted block would stay Hermitian, "
                "keep the right magnitude, and be wrong.".format(label, deviation, tolerance))


def local_decoupling_matrices(
        hcore: FourComponentBlocks, overlap: FourComponentBlocks, light_speed: float,
        partition: Partition, *, source: str = "diagonal",
        local: Optional[Dict[str, Tuple[FourComponentBlocks, FourComponentBlocks]]] = None,
        report: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Block-diagonal ``X`` and ``R`` for the DLU approximation.

    Hand the result to :func:`kuiva.x2c.decouple.picture_change` exactly as the exact
    matrices would be — that is the whole method (see the module docstring).

    Parameters
    ----------
    hcore, overlap : FourComponentBlocks
        The **molecular** four-component one-electron problem.
    partition : Partition
        Which basis functions each local decoupling covers. Validated as an exact cover.
    source : {"diagonal", "isolated"}
        Which four-component problem defines each ``U_A``; see the module docstring, and note
        that these are two different approximations rather than two spellings of one.
    local : mapping, optional
        ``{label: (hcore_A, overlap_A)}`` for ``source="isolated"``. Checked against the
        molecular overlap by :func:`check_local_blocks` before anything is solved.

    Returns
    -------
    (X, R), both ``(2*nao, 2*nao)`` and block-diagonal on ``partition``.
    """
    if source not in LOCAL_SOURCES:
        raise ValueError("unknown local decoupling source {!r}; expected one of {}"
                         .format(source, LOCAL_SOURCES))
    n2c = hcore.n2c
    partition.validate(n2c)
    if source == "isolated":
        if not local:
            raise ValueError(
                "source='isolated' needs the isolated-fragment four-component blocks; none "
                "were supplied. They come from the front-end, which is the only layer that "
                "can build them (kuiva.interface.pyscf_bridge.isolated_fragment_blocks).")
        check_local_blocks(overlap, partition, local)

    # ⚠ Outside the loop (never a memory check inside a loop). The per
    # fragment workspaces are transient and are not accounted; ``X`` and ``R`` are resident and
    # are. The whole point of DLU is that the workspace this *avoids* —
    # :func:`~kuiva.x2c.decouple.exact_decoupling_workspace_gb` over the full basis — never
    # appears here at all; the largest fragment's is what is paid instead.
    res.reserve("DLU decoupling matrices", decoupling_memory_gb(n2c // 2),
                note="{} fragments over nao = {}".format(len(partition), n2c // 2),
                advice=["the local decoupling is already the cheap path; a limit that refuses "
                        "it would refuse the exact one many times over"])

    x = np.zeros((n2c, n2c), dtype=np.complex128)
    r = np.zeros((n2c, n2c), dtype=np.complex128)
    scales = []
    for label, idx in zip(partition.labels, partition.indices):
        if source == "isolated":
            h_a, s_a = local[label]
        else:
            h_a, s_a = sub_blocks(hcore, idx), sub_blocks(overlap, idx)
        x_a, r_a = decoupling_matrices(h_a, s_a, light_speed)
        grid = np.ix_(np.asarray(idx, dtype=np.intp), np.asarray(idx, dtype=np.intp))
        x[grid] = x_a
        r[grid] = r_a
        scales.append((label, float(np.max(np.abs(x_a))) if x_a.size else 0.0))

    if report:
        worst = max(scales, key=lambda item: item[1])
        log.debug("DLU decoupling over %d fragments (source=%s); largest max|X| is %.3g on %s",
                  len(partition), source, worst[1], worst[0])
    return np.ascontiguousarray(x), np.ascontiguousarray(r)


def local_block_scales(x: np.ndarray, partition: Partition) -> Dict[str, float]:
    """``{label: max |X_A|}`` — the conditioning diagnostic, per fragment.

    ⚠ This is the number that catches a near-singular decontracted heavy-element basis, and
    the one-electron reproduction check provably cannot (see
    :func:`kuiva.x2c.decouple.canonical_orth`). Order 10 is healthy, 1e3 is not. Reported per
    fragment rather than globally so that one bad heavy atom cannot hide behind a dozen
    healthy ligands.
    """
    out = {}
    for label, idx in zip(partition.labels, partition.indices):
        i = np.asarray(idx, dtype=np.intp)
        block = x[np.ix_(i, i)]
        out[label] = float(np.max(np.abs(block))) if block.size else 0.0
    return out


def off_block_weight(a: np.ndarray, partition: Partition) -> float:
    """Fraction of ``max |A|`` carried **outside** the partition's diagonal blocks.

    What DLU approximates away in the *transformation* is exactly what this measures in the
    operator: a molecule whose four-component Hamiltonian is already block-diagonal is one for
    which DLU is exact. It is therefore the cheapest available predictor of the DLU error, and
    it goes to zero as fragments separate — which is the statement the dissociation test makes
    quantitative.
    """
    a = np.asarray(a)
    scale = float(np.max(np.abs(a)))
    if scale == 0.0:
        return 0.0
    mask = np.ones(a.shape, dtype=bool)
    for idx in partition.indices:
        i = np.asarray(idx, dtype=np.intp)
        mask[np.ix_(i, i)] = False
    return float(np.max(np.abs(a[mask]))) / scale if mask.any() else 0.0


__all__ = ["LOCAL_SOURCES", "OVERLAP_MATCH_TOLERANCE", "Partition", "check_local_blocks",
           "local_block_scales", "local_decoupling_matrices", "off_block_weight",
           "sub_blocks"]
