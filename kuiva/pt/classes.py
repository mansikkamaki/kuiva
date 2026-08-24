"""The eight SC-NEVPT2 excitation classes, behind a name -> factory registry.

**Orchestration plus dense algebra, not a registered kernel.** The arithmetic is ``tensordot`` over
the auxiliary index and two GEMMs per class; the batching comes from :mod:`kuiva.pt.blocks`.

Why a registry
--------------
The design requires that "dispatch MUST be designed so new classes drop in", and this is
the fourth instance of the project's one extension pattern (``ci/kernels.py``,
``amf/backend.py``, ``qc/algorithms.py``). A registered class declares the orbital spaces its
external labels run over, the contraction primitives it asks the provider for, and what it can
currently produce; FIC-NEVPT2 later re-implements the same eight *names* with a per-class
metric and the driver does not change (the FIC-ready rule (primitives, never SC-assembled quantities)).

⚠ **All eight are registered — deliberately unlike ``qc/algorithms.py``, and the rule outlives
the stage that needed it.** There the rule is "only implemented algorithms are registered",
because an algorithm set is open-ended and a name resolving to something non-functional fails
far from its cause. The eight classes are not an open-ended set: they are a *partition of the
first-order interacting space*, so a missing name would make the total silently incomplete
instead of visibly so. Each carries a :attr:`ExcitationClass.status`; all eight are
``"energy"`` today, and the driver still refuses to call the sum ``E2`` if that ever stops
being true — a class added or disabled behind a flag must not be able to shrink ``E2`` quietly.

The spinor derivation
---------------------
Every formula below is derived in spinor second quantization in the docstring of the class
that uses it, and nothing is transcribed from a spin-free paper. Two
consequences of the spinor setting recur and are stated once here:

* **No alpha/beta factorization and no spin-traced ``E_pq``.** Every operator is a single
  spinor ladder operator, so the antisymmetrized integral ``(ai|bj) - (aj|bi)`` appears
  directly rather than as a ``2J - K`` combination.
* **4-fold integral symmetry only**. ``(pq|rs) = (rs|pq)`` and ``(pq|rs) = (qp|sr)*``
  are used; ``(pq|rs) = (rq|ps)`` is **false** and no expression here may rely on it.

Index convention, fixed for the whole module: ``i, j`` inactive, ``t, u, v, w`` active,
``a, b`` virtual. ⚠ The *class names* keep the literature's ``r, s`` for the virtual labels
(``Sijrs``, ``Srs``, ``Srsi``, ``Sr``) so that a per-class table lines up with PySCF's and with
Angeli's papers; the code says ``a, b``.

Perturbers, norms and denominators
----------------------------------
For a class ``k`` and one set of external labels ``l``, the strongly contracted perturber is
``|Psi_l> = P_l H |Psi_0>``, a single function. Because ``P_l`` is a projector,

::

    N_l  = <Psi_l|Psi_l> = <Psi_0|H P_l H|Psi_0>          (the "norm")
    dE_l = <Psi_l|H_D|Psi_l> / N_l - E_0
    E^k  = - sum_l N_l / dE_l

Both ``N_l`` and ``dE_l`` are independent of how ``|Psi_l>`` is scaled, so they are
convention-free numbers that any correct implementation must reproduce — which is what makes
the per-class comparison against PySCF (Tier 1) a real check rather than a
comparison of two conventions.

⚠ **THE CONTRACTION IS OVER WHOLE DEGENERATE-``eps`` GROUPS, NOT OVER SINGLE SPINORS**, and
that is a correctness requirement rather than a refinement. Strong contraction fixes one
perturber per external *label set*, so how finely the labels are resolved is part of the
method — and a label finer than the arbitrariness in the orbitals is not a label at all. The
pseudo-canonicalization leaves an arbitrary unitary **inside every degenerate ``eps`` block**
, Kramers pairs among them, so a per-spinor contraction makes ``E2`` depend on a choice
the eigensolver made: measured at 1.3e-11 Eh on a 3.3e-6 Eh class energy for ``Sir (0')`` on
LiH/STO-3G, which is 4e-6 *relative* and entirely arbitrary.

Lumping restores exact invariance, and it is a theorem rather than a tolerance: within a group
the perturbers transform among themselves unitarily, and the group's two accumulated
quantities are the **traces** of the overlap and Hamiltonian Gram matrices over that group,
which no unitary can move. So

::

    N_G  = sum_{l in G} N_l,   D_G = sum_{l in G} N_l dE_l,   E^G = - N_G^2 / D_G

⚠ **On a real system it is a no-op for every class but one, and the reason is worth stating
because it is what makes the rule easy to miss.** A physical ``eps`` degeneracy *comes from* a
symmetry the integrals share — Kramers, or a point group — so the denominators are constant
inside the group and lumping changes nothing (measured at <=3e-17 Eh). ``Sir (0')`` is the
exception and structurally so: its same-spin perturber carries the one-body ``f^I_ai`` *and*
the direct integral ``(ai|tu)`` while its spin-flip partner carries only the exchange term
``-(au|ti)``, so the two are not related by any symmetry and their denominators differ by
2.9e-2 Eh on LiH/STO-3G. (With a degeneracy imposed *without* the matching integral symmetry —
which is what the synthetic test does — every class feels the rule; that is how the code is
checked to be applying it at all.)

⚠ **In the scalar limit it reproduces the published spin-free SC-NEVPT2 class by class**,
measured to 1e-14 relative against PySCF on every implemented class. The two partitions are not
literally the same — Kuiva's groups are the *coarser* one wherever two spatial orbitals are
symmetry degenerate — and that they agree anyway is the same statement as the paragraph above.
With SOC on the groups are Kramers pairs, which is the same rule with the only symmetry that
survives. There is no knob here and there will not be one: a per-spinor contraction is not a
well-defined quantity.

⚠ **Sums over label pairs are written over the full range with a prefactor, not over ``a < b``.**
``N_l`` is symmetric under exchanging the two labels of a pair and vanishes identically when
they coincide (the perturber contains ``a+_a a+_b``), so ``sum_{a<b} = 1/2 sum_{a,b}`` exactly
and the diagonal contributes nothing. The prefactor form is one contiguous array instead of a
triangular gather; the vanishing diagonal is a property of the *formula*, not of a mask, and a
test asserts it.

Corrected RDM-rank bookkeeping
------------------------------
The planning-level rank table was an estimate and is superseded by the derivation
(the planning-level guess said it would be). What each class actually needs:

======================  ==================  ==============================
class                   norm needs          denominator additionally needs
======================  ==================  ==============================
``Sijrs (0)``           --                  -- (bare ``eps``)
``Srsi (-1)``           1-RDM               2-RDM
``Sijr (+1)``           hole 1-RDM          2-RDM
``Srs (-2)``            2-RDM               3-RDM
``Sij (+2)``            hole 2-RDM          3-RDM
``Sir (0')``            2-RDM               3-RDM
``Sr (-1')``            3-RDM               4-RDM, contracted
``Si (+1')``            hole 3-RDM          4-RDM, contracted
======================  ==================  ==============================

So the first three classes are complete with the rank-2 provider, which is why they are all
implemented in the first stage rather than only ``Sijrs``.

⚠ **The whole right-hand column says what the quantity IS, not what gets built, and nothing on
the conventional-CI path allocates an ``n_act^6`` array — including for the last two rows.**
Every rank-3 entry has the form ``<Psi| O+ (H_act - E) O |Psi>`` for a ladder string ``O`` and
is served as a Gram matrix of explicitly constructed vectors. The rank-4 entries are one step
further: the primed classes contract the integrals into **one perturber vector per external
label** before any Gram is taken, so the "contracted 4-RDM" is never an object at all. See
:mod:`kuiva.pt.contractions` for both routes and for the crossover between them.

References
----------
* C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger, J.-P. Malrieu, J. Chem. Phys. 114,
  10252 (2001), doi:10.1063/1.1361246 — the classes and the strongly contracted perturbers.
* C. Angeli, R. Cimiraglia, J.-P. Malrieu, J. Chem. Phys. 117, 9138 (2002),
  doi:10.1063/1.1515317 — n-electron valence perturbation theory, second order.
* C. Angeli, M. Pastore, R. Cimiraglia, Theor. Chem. Acc. 117, 743 (2007),
  doi:10.1007/s00214-006-0207-0 — review, and the class-by-class working equations.
* K. G. Dyall, J. Chem. Phys. 102, 4909 (1995), doi:10.1063/1.469539 — ``H_D``.
* Q. Sun et al., WIREs Comput. Mol. Sci. 8, e1340 (2018), doi:10.1002/wcms.1340 — PySCF, whose
  ``mrpt.NEVPT`` per-class print this module's reporting deliberately mirrors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from ..util import resources as res
from ..util.logging import get_logger
from .blocks import IntegralBlocks, batch_gb, batch_slices
from .contractions import perturber_vector_gb

log = get_logger(__name__)

#: Perturber norm below which a label set is dropped from the energy sum. ⚠ This is a
#: **division guard, not a physical threshold**: ``dE_l`` is built as ``eps + K_l / N_l``, so a
#: label whose perturber has vanishing norm is a ``0/0`` and contributes nothing in exact
#: arithmetic. It is applied group-completely (:func:`class_energy`) so that it can never split
#: a Kramers pair.
DEFAULT_NORM_CUTOFF = 1.0e-14

#: Relative width within which two perturber denominators count as degenerate for the
#: group-complete cutoff above. Sized like :data:`kuiva.rdm.rdm.DEFAULT_DEGENERACY_TOL`, but
#: relative, because the quantity grouped is an orbital-energy difference of order 1 Eh rather
#: than a state energy.
DENOMINATOR_DEGENERACY_RTOL = 1.0e-9

#: Relative width within which two orbital energies count as degenerate, and hence within which
#: the canonical spinors are undetermined. ⚠ **Exact degeneracy is the only case that matters,
#: and that is why a tight threshold is right rather than cautious**: a merely *near*-degenerate
#: pair leaves the eigenvectors determined, so there is no arbitrariness to protect against and
#: nothing to lump. Kramers pairs and symmetry partners are degenerate to rounding; anything
#: else is a different orbital.
EPS_DEGENERACY_RTOL = 1.0e-9


# --- results and context ---------------------------------------------------------------------

@dataclass(frozen=True)
class ClassResult:
    """What one excitation class produces for one state."""

    name: str
    #: ``sum_l N_l`` over the class. Always available where the class is implemented at all;
    #: it needs one rank less than the energy does, which is why the stages split here.
    norm: float
    #: ``E^k = -sum_l N_l / dE_l`` [Eh], or ``None`` when the class cannot yet form its
    #: denominators. ⚠ ``None`` is not zero and the driver may not add it as one.
    energy: Optional[float] = None
    #: Distinct label sets whose perturber did not vanish and which entered the sum, and how
    #: many the cutoff removed. ⚠ Counted after the prefactor of the module docstring's
    #: ordered-pair convention, i.e. as *distinct* labels, not as enumerated tuples. For a
    #: class that reports only a norm there is no cutoff and no sum, so it is simply the
    #: combinatorial label count.
    n_perturbers: int = 0
    n_dropped: int = 0
    #: Smallest ``|dE_l|`` over the kept labels — the intruder diagnostic.
    min_denominator: Optional[float] = None
    #: Smallest **signed** ``dE_l`` over the kept labels. ⚠ Kept beside the absolute one
    #: because the two answer different questions: a small ``|dE|`` says the perturbation is
    #: badly conditioned, a **non-positive** ``dE`` says a perturber has fallen below the
    #: reference, which makes the class energy wrong in sign as well as in size. The absolute
    #: value alone cannot see the second, and the second is the more serious.
    min_signed_denominator: Optional[float] = None
    #: Largest discarded imaginary part of a quantity that is real by construction. A
    #: growing value is the cheapest detector of a conjugation error.
    max_imaginary: float = 0.0

    def __repr__(self) -> str:
        e = "None" if self.energy is None else "{:.10f}".format(self.energy)
        return "ClassResult({}, norm={:.6e}, E={})".format(self.name, self.norm, e)


@dataclass(frozen=True)
class ClassContext:
    """Everything a class evaluation is allowed to read. One state, one set of orbitals."""

    blocks: IntegralBlocks
    #: A :class:`kuiva.pt.contractions.CIContractionProvider` (duck-typed: the classes call
    #: only the primitive methods, so a network-backed provider drops in unchanged).
    provider: object
    eps_inactive: np.ndarray
    eps_virtual: np.ndarray
    #: The **inactive** Fock over ``(virtual, inactive)``, i.e. ``f^I_ai``. Only the ``Sir``
    #: class needs it, and it needs it because that class's perturber has a one-body part
    #: (see :func:`_eval_sir`). ⚠ ``F^I``, not ``F^I + F^A``: the active mean field belongs to
    #: the two-body coefficients, and adding it here would double-count.
    fock_vi: Optional[np.ndarray] = None
    #: ``f^I_at`` over ``(virtual, active)`` — the one-body half of the ``Sr (-1')`` perturber.
    fock_va: Optional[np.ndarray] = None
    #: ``f^I_ti`` over ``(active, inactive)`` — the one-body half of the ``Si (+1')`` perturber.
    fock_ai: Optional[np.ndarray] = None
    #: Level shift [Eh] added to every denominator, and whether it is imaginary.
    shift: float = 0.0
    imaginary_shift: bool = False
    norm_cutoff: float = DEFAULT_NORM_CUTOFF

    #: Degeneracy-group index per inactive / virtual spinor. Filled from the ``eps`` on
    #: construction; the contraction is over these groups (see the module docstring).
    group_inactive: Optional[np.ndarray] = None
    group_virtual: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.group_inactive is None:
            object.__setattr__(self, "group_inactive", degeneracy_groups(self.eps_inactive))
        if self.group_virtual is None:
            object.__setattr__(self, "group_virtual", degeneracy_groups(self.eps_virtual))

    @property
    def n_inactive(self) -> int:
        return int(self.eps_inactive.size)

    @property
    def n_virtual(self) -> int:
        return int(self.eps_virtual.size)

    @property
    def n_active(self) -> int:
        return int(self.blocks.size("active"))

    @property
    def n_group_inactive(self) -> int:
        return int(self.group_inactive.max()) + 1 if self.n_inactive else 0

    @property
    def n_group_virtual(self) -> int:
        return int(self.group_virtual.max()) + 1 if self.n_virtual else 0


# --- the shared energy sum, with the group-complete cutoff --------------------------------

def _sorted_groups(values: np.ndarray, rtol: float):
    """``(order, starts)``: the sort permutation and the start of each degeneracy group in it."""
    order = np.argsort(values, kind="stable")
    sortd = values[order]
    scale = max(float(np.max(np.abs(sortd))), 1.0)
    starts = np.concatenate([[0], np.nonzero(np.diff(sortd) > rtol * scale)[0] + 1])
    return order, starts.astype(np.intp)


def degeneracy_groups(values: np.ndarray,
                      rtol: float = EPS_DEGENERACY_RTOL) -> np.ndarray:
    """Group index per orbital, grouping exactly-degenerate orbital energies together.

    The contraction groups of the module docstring: within one of these the canonical spinors
    are defined only up to a unitary, so a per-orbital external label is not well defined and
    the perturbers must be contracted as a set.
    """
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return np.zeros(0, dtype=np.intp)
    order, starts = _sorted_groups(values, rtol)
    sizes = np.diff(np.append(starts, values.size))
    groups = np.empty(values.size, dtype=np.intp)
    groups[order] = np.repeat(np.arange(starts.size, dtype=np.intp), sizes)
    return groups


def unordered_pair_key(first: np.ndarray, second: np.ndarray, n_groups: int) -> np.ndarray:
    """A single integer key for the **unordered** pair of group indices.

    ⚠ Unordered on purpose: the perturber of a label pair is symmetric under exchanging the
    two labels (module docstring), so ``(a, b)`` and ``(b, a)`` are the same contracted
    function and must land in the same group. Ordering them would double the group count and
    halve every group, which changes the lumped energy without changing any norm.
    """
    lo = np.minimum(first, second)
    hi = np.maximum(first, second)
    return lo * int(n_groups) + hi


def denominator_groups(denominators: np.ndarray,
                       rtol: float = DENOMINATOR_DEGENERACY_RTOL) -> np.ndarray:
    """Group index for each perturber, grouping exactly-degenerate denominators together.

    ⚠ **Grouping by the denominator is a deliberate over-approximation of the Kramers orbit,
    and the over-approximation is the safe direction.** Two perturbers related by time reversal
    have *exactly* equal denominators, because the Dyall ``eps`` come in exactly degenerate
    Kramers pairs when ``H0`` is built from a time-even density and the active part is the
    exact active Hamiltonian. So every Kramers orbit lies inside one group here, and dropping
    whole groups can never split a pair — which is the required property. The converse is
    not claimed: a group may contain accidentally degenerate labels too, and dropping those
    together is harmless.
    """
    values = np.asarray(denominators, dtype=float).ravel()
    if values.size == 0:
        return np.zeros(0, dtype=np.intp)
    order, starts = _sorted_groups(values, rtol)
    sizes = np.diff(np.append(starts, values.size))
    groups = np.empty(values.size, dtype=np.intp)
    groups[order] = np.repeat(np.arange(starts.size, dtype=np.intp), sizes)
    return groups


def group_complete_mask(norms: np.ndarray, denominators: np.ndarray, cutoff: float,
                        rtol: float = DENOMINATOR_DEGENERACY_RTOL) -> np.ndarray:
    """Keep-mask for the small-norm cutoff, dropping **whole** degeneracy groups.

    A group is dropped only when its *largest* member is below ``cutoff``. Erring toward
    keeping is deliberate: a group half of whose members sit on either side of the threshold
    through rounding is exactly the situation that would manufacture a Kramers splitting, and
    that is the one thing this cutoff must not do.

    ⚠ **The mask must be built over the class's whole label set, never per batch.** A group
    that straddles two batches would get two different maxima and could then be cut in one and
    kept in the other, which reintroduces the defect the grouping exists to prevent. That is
    why the classes accumulate their flat ``(norm, denominator)`` arrays and call this once.
    """
    norms = np.asarray(norms, dtype=float)
    if norms.size == 0:
        return np.zeros(0, dtype=bool)
    order, starts = _sorted_groups(np.asarray(denominators, dtype=float), rtol)
    sizes = np.diff(np.append(starts, norms.size))
    group_max = np.maximum.reduceat(np.abs(norms)[order], starts)
    keep = np.empty(norms.size, dtype=bool)
    keep[order] = np.repeat(group_max > float(cutoff), sizes)
    return keep


def class_energy(name: str, norms: np.ndarray, denominators: np.ndarray,
                 ctx: ClassContext, *, prefactor: float = 1.0,
                 groups: Optional[np.ndarray] = None,
                 cutoff: bool = True) -> ClassResult:
    """The class energy, contracted over ``groups`` and with the small-norm cutoff.

    ``norms`` and ``denominators`` are flat, same-shape, real arrays over the class's label
    sets, already carrying whatever redundancy the prefactor compensates for (see the module
    docstring on the ``sum_{a<b} = 1/2 sum_{a,b}`` convention).

    With ``groups`` — an integer contraction-group index per label — the sum is
    ``-prefactor * sum_G N_G^2 / D_G`` with ``N_G = sum_G N_l`` and ``D_G = sum_G N_l dE_l``,
    which is the *one* contracted function per degenerate-``eps`` group the module docstring
    requires. ⚠ The prefactor cancels correctly against the redundant enumeration: a
    ``k``-fold redundancy multiplies both ``N_G`` and ``D_G`` by ``k``, so ``N_G^2 / D_G``
    picks up one factor of ``k`` and the ``1/k`` prefactor removes it.

    Without ``groups`` every label is its own contracted function. ⚠ Only legitimate where the
    denominator is provably constant inside a group — which is exactly ``Sijrs``, whose
    denominator is built from ``eps`` alone. Anywhere else it makes ``E2`` depend on an
    arbitrary eigenvector choice.

    ``cutoff=False`` additionally skips the small-norm guard, for the same class and the same
    reason: its denominator does not contain ``1/N_l``, so a vanishing norm contributes exactly
    zero and the class may be evaluated batch by batch. Every other class must assemble its
    whole label set first — see :func:`group_complete_mask`.
    """
    norms = np.ascontiguousarray(np.asarray(norms, dtype=float).ravel())
    denominators = np.ascontiguousarray(np.asarray(denominators, dtype=float).ravel())
    if norms.shape != denominators.shape:
        raise ValueError("{}: {} norms against {} denominators"
                         .format(name, norms.size, denominators.size))
    total_norm = prefactor * float(norms.sum())
    if norms.size == 0:
        return ClassResult(name=name, norm=0.0, energy=0.0, n_perturbers=0)

    if groups is not None:
        return _lumped_energy(name, norms, denominators, np.asarray(groups).ravel(), ctx,
                              prefactor, total_norm)

    if cutoff:
        keep = group_complete_mask(norms, denominators, ctx.norm_cutoff)
    else:
        keep = np.abs(norms) > 0.0
    n_dropped = int(round(prefactor * np.count_nonzero(
        np.logical_and(~keep, np.abs(norms) > 0.0))))

    kept_norms = norms[keep]
    kept_denom = denominators[keep]
    if kept_norms.size == 0:
        return ClassResult(name=name, norm=total_norm, energy=0.0, n_perturbers=0,
                           n_dropped=n_dropped)
    contributions = _shifted_ratio(kept_norms, kept_denom, ctx)
    energy = -prefactor * float(contributions.sum())
    return ClassResult(name=name, norm=total_norm, energy=energy,
                       n_perturbers=int(round(prefactor * kept_norms.size)),
                       n_dropped=n_dropped,
                       min_denominator=float(np.min(np.abs(kept_denom))),
                       min_signed_denominator=float(np.min(kept_denom)))


def _shifted_ratio(norms: np.ndarray, denominators: np.ndarray,
                   ctx: ClassContext) -> np.ndarray:
    """``N / dE`` with the optional optional level shift."""
    if ctx.shift == 0.0:
        return norms / denominators
    if ctx.imaginary_shift:
        # E = -sum N * dE / (dE^2 + sigma^2): the shift never changes the sign of a
        # contribution and vanishes quadratically away from the intruder.
        return norms * denominators / (denominators ** 2 + ctx.shift ** 2)
    return norms / (denominators + np.sign(denominators) * ctx.shift)


def _lumped_energy(name: str, norms: np.ndarray, denominators: np.ndarray,
                   groups: np.ndarray, ctx: ClassContext, prefactor: float,
                   total_norm: float) -> ClassResult:
    """One contracted function per group; see :func:`class_energy`."""
    unique, index = np.unique(groups, return_inverse=True)
    n_group = np.bincount(index, weights=norms, minlength=unique.size)
    d_group = np.bincount(index, weights=norms * denominators, minlength=unique.size)
    # ⚠ Group-complete by construction: the cutoff sees whole contracted functions, so the
    # separate grouping has nothing left to do here.
    keep = np.abs(n_group) > float(ctx.norm_cutoff)
    n_dropped = int(np.count_nonzero(np.logical_and(~keep, np.abs(n_group) > 0.0)))
    if not np.any(keep):
        return ClassResult(name=name, norm=total_norm, energy=0.0, n_perturbers=0,
                           n_dropped=n_dropped)
    kept_norm = n_group[keep]
    # dE_G = D_G / N_G, so E_G = -N_G^2 / D_G = -N_G / dE_G: the same shape as the unlumped
    # sum with the group's accumulated pair in place of the label's.
    kept_denom = d_group[keep] / kept_norm
    energy = -prefactor * float(_shifted_ratio(kept_norm, kept_denom, ctx).sum())
    return ClassResult(name=name, norm=total_norm, energy=energy,
                       n_perturbers=int(np.count_nonzero(keep)), n_dropped=n_dropped,
                       min_denominator=float(np.min(np.abs(kept_denom))),
                       min_signed_denominator=float(np.min(kept_denom)))


def label_buffer_gb(n_labels: int) -> float:
    """[GB] of the three arrays a class holds over its whole label set.

    Exact and unpadded: a norm, a denominator and a contraction-group index per label
    tuple of the *redundant* enumeration. They exist because both the contraction and the
    small-norm cutoff have to see every label of a group at once, and a group can straddle any
    batching — so they are the price of the module docstring's grouping rule, and are
    accounted as such.
    """
    return (2.0 * res.array_gb((int(n_labels),), np.float64)
            + res.array_gb((int(n_labels),), np.int64))


def _label_buffers(name: str, n_labels: int):
    res.require("NEVPT2 {} perturber labels".format(name), label_buffer_gb(n_labels),
                note="{} norm/denominator/group triples over the redundant label enumeration"
                     .format(n_labels),
                advice=["freeze core spinors: the label count of this "
                        "class is linear or quadratic in the correlated core",
                        "delete high virtuals, which enter every class label range"])
    return (np.empty(int(n_labels), dtype=np.float64),
            np.empty(int(n_labels), dtype=np.float64),
            np.empty(int(n_labels), dtype=np.int64))


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """``K_l / N_l`` with the ``N_l = 0`` entries set to zero rather than to ``nan``.

    A label whose perturber vanishes identically (the coincident-label entries of the
    redundant enumeration, see the module docstring) has ``K_l = 0`` too, so the ratio is a
    genuine ``0/0`` that contributes nothing: its energy term is ``-N_l/dE_l = 0`` for any
    finite ``dE_l``. Substituting zero keeps the denominator finite so that the grouping in
    :func:`group_complete_mask` sees a real number instead of a ``nan`` that would sort
    unpredictably.
    """
    out = np.zeros_like(numerator)
    nonzero = denominator != 0.0
    np.divide(numerator, denominator, out=out, where=nonzero)
    return out


def _quadratic_form(x: np.ndarray, kernel: np.ndarray) -> Tuple[np.ndarray, float]:
    """``sum_kl conj(x[P,k]) M[k,l] x[P,l]`` for a batch of rows, as two GEMMs.

    Returns the real part and the largest discarded imaginary part. Hermitian ``M`` makes the
    form real; a nonzero imaginary part is a conjugation error, not rounding, once it is
    above the scale of the result.
    """
    z = x @ kernel.T                                       # z[P,k] = sum_l M[k,l] x[P,l]
    value = np.einsum("Pk,Pk->P", x.conj(), z)
    return np.ascontiguousarray(value.real), float(np.max(np.abs(value.imag)) if value.size
                                                   else 0.0)


# --- the classes -------------------------------------------------------------------------------

def _eval_sijrs(ctx: ClassContext) -> ClassResult:
    """``S_ijrs^(0)``: ``ij -> ab``. Two inactive holes, two virtual particles, ``dN_act = 0``.

    The only class with no active-space content at all. The part of ``H`` that reaches it is
    ``1/2 sum (ai|bj) a+_a a+_b a_j a_i``, and for one label set ``(i<j, a<b)`` the perturber
    is a *single determinant-like function* proportional to the reference,

    ::

        |Psi_{ijab}> = g_{abij} a+_a a+_b a_j a_i |Psi_0>,  g_{abij} = (ai|bj) - (aj|bi)

    so ``N = |g|^2`` and, since ``H_D`` counts one ``eps`` per changed occupation and leaves
    the active part untouched, ``dE = eps_a + eps_b - eps_i - eps_j`` exactly. That makes this
    class formally an MP2 over the inactive/virtual spinors, and it is where the bulk of the
    correlation energy of a large system sits.

    ⚠ Its denominator is bounded away from zero by the inactive/virtual gap and does not
    contain ``1/N_l``, so this class needs no small-norm guard and is therefore the only one
    that may be summed **batch by batch** (:func:`class_energy` with ``cutoff=False``). Every
    other class has to see its whole label set before it can cut group-completely.
    """
    nv, nc = ctx.n_virtual, ctx.n_inactive
    eps_v, eps_i = ctx.eps_virtual, ctx.eps_inactive
    if nv < 2 or nc < 2:
        return ClassResult(name="Sijrs", norm=0.0, energy=0.0)
    b = ctx.blocks.three_index("virtual", "inactive")       # B^P_{a i}

    norm_total = 0.0
    energy_total = 0.0
    n_kept = 0
    min_denom = np.inf
    min_signed = np.inf
    # Batch over the *virtual* index: the four-index batch is (nba, nc, nv, nc), so a core
    # batch would scale as nv^2 and a virtual one only as nv. One transient-budget query.
    per_a = batch_gb((nc, nv, nc), count=3)
    for sl in batch_slices(nv, per_a):
        # g[a,i,b,j] = (ai|bj) - (aj|bi). The exchange term is the same array read at
        # [a, j, b, i], available because only `a` is batched. ⚠ Not an in-place subtraction:
        # the transpose is a view of the same buffer.
        gab = np.tensordot(b[:, sl, :], b, axes=([0], [0]))         # (nba, nc, nv, nc)
        gab = gab - gab.transpose(0, 3, 2, 1)
        d = (eps_v[sl][:, None, None, None] + eps_v[None, None, :, None]
             - eps_i[None, :, None, None] - eps_i[None, None, None, :])
        n = (gab.real ** 2 + gab.imag ** 2).ravel()
        part = class_energy("Sijrs", n, d.ravel(), ctx, prefactor=0.25, cutoff=False)
        norm_total += part.norm
        energy_total += part.energy
        n_kept += part.n_perturbers
        if part.min_denominator is not None:
            min_denom = min(min_denom, part.min_denominator)
        if part.min_signed_denominator is not None:
            min_signed = min(min_signed, part.min_signed_denominator)
    return ClassResult(name="Sijrs", norm=norm_total, energy=energy_total,
                       n_perturbers=n_kept,
                       min_denominator=None if not np.isfinite(min_denom) else min_denom,
                       min_signed_denominator=(None if not np.isfinite(min_signed)
                                               else min_signed))


def _eval_srsi(ctx: ClassContext) -> ClassResult:
    """``S_rsi^(-1)``: ``i t -> a b``. One inactive hole, two virtual particles, ``dN_act = -1``.

    Collecting the terms of ``H`` with two virtual creations, one inactive and one active
    annihilation, and adding the ``(a,b)`` and ``(b,a)`` contributions gives, for one label
    set ``(i, a<b)``,

    ::

        |Psi_{iab}> = sum_t g_t a+_a a+_b a_t a_i |Psi_0>,  g_t = (ai|bt) - (at|bi)

    The inactive index factorizes out of every matrix element (``a+_i ... a_i`` becomes the
    occupation number of ``i``, and the active string in between has even length so it passes
    through without a sign), leaving a plain quadratic form in ``g``:

    ::

        N    = sum_tu conj(g_t) gamma_tu g_u                       (1-RDM only)
        dE   = eps_a + eps_b - eps_i + K_iab / N,
        K_iab = sum_tu conj(g_t) K_tu g_u,  K_tu = <a+_t (H_act - E) a_u>

    ``K`` is :func:`kuiva.pt.contractions.koopmans_annihilation` and needs only the 2-RDM, so
    this class was complete far earlier than expected — the ``1,2-RDM`` estimate for the norm
    was one rank too pessimistic.
    """
    nv, nc, na = ctx.n_virtual, ctx.n_inactive, ctx.n_active
    if nv < 2 or nc == 0 or na == 0:
        return ClassResult(name="Srsi", norm=0.0, energy=0.0)
    b_vi = ctx.blocks.three_index("virtual", "inactive")    # (naux, nv, nc)
    b_va = ctx.blocks.three_index("virtual", "active")      # (naux, nv, na)
    gamma = ctx.provider.rdm1()
    koop = ctx.provider.koopmans_annihilation()
    eps_v, eps_i = ctx.eps_virtual, ctx.eps_inactive

    norms, denominators, keys = _label_buffers("Srsi", nc * nv * nv)
    gv, gi_all, ngv = ctx.group_virtual, ctx.group_inactive, ctx.n_group_virtual
    max_imag = 0.0
    filled = 0
    per_a = batch_gb((nc, nv, na), count=4)
    for sl in batch_slices(nv, per_a):
        # (ai|bt) with a batched -> [a, i, b, t];  (at|bi) -> [a, t, b, i], transposed to match
        g = np.tensordot(b_vi[:, sl, :], b_va, axes=([0], [0]))     # (nba, nc, nv, na)
        g -= np.tensordot(b_va[:, sl, :], b_vi,
                          axes=([0], [0])).transpose(0, 3, 2, 1)
        x = g.reshape(-1, na)                                       # rows (a, i, b)
        n, imag_n = _quadratic_form(x, gamma)
        k, imag_k = _quadratic_form(x, koop)
        max_imag = max(max_imag, imag_n, imag_k)
        eps = (eps_v[sl][:, None, None] + eps_v[None, None, :]
               - eps_i[None, :, None]).ravel()
        norms[filled:filled + n.size] = n
        denominators[filled:filled + n.size] = eps + _safe_ratio(k, n)
        keys[filled:filled + n.size] = (
            gi_all[None, :, None] * (ngv * ngv)
            + unordered_pair_key(gv[sl][:, None, None], gv[None, None, :], ngv)).ravel()
        filled += n.size
    result = class_energy("Srsi", norms[:filled], denominators[:filled], ctx, prefactor=0.5,
                          groups=keys[:filled])
    return ClassResult(name="Srsi", norm=result.norm, energy=result.energy,
                       n_perturbers=result.n_perturbers, n_dropped=result.n_dropped,
                       min_denominator=result.min_denominator,
                       min_signed_denominator=result.min_signed_denominator,
                       max_imaginary=max_imag)


def _eval_sijr(ctx: ClassContext) -> ClassResult:
    """``S_ijr^(+1)``: ``i j -> a t``. Two inactive holes, one virtual particle, ``dN_act = +1``.

    The particle-hole mirror of ``Srsi``. For one label set ``(i<j, a)``,

    ::

        |Psi_{ija}> = sum_t g_t a+_a a+_t a_j a_i |Psi_0>,  g_t = (ai|tj) - (aj|ti)
        N     = sum_tu conj(g_t) <a_t a+_u> g_u             (hole 1-RDM only)
        dE    = eps_a - eps_i - eps_j + K'_ija / N,
        K'_tu = <a_t (H_act - E) a+_u>

    ``<a_t a+_u> = delta_tu - gamma_ut`` is :func:`kuiva.pt.contractions.hole_rdm1` and ``K'``
    is :func:`kuiva.pt.contractions.koopmans_creation`, which again needs only the 2-RDM.
    """
    nv, nc, na = ctx.n_virtual, ctx.n_inactive, ctx.n_active
    if nv == 0 or nc < 2 or na == 0:
        return ClassResult(name="Sijr", norm=0.0, energy=0.0)
    b_vi = ctx.blocks.three_index("virtual", "inactive")    # (naux, nv, nc)
    b_ai = ctx.blocks.three_index("active", "inactive")     # (naux, na, nc)
    hole = ctx.provider.hole_rdm1()
    koop = ctx.provider.koopmans_creation()
    eps_v, eps_i = ctx.eps_virtual, ctx.eps_inactive

    norms, denominators, keys = _label_buffers("Sijr", nv * nc * nc)
    gv, gi_all, ngi = ctx.group_virtual, ctx.group_inactive, ctx.n_group_inactive
    max_imag = 0.0
    filled = 0
    per_a = batch_gb((nc, na, nc), count=4)
    for sl in batch_slices(nv, per_a):
        # (ai|tj) with a batched -> [a, i, t, j]; the exchange term swaps i and j. ⚠ Not an
        # in-place subtraction: the transpose is a view of the same buffer.
        g = np.tensordot(b_vi[:, sl, :], b_ai, axes=([0], [0]))     # (nba, nc, na, nc)
        g = g - g.transpose(0, 3, 2, 1)
        x = np.ascontiguousarray(g.transpose(0, 1, 3, 2)).reshape(-1, na)   # rows (a, i, j)
        n, imag_n = _quadratic_form(x, hole)
        k, imag_k = _quadratic_form(x, koop)
        max_imag = max(max_imag, imag_n, imag_k)
        eps = (eps_v[sl][:, None, None] - eps_i[None, :, None]
               - eps_i[None, None, :]).ravel()
        norms[filled:filled + n.size] = n
        denominators[filled:filled + n.size] = eps + _safe_ratio(k, n)
        keys[filled:filled + n.size] = (
            gv[sl][:, None, None] * (ngi * ngi)
            + unordered_pair_key(gi_all[None, :, None], gi_all[None, None, :], ngi)).ravel()
        filled += n.size
    result = class_energy("Sijr", norms[:filled], denominators[:filled], ctx, prefactor=0.5,
                          groups=keys[:filled])
    return ClassResult(name="Sijr", norm=result.norm, energy=result.energy,
                       n_perturbers=result.n_perturbers, n_dropped=result.n_dropped,
                       min_denominator=result.min_denominator,
                       min_signed_denominator=result.min_signed_denominator,
                       max_imaginary=max_imag)


def _eval_srs(ctx: ClassContext) -> ClassResult:
    """``S_rs^(-2)``: ``t u -> a b``. Two virtual particles, ``dN_act = -2``. **Norm only.**

    Collecting the two-electron terms with two virtual creations and two active annihilations,
    and summing the ``(a,b)`` and ``(b,a)`` contributions,

    ::

        |Psi_{ab}> = a+_a a+_b sum_tu (at|bu) a_u a_t |Psi_0>
        N_ab       = sum_tuvw conj((at|bu)) (av|bw) Gamma_{t v u w}

    which is :func:`kuiva.pt.contractions.pair_matrix` as a quadratic form in the integral pair
    ``(t,u)``. Writing ``|D_tu> = a_u a_t |Psi_0>``, that matrix is the Gram matrix
    ``<D_tu|D_vw>`` and the denominator is the same Gram matrix with ``H_act - E`` between:

    ::

        dE_ab = eps_a + eps_b + K_ab / N_ab,
        K[(t,u),(v,w)] = <Psi| a+_t a+_u (H_act - E) a_w a_v |Psi> = <D_tu|(H_act - E)|D_vw>

    ⚠ **No 3-RDM is formed for this** — the vectors are built explicitly in the ``N-2``
    electron space and the active Hamiltonian is applied there
    (:meth:`kuiva.pt.contractions.CIContractionProvider.pair_koopmans`). That is a departure
    from the planning-level design, recorded in the provider's module docstring.
    """
    nv, na = ctx.n_virtual, ctx.n_active
    if nv < 2 or na < 2:
        return ClassResult(name="Srs", norm=0.0, energy=0.0)
    b_va = ctx.blocks.three_index("virtual", "active")
    overlap = ctx.provider.pair_matrix()
    koopmans = ctx.provider.pair_koopmans()
    eps_v = ctx.eps_virtual

    norms, denominators, keys = _label_buffers("Srs", nv * nv)
    gv, ngv = ctx.group_virtual, ctx.n_group_virtual
    max_imag = 0.0
    filled = 0
    per_a = batch_gb((na, nv, na), count=2)
    for sl in batch_slices(nv, per_a):
        v = np.tensordot(b_va[:, sl, :], b_va, axes=([0], [0]))     # (nba, na, nv, na)=[a,t,b,u]
        x = np.ascontiguousarray(v.transpose(0, 2, 1, 3)).reshape(-1, na * na)
        n, imag_n = _quadratic_form(x, overlap)
        k, imag_k = _quadratic_form(x, koopmans)
        max_imag = max(max_imag, imag_n, imag_k)
        eps = (eps_v[sl][:, None] + eps_v[None, :]).ravel()
        norms[filled:filled + n.size] = n
        denominators[filled:filled + n.size] = eps + _safe_ratio(k, n)
        keys[filled:filled + n.size] = unordered_pair_key(
            gv[sl][:, None], gv[None, :], ngv).ravel()
        filled += n.size
    result = class_energy("Srs", norms[:filled], denominators[:filled], ctx, prefactor=0.5,
                          groups=keys[:filled])
    return ClassResult(name="Srs", norm=result.norm, energy=result.energy,
                       n_perturbers=result.n_perturbers, n_dropped=result.n_dropped,
                       min_denominator=result.min_denominator,
                       min_signed_denominator=result.min_signed_denominator,
                       max_imaginary=max_imag)


def _eval_sij(ctx: ClassContext) -> ClassResult:
    """``S_ij^(+2)``: ``i j -> t u``. Two inactive holes, ``dN_act = +2``.

    The particle-hole mirror of ``Srs``. With ``|C_tu> = a+_t a+_u |Psi_0>`` in the ``N+2``
    electron space,

    ::

        |Psi_{ij}> = sum_tu (ti|uj) a+_t a+_u a_j a_i |Psi_0>
        N_ij       = sum_tuvw conj((ti|uj)) (vi|wj) <C_tu|C_vw>
        dE_ij      = -eps_i - eps_j + K_ij / N_ij,   K = <C_tu|(H_act - E)|C_vw>

    The overlap is :func:`kuiva.pt.contractions.hole_pair_matrix` — six ``delta`` terms and the
    2-RDM — and it is *also* the Gram matrix of the same vectors, so the two constructions
    check each other (they are asserted equal in the tests).
    """
    nc, na = ctx.n_inactive, ctx.n_active
    if nc < 2 or na < 2:
        return ClassResult(name="Sij", norm=0.0, energy=0.0)
    b_ai = ctx.blocks.three_index("active", "inactive")     # (naux, na, nc) = B^P_{t i}
    overlap = ctx.provider.hole_pair_matrix()
    koopmans = ctx.provider.hole_pair_koopmans()
    eps_i = ctx.eps_inactive

    norms, denominators, keys = _label_buffers("Sij", nc * nc)
    gi_all, ngi = ctx.group_inactive, ctx.n_group_inactive
    max_imag = 0.0
    filled = 0
    per_i = batch_gb((na, na, nc), count=2)
    for sl in batch_slices(nc, per_i):
        v = np.tensordot(b_ai[:, :, sl], b_ai, axes=([0], [0]))     # (na, nbi, na, nc)=[t,i,u,j]
        x = np.ascontiguousarray(v.transpose(1, 3, 0, 2)).reshape(-1, na * na)
        n, imag_n = _quadratic_form(x, overlap)
        k, imag_k = _quadratic_form(x, koopmans)
        max_imag = max(max_imag, imag_n, imag_k)
        eps = (-eps_i[sl][:, None] - eps_i[None, :]).ravel()
        norms[filled:filled + n.size] = n
        denominators[filled:filled + n.size] = eps + _safe_ratio(k, n)
        keys[filled:filled + n.size] = unordered_pair_key(
            gi_all[sl][:, None], gi_all[None, :], ngi).ravel()
        filled += n.size
    result = class_energy("Sij", norms[:filled], denominators[:filled], ctx, prefactor=0.5,
                          groups=keys[:filled])
    return ClassResult(name="Sij", norm=result.norm, energy=result.energy,
                       n_perturbers=result.n_perturbers, n_dropped=result.n_dropped,
                       min_denominator=result.min_denominator,
                       min_signed_denominator=result.min_signed_denominator,
                       max_imaginary=max_imag)


def _eval_sir(ctx: ClassContext) -> ClassResult:
    """``S_ir^(0')``: ``i -> a`` with active rearrangement. ``dN_act = 0``.

    The one class whose perturber mixes a one-body and a two-body term, and the messiest to
    derive. Collecting **every** part of ``H`` that makes one inactive hole and one virtual
    particle while leaving the active electron count alone — the bare one-electron element, the
    inactive-inactive two-electron terms, and the two active-active ones — and doubling for the
    equal contribution with the virtual creation in the second operator slot:

    ::

        |Psi_ia> = a+_a a_i [ f^I_ai + sum_tu g_tu E_tu ] |Psi_0>,
        g_tu     = (ai|tu) - (au|ti)

    ⚠ The one-body coefficient is the **inactive Fock element** ``f^I_ai``, not ``h_ai``: the
    core sum ``sum_k [(ai|kk) - (ak|ki)]`` is exactly what completes it, and the ``k = i`` term
    of that sum cancels identically, which is why the naive "skip the hole orbital" and the
    plain Fock element agree. It is generally **nonzero** even at a converged CASSCF — what
    convergence kills is the *generalized* Fock element ``(F^I + F^A)_ai``, not this one.

    With ``|X_tu> = E_tu|Psi_0>`` and the reference itself as a zeroth basis vector, both the
    norm and the denominator are quadratic forms in the augmented coefficient
    ``w = [f^I_ai, g]`` against the augmented Gram matrices of
    :meth:`kuiva.pt.contractions.CIContractionProvider.excitation_overlap` and
    ``excitation_koopmans``:

    ::

        N_ia  = w^dag S w,   dE_ia = eps_a - eps_i + (w^dag M w) / N_ia

    ⚠ The cross terms between the reference and ``E_tu|Psi_0>`` vanish in ``M`` and **not** in
    ``S`` — ``(H_act - E)|Psi_0> = 0`` kills the first row and column of one and not of the
    other. Dropping the augmentation and treating ``f^I_ai`` as a separate additive term is the
    natural-looking simplification that gets this wrong.

    ⚠ **This is the one class for which the group-complete contraction is not a no-op on a
    real system**, and the reason is visible in the formula above: for a same-spin label
    ``(i, a)`` the perturber carries ``f^I_ai`` and the direct integral ``(ai|tu)``, while for
    a spin-flip label it carries only the exchange term ``-(au|ti)``. Nothing relates the two,
    so their denominators differ — 2.9e-2 Eh on LiH/STO-3G — and a per-spinor contraction makes
    ``E2`` depend on the arbitrary unitary the canonicalization leaves inside the degenerate
    block. See the module docstring: this class is why that rule exists.
    """
    nv, nc, na = ctx.n_virtual, ctx.n_inactive, ctx.n_active
    if nv == 0 or nc == 0:
        return ClassResult(name="Sir", norm=0.0, energy=0.0)
    if ctx.fock_vi is None:
        raise ValueError("the Sir (0') class needs the inactive Fock block over "
                         "(virtual, inactive); the driver fills ClassContext.fock_vi")
    b_vi = ctx.blocks.three_index("virtual", "inactive")    # (naux, nv, nc)
    b_aa = ctx.blocks.three_index("active", "active")       # (naux, na, na)
    b_va = ctx.blocks.three_index("virtual", "active")      # (naux, nv, na)
    b_ai = ctx.blocks.three_index("active", "inactive")     # (naux, na, nc)
    overlap = ctx.provider.excitation_overlap()
    koopmans = ctx.provider.excitation_koopmans()
    eps_v, eps_i = ctx.eps_virtual, ctx.eps_inactive

    norms, denominators, keys = _label_buffers("Sir", nv * nc)
    gv, gi_all, ngi = ctx.group_virtual, ctx.group_inactive, ctx.n_group_inactive
    max_imag = 0.0
    filled = 0
    per_a = batch_gb((nc, na, na), count=4)
    for sl in batch_slices(nv, per_a):
        nba = len(range(*sl.indices(nv)))
        # (ai|tu) -> [a, i, t, u];  (au|ti) -> [a, u, t, i], transposed to match.
        g = np.tensordot(b_vi[:, sl, :], b_aa, axes=([0], [0]))     # (nba, nc, na, na)
        g -= np.tensordot(b_va[:, sl, :], b_ai,
                          axes=([0], [0])).transpose(0, 3, 2, 1)
        x = np.empty((nba, nc, na * na + 1), dtype=np.complex128)
        x[:, :, 0] = ctx.fock_vi[sl]
        x[:, :, 1:] = g.reshape(nba, nc, na * na)
        x = x.reshape(-1, na * na + 1)
        n, imag_n = _quadratic_form(x, overlap)
        k, imag_k = _quadratic_form(x, koopmans)
        max_imag = max(max_imag, imag_n, imag_k)
        eps = (eps_v[sl][:, None] - eps_i[None, :]).ravel()
        norms[filled:filled + n.size] = n
        denominators[filled:filled + n.size] = eps + _safe_ratio(k, n)
        keys[filled:filled + n.size] = (gv[sl][:, None] * ngi + gi_all[None, :]).ravel()
        filled += n.size
    result = class_energy("Sir", norms[:filled], denominators[:filled], ctx,
                          groups=keys[:filled])
    return ClassResult(name="Sir", norm=result.norm, energy=result.energy,
                       n_perturbers=result.n_perturbers, n_dropped=result.n_dropped,
                       min_denominator=result.min_denominator,
                       min_signed_denominator=result.min_signed_denominator,
                       max_imaginary=max_imag)


def _eval_sr(ctx: ClassContext) -> ClassResult:
    """``S_r^(-1')``: ``t -> a`` with active rearrangement. ``dN_act = -1``.

    One external label — a single virtual spinor — and therefore the class with the *fewest*
    perturbers and the *deepest* active-space content: everything except the one particle sits
    inside the active space and has to be resolved there.

    Collecting every part of ``H`` that creates one virtual particle, removes one active
    electron and leaves the inactive occupation alone gives

    ::

        |Psi_a> = a+_a [ sum_t f^I_at a_t + sum_tuv (at|uv) a+_u a_v a_t ] |Psi_0>

    ⚠ **The two-body coefficient is ``(at|uv)`` with no antisymmetrization and no factor of a
    half, and getting there needs one relabelling that is easy to skip.** The two ways the
    virtual creation can sit in ``a+_p a+_r a_s a_q`` give ``1/2 (at|uv)`` and
    ``-1/2 (ut|av)``; they are equal only after using the antisymmetry of the *annihilation*
    pair (``a_v a_t = -a_t a_v``) to relabel ``t <-> v`` in the second, whereupon 4-fold
    symmetry turns ``(uv|at)`` into ``(at|uv)``. Stopping at the first term alone halves the
    class and stopping at the raw sum leaves a term that is not even Hermitian in the pair.

    ⚠ **The one-body coefficient is the inactive Fock ``f^I_at``, not ``h_at``** — the core sum
    ``sum_k [(at|kk) - (ak|kt)]`` is what completes it, exactly as in ``Sir``, and here no core
    index coincides with an external label so nothing cancels. It is generally nonzero at a
    converged CASSCF: what convergence kills is the *generalized* Fock element
    ``(F^I + F^A)_at``, and the active mean field ``F^A`` is carried by the two-body term.

    Both halves are linear in one vector set, so the norm and the denominator are

    ::

        N_a  = <Psi_a|Psi_a>,   dE_a = eps_a + <Psi_a|(H_act - E)|Psi_a> / N_a

    ⚠ **evaluated with one vector per label rather than as a Gram matrix over the
    ``n_act^3`` ladder strings** — see :mod:`kuiva.pt.contractions` for the crossover and for
    why the ``n_act^6`` object the literature calls the contracted 4-RDM is never formed.
    """
    nv, na = ctx.n_virtual, ctx.n_active
    if nv == 0 or na == 0:
        return ClassResult(name="Sr", norm=0.0, energy=0.0)
    if ctx.fock_va is None:
        raise ValueError("the Sr (-1') class needs the inactive Fock block over "
                         "(virtual, active); the driver fills ClassContext.fock_va")
    ndet = ctx.provider.shifted_ndet(-1)
    if ndet == 0:
        # No N-1 electron space: every perturber of this class vanishes identically.
        return ClassResult(name="Sr", norm=0.0, energy=0.0)
    b_va = ctx.blocks.three_index("virtual", "active")      # (naux, nv, na) = B^P_{a t}
    b_aa = ctx.blocks.three_index("active", "active")       # (naux, na, na) = B^P_{u v}
    eps_v = ctx.eps_virtual

    norms, denominators, keys = _label_buffers("Sr", nv)
    gv = ctx.group_virtual
    max_imag = 0.0
    filled = 0
    per_a = batch_gb((na, na, na), count=2) + perturber_vector_gb(ndet)
    for sl in batch_slices(nv, per_a):
        w3 = np.tensordot(b_va[:, sl, :], b_aa, axes=([0], [0]))    # (nba, na, na, na)
        w1 = np.ascontiguousarray(ctx.fock_va[sl])
        n, k, imag = ctx.provider.annihilation_perturbers(w1, w3)
        max_imag = max(max_imag, imag)
        norms[filled:filled + n.size] = n
        denominators[filled:filled + n.size] = eps_v[sl] + _safe_ratio(k, n)
        keys[filled:filled + n.size] = gv[sl]
        filled += n.size
    result = class_energy("Sr", norms[:filled], denominators[:filled], ctx,
                          groups=keys[:filled])
    return ClassResult(name="Sr", norm=result.norm, energy=result.energy,
                       n_perturbers=result.n_perturbers, n_dropped=result.n_dropped,
                       min_denominator=result.min_denominator,
                       min_signed_denominator=result.min_signed_denominator,
                       max_imaginary=max_imag)


def _eval_si(ctx: ClassContext) -> ClassResult:
    """``S_i^(+1')``: ``i -> t`` with active rearrangement. ``dN_act = +1``.

    The particle-hole mirror of :func:`_eval_sr`, with one inactive hole as the external label:

    ::

        |Psi_i> = [ sum_t f^I_ti a+_t + sum_tuv (ti|uv) a+_t a+_u a_v ] a_i |Psi_0>
        N_i     = <Psi_i|Psi_i>,   dE_i = -eps_i + <Psi_i|(H_act - E)|Psi_i> / N_i

    ⚠ **The ``k = i`` term of the core sum inside ``f^I_ti`` cancels identically** — direct
    against exchange, ``(ti|ii) - (ti|ii)`` — so the plain inactive Fock element is right even
    though the orbital it refers to is the one being emptied. That is the same cancellation
    ``Sir`` relies on, and it is why neither class needs a "skip the hole orbital" special case.

    ⚠ **Moving the active string past the inactive hole flips the sign of every term equally**,
    because ``a+_t`` and ``a+_t a+_u a_v`` are both odd, so the global factor drops out of both
    ``N_i`` and ``dE_i``. A *mixed*-parity operator would not have that luxury, and the reason
    to state it is that the code never applies ``a_i`` at all — it works entirely in the active
    space, which is only legitimate because of this.
    """
    nc, na = ctx.n_inactive, ctx.n_active
    if nc == 0 or na == 0:
        return ClassResult(name="Si", norm=0.0, energy=0.0)
    if ctx.fock_ai is None:
        raise ValueError("the Si (+1') class needs the inactive Fock block over "
                         "(active, inactive); the driver fills ClassContext.fock_ai")
    ndet = ctx.provider.shifted_ndet(+1)
    if ndet == 0:
        # No N+1 electron space: a full active space has no Si perturbers (the null test).
        return ClassResult(name="Si", norm=0.0, energy=0.0)
    b_ai = ctx.blocks.three_index("active", "inactive")     # (naux, na, nc) = B^P_{t i}
    b_aa = ctx.blocks.three_index("active", "active")       # (naux, na, na) = B^P_{u v}
    eps_i = ctx.eps_inactive

    norms, denominators, keys = _label_buffers("Si", nc)
    gi_all = ctx.group_inactive
    max_imag = 0.0
    filled = 0
    per_i = batch_gb((na, na, na), count=2) + perturber_vector_gb(ndet)
    for sl in batch_slices(nc, per_i):
        # (ti|uv) with i batched -> [t, i, u, v]; the class wants the label index leading.
        v = np.tensordot(b_ai[:, :, sl], b_aa, axes=([0], [0]))     # (na, nbi, na, na)
        w3 = np.ascontiguousarray(v.transpose(1, 0, 2, 3))          # (nbi, na, na, na)
        w1 = np.ascontiguousarray(ctx.fock_ai[:, sl].T)
        n, k, imag = ctx.provider.creation_perturbers(w1, w3)
        max_imag = max(max_imag, imag)
        norms[filled:filled + n.size] = n
        denominators[filled:filled + n.size] = -eps_i[sl] + _safe_ratio(k, n)
        keys[filled:filled + n.size] = gi_all[sl]
        filled += n.size
    result = class_energy("Si", norms[:filled], denominators[:filled], ctx,
                          groups=keys[:filled])
    return ClassResult(name="Si", norm=result.norm, energy=result.energy,
                       n_perturbers=result.n_perturbers, n_dropped=result.n_dropped,
                       min_denominator=result.min_denominator,
                       min_signed_denominator=result.min_signed_denominator,
                       max_imaginary=max_imag)


# --- the registry ------------------------------------------------------------------------------

#: What a class can currently produce. ``"energy"`` contributes to ``E2``; ``"norm"`` reports a
#: norm and nothing else; ``"planned"`` cannot be evaluated at all. ⚠ Only ``"energy"`` classes
#: may be summed, and the driver says so in its output rather than quietly adding zeros.
CLASS_STATUS = ("energy", "norm", "planned")


@dataclass(frozen=True)
class ExcitationClass:
    """One registered class of the first-order interacting space."""

    name: str
    #: The literature label, used verbatim in the per-class table so it lines up with PySCF's.
    label: str
    #: Change in the active electron count.
    delta_n_act: int
    evaluate: Callable[[ClassContext], ClassResult]
    status: str = "energy"
    #: Contraction primitives the evaluation asks the provider for — declared so that a
    #: provider's capabilities can be checked before anything is built (the ``qc`` pattern).
    requires: Tuple[str, ...] = ()
    #: Whether the class reads nothing state-specific, so one evaluation serves every state.
    #: ⚠ **True only for ``Sijrs``, and only sound because the Dyall Fock is state-averaged**: its
    #: perturbers contain no active-space operator at all, so its norms and denominators are
    #: built from the integral blocks and the ``eps`` alone. The driver honours this only when
    #: ``fock="state-averaged"``, because a state-specific Fock moves the ``eps`` (and the
    #: canonical orbitals) between states and then nothing is shared.
    state_independent: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.status not in CLASS_STATUS:
            raise ValueError("class {!r} declares unknown status {!r}; expected one of {}"
                             .format(self.name, self.status, list(CLASS_STATUS)))

    def __repr__(self) -> str:
        return "ExcitationClass({}, dN_act={:+d}, status={})".format(
            self.label, self.delta_n_act, self.status)


_CLASSES: "Dict[str, ExcitationClass]" = {}


def register_class(cls: ExcitationClass) -> None:
    """Register (or replace) an excitation class."""
    if cls.name in _CLASSES:
        log.debug("replacing already-registered NEVPT2 class %r", cls.name)
    _CLASSES[cls.name] = cls


def excitation_class(name: str) -> ExcitationClass:
    """The class registered under ``name``; refuse, naming what exists."""
    if name not in _CLASSES:
        raise ValueError("unknown NEVPT2 excitation class {!r}; registered: {}"
                         .format(name, ", ".join(available_classes())))
    return _CLASSES[name]


def available_classes() -> Tuple[str, ...]:
    """Every registered class, in the canonical reporting order."""
    return tuple(_ORDER)


def implemented_classes() -> Tuple[str, ...]:
    """The classes that currently contribute an energy."""
    return tuple(n for n in _ORDER if _CLASSES[n].status == "energy")


#: Reporting order: the eight classes as Angeli lists them, which is also PySCF's print order.
_ORDER = ("Sr", "Si", "Sijrs", "Sijr", "Srsi", "Srs", "Sij", "Sir")


def _register_default_classes() -> None:
    register_class(ExcitationClass(
        name="Sijrs", label="Sijrs (0)  ", delta_n_act=0, evaluate=_eval_sijrs,
        status="energy", requires=(), state_independent=True,
        description="ij -> ab, the MP2-like bulk; no active-space content at all"))
    register_class(ExcitationClass(
        name="Srsi", label="Srsi  (-1) ", delta_n_act=-1, evaluate=_eval_srsi,
        status="energy", requires=("rdm1", "koopmans_annihilation"),
        description="i t -> a b; one active electron removed"))
    register_class(ExcitationClass(
        name="Sijr", label="Sijr  (+1) ", delta_n_act=+1, evaluate=_eval_sijr,
        status="energy", requires=("hole_rdm1", "koopmans_creation"),
        description="i j -> a t; one active electron added"))
    register_class(ExcitationClass(
        name="Srs", label="Srs   (-2) ", delta_n_act=-2, evaluate=_eval_srs,
        status="energy", requires=("pair_matrix", "pair_koopmans"),
        description="t u -> a b; two active electrons removed"))
    register_class(ExcitationClass(
        name="Sij", label="Sij   (+2) ", delta_n_act=+2, evaluate=_eval_sij,
        status="energy", requires=("hole_pair_matrix", "hole_pair_koopmans"),
        description="i j -> t u; two active electrons added"))
    register_class(ExcitationClass(
        name="Sir", label="Sir   (0') ", delta_n_act=0, evaluate=_eval_sir,
        status="energy", requires=("excitation_overlap", "excitation_koopmans"),
        description="i -> a with active rearrangement; the one mixed one-/two-body perturber"))
    register_class(ExcitationClass(
        name="Sr", label="Sr    (-1')", delta_n_act=-1, evaluate=_eval_sr,
        status="energy", requires=("annihilation_perturbers",),
        description="t -> a with active rearrangement; one perturber vector per virtual"))
    register_class(ExcitationClass(
        name="Si", label="Si    (+1')", delta_n_act=+1, evaluate=_eval_si,
        status="energy", requires=("creation_perturbers",),
        description="i -> t with active rearrangement; one perturber vector per core hole"))


_register_default_classes()


__all__ = ["ClassContext", "ClassResult", "ExcitationClass", "CLASS_STATUS",
           "DEFAULT_NORM_CUTOFF", "DENOMINATOR_DEGENERACY_RTOL",
           "available_classes", "class_energy", "denominator_groups", "excitation_class",
           "group_complete_mask", "implemented_classes", "label_buffer_gb",
           "register_class"]
