"""Carrying a converged orbital set from one basis set onto another.

Why this exists
---------------
A CASSCF is expensive in proportion to the basis it runs in, and almost all of that expense
buys the *orbitals*, not the active space. The standard practical route is therefore to
converge the calculation in a small (often minimal) basis, where the active orbitals are
easy to identify and the optimization is cheap, and to use that result as the starting guess
for the same calculation in the production basis. This module is the step in the middle: it
takes a spinor set expressed over one AO basis and returns the closest orthonormal set over
another, with the inactive / active / virtual partition intact, so the large-basis run starts
from active orbitals that are already the right ones.

The reverse direction — large basis onto small — is the same operation and is supported. It
is not merely the inverse: the small basis genuinely cannot represent everything the large
one could, so orbitals lose norm and the diagnostics below stop being decoration.

The projection itself
---------------------
An orbital ``|psi> = |AO(s)> c`` is carried onto the target space by the orthogonal projector
onto that space. Written with the target's **orthonormal working basis** ``|w> = |AO(t)> X``
(the object the rest of the program is built on) the projector is ``sum_i |w_i><w_i|``, so

    c_work = X^T <AO(t)|AO(s)> c                                                        (1)

and no linear system is solved at all. That is deliberate. The textbook form of the same
projection is ``c_t = S_tt^-1 <AO(t)|AO(s)> c``, which is what PySCF's ``project_mo_nr2nr``
and Q-Chem's ``OVPROJECTION`` evaluate; it is identical to (1) whenever ``S_tt`` is
invertible, and it is exactly the wrong thing to write down when it is not. Large
uncontracted and mixed relativistic bases on heavy elements are near-linearly dependent as a
matter of course — that is why the working basis exists — and inverting ``S_tt`` there
amplifies noise by the same ``s^-1/2`` the working basis was built to discard. Form (1)
inherits the linear-dependence removal instead of fighting it, for free.

Because the AO overlap is spin-diagonal, the two-component projector is
``1_2 (x) X^T <AO(t)|AO(s)>``: the same *real* matrix applied to the alpha and beta row
blocks of the spinor convention. Two consequences worth stating, because they are what make
this safe to apply to a relativistic orbital set:

* it commutes with time reversal ``T = -i sigma_y K`` (``K`` conjugates coefficients, and the
  projector is real and spin-independent), so **a Kramers-paired set projects to a
  Kramers-paired set** and a Kramers pair cannot be split by the projection;
* it is diagonal in nothing else, so orbital character, spin-orbit mixing and the complex
  phase structure of a two-component orbital are all carried across unchanged.

What crosses, and what does not (``carry``)
------------------------------------------
The first choice, and the one that matters more, is **which** orbitals are carried at all.

``"active"`` (the default)
    Only the active space crosses. The inactive and virtual orbitals are taken from the
    *target's own SCF*, orthogonalized against the carried active space and pseudo-canonical
    inside what is left, so the inactive orbitals are the target basis' own core.

``"all"``
    Every orbital crosses, and only the dimensions the target basis adds beyond the source
    are built in the target's terms.

⚠ **The default deliberately carries less, and it is measured rather than obvious.** The
source's inactive orbitals are not eigenvectors of anything in the target basis: carrying
them re-introduces an inactive-virtual gradient — and the core orbital energies are the
largest numbers in that block — which the target's own SCF had already removed. Across three
systems and both directions, ``"all"`` costs between nothing and twice the macro-iterations
and never fewer. What the small-basis calculation was *for* is the active space; the rest of
the orbitals the target's own SCF already has, better.

What projection destroys, and the three ways to repair it
---------------------------------------------------------
A projector is not unitary. ``c_work`` is **not orthonormal**: the projected set has Gram
matrix ``M = c_work^dag c_work``, which is the identity only if the source space is contained
in the target space. Restoring orthonormality is where the methods differ, and it is the only
real choice this module makes. All three keep the set inside the span of the projection; they
differ in what they are allowed to mix.

``"blocked"`` (the default)
    Space by space, in the order inactive, active, virtual: project the block off everything
    already finished, then orthonormalize *within* the block symmetrically (Loewdin). The
    inactive space is exactly the projected inactive space, the active space is exactly the
    projected active space with the inactive part removed, and so on. **The CAS partition
    survives exactly**, which is the property a CASSCF restart needs and the reason this is
    the default: a guess whose "active" orbitals have picked up inactive character is a guess
    for a different calculation. Within a block, Loewdin is the unique orthonormalization
    that minimises ``sum_p |phi'_p - phi_p|^2`` (Carlson & Keller 1957), so the block changes
    as little as orthonormality allows.

``"symmetric"``
    One Loewdin orthonormalization of the whole set. Minimal change overall — smaller, in the
    least-squares sense, than any blocked scheme — but it mixes the spaces, so an active
    orbital comes back with a little inactive and virtual character. Useful when the
    partition is not meaningful (projecting a plain SCF guess) and as the measurement the
    default is judged against.

``"gram-schmidt"``
    Modified Gram-Schmidt, column by column, in the same block order. Not minimal-change and
    it privileges the early columns, but it is the most forgiving of a badly conditioned
    (heavily truncating) projection, because it never inverts a Gram matrix. The fallback for
    a large-to-small projection that the other two refuse.

⚠ **All three preserve exact Kramers pairing** when the source has it, and this is structural
rather than incidental: ``M`` of a Kramers-paired set is *self-dual* (``J M* J^-1 = M`` with
``J`` the symplectic pair permutation), self-duality survives ``M -> M^-1/2``, and the
condition for a column transformation to map a paired set to a paired set is exactly
self-duality. Modified Gram-Schmidt preserves it for the more elementary reason that
``<u|Tu> = 0`` identically when ``T^2 = -1``, so the barred column is already orthogonal to
its own partner and comes out as ``T u`` untouched.

Completing the set
------------------
Every target column no source orbital was carried into — the inactive and virtual ones under
``carry="active"``, and under ``carry="all"`` the dimensions the larger basis adds beyond the
source — is built as the orthogonal complement of the carried columns, seeded by the target's
own SCF guess spinors, and then **pseudo-canonicalized**
inside the complement against the guess orbital-energy operator ``F = sum_p |g_p> eps_p
<g_p|``. That last step costs nothing and is what makes the completion an orbital set rather
than an arbitrary basis of a subspace: the new columns come out energy-ordered, with
energies, in the place an orbital of that energy belongs. The complement is handed to the
free positions in **ascending** energy, so under ``carry="active"`` the lowest of them land
in the inactive positions: that is the aufbau assignment, and it is what makes the inactive
space of a projected guess the target's own SCF core rather than whatever was left over.

⚠ ``F`` built from a Kramers-paired guess is time-reversal even, so restricted to a
time-reversal-closed complement its spectrum is **doubly degenerate** and ``eigh`` returns an
arbitrary basis of each degenerate pair — orthonormal, energy-ordered, and not Kramers
paired. The pairing is therefore rebuilt *inside each degenerate group of* ``F``, which
changes only a basis that was arbitrary and leaves the energy ordering exactly as it was.
Where the source is not Kramers paired at all — an unrestricted reference — the complement is
not time-reversal closed, the groups come out odd, and the completion is delivered unpaired,
which is the correct answer for an unrestricted orbital set.

⚠ **The carried columns have their Kramers pairs rebuilt, and the source is why.** The
projection preserves pairing exactly, but a converged general-complex CASSCF does not *have*
it: active-active rotations are redundant, so nothing in the optimization pushes back and the
converged active orbitals are an arbitrary unitary mixture of the pairs — the partner
deviation is routinely O(1) there, not the 1e-8..1e-6 band the *states* show. Every consumer
of the convention downstream needs pairs, so the carried blocks are rebuilt as explicit
``(u, T u)`` pairs before the complement is built against them. That costs nothing where it
matters: it is a rotation inside a span the optimizer left arbitrary, and it is applied only
to the carried blocks, so the completion keeps the energy ordering a later selection by
orbital character reads. Where the carried span is not time-reversal closed at all — an
unrestricted reference — the rebuild is impossible and ``repair_pairing="auto"`` says so and
carries on.

Going the other way, a smaller target basis has *fewer* orbitals than the source, and columns
must be dropped. The highest-index virtual ones are dropped — never an inactive or active
one, and a request that would need to is refused rather than reinterpreted.

What is measured, and why each number is here
---------------------------------------------
None of the failures of a basis projection announce themselves: every one of them produces an
orthonormal orbital set of the right shape that starts a calculation which converges to
something. The diagnostics are therefore not optional decoration.

* **Retained norm** ``||P psi||^2`` per source orbital. 1 means the target basis contains the
  orbital; anything less is the part of it that does not exist in the target basis. This is
  the number that says whether a large-to-small projection is meaningful at all.
* **Block conditioning**, the smallest eigenvalue of each block's Gram matrix after the
  preceding blocks are projected off. It is 1 for a faithful projection and approaches 0 when
  the projected block is becoming linearly dependent — at which point orthonormalizing it
  amplifies noise, exactly as an ill-conditioned overlap does. Below
  :data:`MIN_BLOCK_EIGENVALUE` the projection is **refused**.
* **Principal overlaps** between each source space and the space finally handed over —
  singular values of ``C_final^dag c_work`` restricted to the block, which are the cosines of
  the principal angles between the two subspaces and are invariant to how either set is
  rotated inside itself (the same discipline the cross-code property comparison uses: compare
  invariants, never coefficients). The smallest one is the honest single-number answer to
  "did the active space survive?", and it is what warns.

References
----------
* The projection of MO coefficients between basis sets, and the Fock/density-matrix
  alternative to it (Q-Chem's ``OVPROJECTION`` and ``FOPPROJECTION``): R. P. Steele,
  R. A. DiStasio Jr., Y. Shao, J. Kong, M. Head-Gordon, "Dual-basis second-order
  Moller-Plesset perturbation theory: A reduced-cost reference for correlation calculations",
  J. Chem. Phys. 125, 074108 (2006), doi:10.1063/1.2234371.
* Projected starting vectors from a smaller basis as the standard SCF guess: J. Almloef,
  K. Faegri, K. Korsell, "Principles for a direct SCF approach to LCAO-MO ab-initio
  calculations", J. Comput. Chem. 3, 385 (1982), doi:10.1002/jcc.540030314.
* The same operation for the *active* orbitals of a CASSCF, as a production workflow
  (OpenMolcas' ``EXPBAS``): I. Fdez. Galvan et al., "OpenMolcas: From Source Code to
  Insight", J. Chem. Theory Comput. 15, 5925 (2019), doi:10.1021/acs.jctc.9b00532; and
  F. Aquilante et al., "Modern quantum chemistry with [Open]Molcas", J. Chem. Phys. 152,
  214117 (2020), doi:10.1063/5.0004835.
* Symmetric (Loewdin) orthonormalization and its least-squares optimality: P.-O. Loewdin,
  J. Chem. Phys. 18, 365 (1950), doi:10.1063/1.1747632; B. C. Carlson, J. M. Keller,
  "Orthogonalization Procedures and the Localization of Wannier Functions", Phys. Rev. 105,
  102 (1957), doi:10.1103/PhysRev.105.102.
* Principal angles between orbital subspaces as the invariant comparison ("corresponding
  orbitals"): A. T. Amos, G. G. Hall, Proc. R. Soc. London A 263, 483 (1961),
  doi:10.1098/rspa.1961.0175.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..spinor.expand import nearest_kramers_paired, time_reverse
from ..util import output as out
from ..util.degeneracy import DEFAULT_GROUP_RTOL, relative_gap
from ..util.logging import get_logger
from ..util.timing import timer
from .canonical import OrthonormalBasis

log = get_logger(__name__)

#: The orthonormalization schemes, in the order they are documented above.
SCHEMES = ("blocked", "symmetric", "gram-schmidt")

#: The default: it is the only one that keeps the inactive / active / virtual partition
#: exact, which is what a CASSCF restart is asking for.
DEFAULT_SCHEME = "blocked"

#: What is carried across the basis change: the whole orbital set, or only the active space
#: with everything else taken from the target's own SCF orbitals.
CARRY = ("active", "all")

#: The default, and it is a **measured** one rather than an obvious one; see this package's
#: validation notes. Carrying the whole set starts at a lower energy and a much larger
#: gradient, because the source's inactive orbitals are not eigenvectors of anything in the
#: target basis and the inactive-virtual block of the Fock operator — where the core orbital
#: energies are — is no longer small. The active space is what the small-basis calculation
#: was for; the rest of the orbitals the target's own SCF already has, better.
DEFAULT_CARRY = "active"

#: Smallest Gram eigenvalue a block may have before the projection is refused. A block at
#: this point is numerically linearly dependent after projection: orthonormalizing it
#: multiplies whatever noise it carries by ``s^-1/2``, i.e. 1e5 here, which is the same
#: failure the working basis' linear-dependence threshold exists to prevent.
MIN_BLOCK_EIGENVALUE = 1.0e-10

#: Below this smallest principal overlap between a source space and the space actually handed
#: on, the projection warns. It is not a tolerance on anything physical — it says the guess is
#: no longer recognisably the calculation it was projected from.
FIDELITY_WARN = 0.90

#: Below this retained norm an individual inactive or active orbital is named in a warning:
#: the target basis does not contain it. Virtual orbitals are exempt (a virtual space is
#: reshaped by a change of basis as a matter of course).
RETAINED_WARN = 0.98

#: Relative eigenvalue gap grouping the pseudo-canonicalization's spectrum, for the pairing
#: rebuild inside the completion. The same value the rest of the project groups a spectrum by,
#: deliberately: two places that group differently are two conventions.
COMPLETION_GROUP_RTOL = DEFAULT_GROUP_RTOL


# --- column bookkeeping (pure integers; no arrays are touched) ------------------------------

@dataclass(frozen=True)
class ColumnPlan:
    """Which source columns survive, and where every space ends up in the target set.

    Separated from the projection itself because it needs no integrals: a caller that must
    validate an active space *eagerly* (the class API does) can resolve the target spaces at
    construction time and leave the expensive part to ``run()``.
    """

    #: Source columns kept, ascending. Equal to ``arange(n_source)`` unless the target basis
    #: is smaller and columns had to be dropped.
    keep: np.ndarray
    #: Target-set positions of the three spaces, in the target's own numbering.
    inactive: np.ndarray
    active: np.ndarray
    virtual: np.ndarray
    n_source: int
    n_target: int

    @property
    def n_kept(self) -> int:
        return int(self.keep.size)

    @property
    def n_completed(self) -> int:
        """Target orbitals with no source: the dimensions the larger basis adds."""
        return int(self.n_target - self.keep.size)

    @property
    def n_dropped(self) -> int:
        """Source orbitals with no room in the target: virtuals a smaller basis cannot hold."""
        return int(self.n_source - self.keep.size)


def plan_columns(inactive, active, virtual, n_target: int) -> ColumnPlan:
    """Map a source orbital partition onto a target set of ``n_target`` spinors.

    The three index arrays partition the source spinors (they are
    :class:`kuiva.mcscf.orbopt.OrbitalSpaces`' three fields, taken as plain arrays so that
    this module does not depend on the multireference layer).

    Growing (``n_target`` larger) appends the new dimensions to the **virtual** space.
    Shrinking drops the **highest-index virtual** source columns, and refuses rather than
    touch an inactive or active one: dropping those would silently change the calculation
    into a different one, which is the whole failure mode this module has to avoid.
    """
    inactive = np.asarray(inactive, dtype=int).ravel()
    active = np.asarray(active, dtype=int).ravel()
    virtual = np.asarray(virtual, dtype=int).ravel()
    n_source = int(inactive.size + active.size + virtual.size)
    n_target = int(n_target)
    allidx = np.concatenate([inactive, active, virtual])
    if np.unique(allidx).size != n_source or (n_source and int(allidx.max()) != n_source - 1):
        raise ValueError("the three spaces must partition the source spinors exactly; got "
                         "{} indices ({} unique)".format(n_source, np.unique(allidx).size))

    if n_target >= n_source:
        keep = np.arange(n_source, dtype=int)
        extra = np.arange(n_source, n_target, dtype=int)
        return ColumnPlan(keep=keep, inactive=inactive, active=active,
                          virtual=np.concatenate([virtual, extra]),
                          n_source=n_source, n_target=n_target)

    n_drop = n_source - n_target
    if virtual.size < n_drop:
        raise ValueError(
            "projecting onto a basis with {} spinors from one with {} needs {} orbital(s) "
            "dropped, and only {} of them are virtual; the inactive and active spaces are "
            "the calculation and are never dropped. Use a target basis with at least {} "
            "spinors (i.e. {} basis functions after linear-dependence removal)."
            .format(n_target, n_source, n_drop, virtual.size,
                    n_source - virtual.size, (n_source - virtual.size + 1) // 2))
    dropped = np.sort(virtual)[virtual.size - n_drop:]
    keep = np.setdiff1d(np.arange(n_source, dtype=int), dropped, assume_unique=True)
    # Positions renumber only for columns after a dropped one; searchsorted is that map.
    remap = np.searchsorted(keep, np.arange(n_source, dtype=int))
    kept_virtual = np.sort(virtual)[:virtual.size - n_drop]
    return ColumnPlan(keep=keep, inactive=remap[inactive], active=remap[active],
                      virtual=remap[kept_virtual], n_source=n_source, n_target=n_target)


# --- orthonormalization schemes -------------------------------------------------------------

def _lowdin(b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Symmetric (Loewdin) orthonormalization ``b (b^dag b)^-1/2``, and the Gram floor.

    ⚠ Self-duality, and therefore Kramers pairing, is preserved exactly by this: it is a
    function of the Gram matrix alone, and ``J M* J^-1 = M`` implies the same for any
    ``f(M)``. Nothing here has to know that — it is why nothing here has to do anything.
    """
    m = b.conj().T @ b
    m = 0.5 * (m + m.conj().T)
    w, v = np.linalg.eigh(m)
    floor = float(w[0]) if w.size else 1.0
    if floor <= MIN_BLOCK_EIGENVALUE:
        return b, floor
    return b @ ((v * w ** -0.5) @ v.conj().T), floor


def _modified_gram_schmidt(b: np.ndarray) -> Tuple[np.ndarray, float]:
    """Column-by-column MGS. Returns the set and the smallest surviving squared norm.

    ⚠ Exact Kramers pairing survives this for an elementary reason: ``<u|Tu> = 0`` whenever
    ``T^2 = -1``, so once column ``2p`` has come out as ``u`` the barred column arrives as
    ``T u`` already orthogonal to it and is passed through untouched.
    """
    q = np.array(b, copy=True)
    floor = 1.0
    for j in range(q.shape[1]):
        col = q[:, j]
        if j:
            done = q[:, :j]
            col -= done @ (done.conj().T @ col)
            col -= done @ (done.conj().T @ col)          # one reorthogonalization; classic
        norm2 = float(np.vdot(col, col).real)
        floor = min(floor, norm2)
        if norm2 <= MIN_BLOCK_EIGENVALUE:
            continue
        q[:, j] = col / np.sqrt(norm2)
    return q, floor


def _orthonormalize(work: np.ndarray, blocks: Sequence[Tuple[str, np.ndarray]],
                    scheme: str) -> Dict[str, float]:
    """Orthonormalize ``work`` in place over the named blocks; return each block's floor."""
    floors: Dict[str, float] = {}
    if scheme == "symmetric":
        cols = np.concatenate([idx for _n, idx in blocks]) if blocks else np.zeros(0, int)
        if cols.size:
            work[:, cols], floors["all"] = _lowdin(work[:, cols])
        return floors

    finished: list = []
    for name, idx in blocks:
        if idx.size == 0:
            continue
        b = np.ascontiguousarray(work[:, idx])
        if finished:
            q = np.concatenate(finished, axis=1)
            b -= q @ (q.conj().T @ b)
            b -= q @ (q.conj().T @ b)
        b, floors[name] = (_modified_gram_schmidt(b) if scheme == "gram-schmidt"
                           else _lowdin(b))
        work[:, idx] = b
        finished.append(b)
    return floors


# --- the result -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BasisProjection:
    """A projected orbital set and everything needed to judge it.

    ``coeff`` is in the **target AO basis**, which is what every driver takes as ``coeff=``.
    """

    coeff: np.ndarray
    #: The column bookkeeping — the target spaces are :attr:`ColumnPlan.inactive` etc.
    plan: ColumnPlan
    #: ``||P psi||^2`` per *kept* source orbital, in target column order.
    retained: np.ndarray
    #: Smallest Gram eigenvalue per orthonormalized block (``"all"`` for ``"symmetric"``).
    block_floor: Dict[str, float]
    #: Principal overlaps (cosines of the principal angles) between each source space and the
    #: space finally handed on, ascending. Invariant to any rotation inside either space.
    overlaps: Dict[str, np.ndarray]
    scheme: str
    #: Which orbitals crossed the basis change (:data:`CARRY`).
    carry: str = DEFAULT_CARRY
    #: How many target columns came from the source; the rest were built in the target's own
    #: terms from ``complete_with``.
    n_carried: int = 0
    #: Gap between the smallest kept and largest discarded singular value of the complement
    #: construction; ``None`` when nothing had to be completed. A clean separation is the
    #: evidence that the complement really is the complement.
    complement_gap: Optional[Tuple[float, float]] = None
    #: Kramers partner deviation of the source set, and of the delivered set.
    partner_deviation: Tuple[float, float] = (0.0, 0.0)
    #: Whether the delivered set was repaired to exact Kramers pairing.
    repaired: bool = False

    @property
    def fidelity(self) -> float:
        """Smallest principal overlap over the inactive and active spaces.

        The single number to look at. 1 means the target basis reproduces the reference
        exactly; the guess is worth what this is.
        """
        vals = [float(v.min()) for k, v in self.overlaps.items()
                if k in ("inactive", "active") and v.size]
        return min(vals) if vals else 1.0

    def report(self, logger=None) -> None:
        """Log the standard projection block."""
        logger = logger or log
        plan = self.plan
        rows = [
            ("source / target spinors", "{} -> {}".format(plan.n_source, plan.n_target)),
            ("carried across", self.carry, "",
             "the active space only" if self.carry == "active" else "every orbital"),
            ("orthonormalization", self.scheme),
            ("orbitals projected", self.n_carried),
        ]
        if plan.n_target - self.n_carried:
            rows.append(("orbitals taken from the target's own SCF",
                         plan.n_target - self.n_carried))
        if plan.n_dropped:
            rows.append(("virtual orbitals dropped", plan.n_dropped, "",
                         "the target basis is smaller"))
        if self.retained.size:
            rows.append(("retained norm, all source orbitals (min / mean)",
                         "{:.6f} / {:.6f}".format(float(self.retained.min()),
                                                  float(self.retained.mean()))))
        for name in ("inactive", "active", "virtual"):
            vals = self.overlaps.get(name)  # carried spaces only
            if vals is not None and vals.size:
                rows.append(("{} space overlap (min / mean)".format(name),
                             "{:.6f} / {:.6f}".format(float(vals.min()),
                                                      float(vals.mean()))))
        for name, floor in self.block_floor.items():
            rows.append(("{} Gram floor".format(name), floor, "", "", "{:.2e}"))
        if self.complement_gap is not None:
            rows.append(("complement separation (kept / discarded)",
                         "{:.2e} / {:.2e}".format(*self.complement_gap)))
        rows.append(("Kramers pairing deviation (before / after)",
                     "{:.2e} / {:.2e}".format(*self.partner_deviation), "",
                     "pairs rebuilt" if self.repaired else "left as projected"))
        out.entries(logger, rows)


# --- the projection ---------------------------------------------------------------------------

def _partner_deviation(c: np.ndarray) -> float:
    """``max_p (1 - |<c_2p+1 | T c_2p>|)`` — phase-invariant, over an orthonormal basis."""
    if c.shape[1] < 2:
        return 0.0
    npair = c.shape[1] // 2
    tc = time_reverse(c[:, :2 * npair:2])
    ov = np.abs(np.sum(np.conj(c[:, 1:2 * npair:2]) * tc, axis=0))
    return float(np.max(np.abs(1.0 - ov)))


def project_spinors(c_source: np.ndarray, s_cross: np.ndarray, target: OrthonormalBasis, *,
                    inactive=None, active=None, virtual=None,
                    complete_with: Optional[np.ndarray] = None,
                    complete_energy: Optional[np.ndarray] = None,
                    carry: str = DEFAULT_CARRY, scheme: str = DEFAULT_SCHEME,
                    repair_pairing="auto",
                    report: bool = True) -> BasisProjection:
    """Project a spinor set from the source AO basis onto the target one.

    Parameters
    ----------
    c_source : ndarray ``(2*nao_source, n_source)`` complex
        The set to carry over, in the **source AO basis**, spin-blocked rows and interleaved
        Kramers columns (the project's spinor convention). It must be orthonormal in the
        source overlap; the retained norms are meaningless otherwise and will say so.
    s_cross : ndarray ``(nao_target, nao_source)`` real
        ``<AO(target)|AO(source)>``, from
        :func:`kuiva.interface.pyscf_bridge.cross_overlap`.
    target : :class:`~kuiva.orth.canonical.OrthonormalBasis`
        The target's working basis. Its linear-dependence removal is inherited by the
        projection — that is the reason the projector is written through ``X`` rather than
        through ``S_tt^-1``.
    inactive, active, virtual : index arrays, optional
        The source orbital partition. Given, the blocked schemes keep it exact and the
        diagnostics are reported per space; omitted, the whole set is one block and is
        assumed to be ordered occupied-first for the purpose of dropping columns.
    complete_with : ndarray ``(2*nao_target, m)``, optional
        Target-basis spinors (AO basis) seeding the orthogonal complement when the target has
        more orbitals than the source — normally the target reference's own guess. Without
        it the complement is built from the working basis itself, which is orthonormal and
        correct but carries no energy ordering.
    complete_energy : ndarray ``(m,)``, optional
        Orbital energies of ``complete_with``. Given, the completion is pseudo-canonicalized
        against ``F = sum_p |g_p> eps_p <g_p|`` inside the complement, so the new columns come
        out energy-ordered instead of in whatever order the complement construction produced.
    carry : str
        What crosses the basis change. ``"active"`` (**the default**) carries the active
        space and takes the inactive and virtual orbitals from ``complete_with``, i.e. from
        the target's own SCF; ``"all"`` carries every orbital and completes only the
        dimensions the target basis adds. See the module docstring for why the default is
        the one that carries *less*.
    scheme : str
        One of :data:`SCHEMES`; see the module docstring. ``"blocked"`` by default because it
        is the only one that leaves the CAS partition exact. With ``carry="active"`` there is
        one carried block and the three coincide.
    repair_pairing : bool or ``"auto"``
        Rebuild the *carried* blocks as explicit Kramers pairs
        (:func:`kuiva.spinor.expand.nearest_kramers_paired`), before the complement is built
        against them. ``"auto"`` (default) does it and degrades to a warning when the carried
        span is not time-reversal closed — which is the honest answer for an unrestricted
        reference, whose orbitals are not Kramers pairs at all. ``True`` refuses instead;
        ``False`` never tries. The completion columns are built paired regardless.

    Returns
    -------
    :class:`BasisProjection`, whose ``coeff`` is in the **target AO basis**.
    """
    if scheme not in SCHEMES:
        raise ValueError("scheme must be one of {}; got {!r}".format(SCHEMES, scheme))
    if carry not in CARRY:
        raise ValueError("carry must be one of {}; got {!r}".format(CARRY, carry))
    c_source = np.ascontiguousarray(c_source, dtype=np.complex128)
    s_cross = np.ascontiguousarray(s_cross, dtype=float)
    nao_t, nao_s = s_cross.shape
    if c_source.shape[0] != 2 * nao_s:
        raise ValueError("the orbital set has {} rows for a {}-function source basis; a "
                         "spinor set has 2*nao rows (alpha block then beta block)"
                         .format(c_source.shape[0], nao_s))
    if target.nao != nao_t:
        raise ValueError("the target working basis is built on {} AO functions and the cross "
                         "overlap has {} target rows".format(target.nao, nao_t))
    n_source = int(c_source.shape[1])
    n_target = 2 * int(target.nwork)

    if inactive is None and active is None and virtual is None:
        inactive = np.zeros(0, dtype=int)
        active = np.zeros(0, dtype=int)
        virtual = np.arange(n_source, dtype=int)
    plan = plan_columns(inactive, active, virtual, n_target)
    if plan.n_source != n_source:
        raise ValueError("the orbital partition covers {} spinors and the coefficient array "
                         "has {} columns".format(plan.n_source, n_source))

    with timer("basis projection"):
        # (1) the projector, applied to each spin block: 1_2 (x) X^T <AO(t)|AO(s)>.
        p = target.x.T @ s_cross                                    # (nwork, nao_source)
        nw = p.shape[0]
        raw = np.empty((2 * nw, n_source), dtype=np.complex128)
        raw[:nw] = p @ c_source[:nao_s]
        raw[nw:] = p @ c_source[nao_s:]
        retained_all = np.einsum("ij,ij->j", raw.conj(), raw).real

        blocks = [(name, np.asarray(idx, dtype=int))
                  for name, idx in (("inactive", plan.inactive), ("active", plan.active),
                                    ("virtual", plan.virtual))]
        # Which target columns come from the source, and which are built in the target's own
        # terms. ``carry="active"`` leaves the inactive and virtual positions free, so the
        # completion — pseudo-canonicalized and assigned in ascending energy — supplies a core
        # that IS an eigenvector of the target's own Fock, which is the whole point of it.
        if carry == "all":
            carried = [(name, idx[idx < plan.n_kept]) for name, idx in blocks]
        elif plan.active.size:
            carried = [("active", np.asarray(plan.active, dtype=int))]
        else:
            # ⚠ Otherwise this would carry nothing at all and hand back the target's own
            # guess, silently: an orthonormal set of the right shape that starts a
            # calculation which converges, and no orbital of the source in it anywhere.
            raise ValueError(
                "carry='active' has no active space to carry: no orbital partition was "
                "given, so every column of the source is virtual. Pass the source's "
                "inactive/active/virtual partition, or carry='all' to project the whole "
                "orbital set.")
        held = np.concatenate([idx for _n, idx in carried]) if carried else np.zeros(0, int)
        free = np.setdiff1d(np.arange(n_target, dtype=int), held, assume_unique=False)

        work = np.zeros((2 * nw, n_target), dtype=np.complex128)
        work[:, held] = raw[:, plan.keep[held]]

        floors = _orthonormalize(work, carried, scheme)
        bad = {k: v for k, v in floors.items() if v <= MIN_BLOCK_EIGENVALUE}
        if bad:
            raise ValueError(
                "the projected {} space is linearly dependent in the target basis (smallest "
                "Gram eigenvalue {:.2e}, floor {:.0e}): the target basis cannot hold this "
                "orbital set, and orthonormalizing it would amplify noise by its inverse "
                "square root. Project onto a larger basis, or use scheme='gram-schmidt', "
                "which does not invert the Gram matrix."
                .format("/".join(sorted(bad)), min(bad.values()), MIN_BLOCK_EIGENVALUE))

        # (2) Kramers pairing of the carried blocks — done HERE, before the complement is
        # built, so that the complement is the complement of the columns actually delivered.
        # ⚠ A converged general-complex CASSCF is entitled to leave its active orbitals far
        # from pair-aligned: active-active rotations are redundant, so nothing in the
        # optimization pushes back, and the deviation is routinely O(1) rather than the
        # 1e-8..1e-6 band the *states* show. Rebuilding the pairs costs nothing there (it is
        # a rotation inside a span the optimizer left arbitrary) and every consumer of the
        # pairing convention downstream — a contiguous pair-aligned active space, the
        # state-averaging gate, the Kramers-restricted CI — needs it.
        dev_before = _partner_deviation(work[:, held]) if held.size else 0.0
        repaired = False
        if repair_pairing and held.size:
            try:
                work = nearest_kramers_paired(work, [idx for _n, idx in carried])
                repaired = True
            except ValueError as exc:
                if repair_pairing is True:
                    raise
                log.warning("the carried orbitals were left unpaired: %s. That is the "
                            "expected answer for an unrestricted reference, whose spinors "
                            "are not Kramers pairs; downstream, a contiguous pair-aligned "
                            "active space is then not available either", exc)

        # (3) every position no source orbital was carried into, as the orthogonal
        # complement of the carried ones.
        complement_gap = None
        if free.size:
            complement_gap = _complete(work, held, free, complete_with, complete_energy,
                                       target)

        # (4) principal overlaps: an invariant statement about each carried space. Taken
        # after the repair, which is a rotation inside a time-reversal-closed span and
        # therefore moves no span — so this measures the projection and not the repair.
        overlaps: Dict[str, np.ndarray] = {}
        for name, idx in carried:
            if idx.size:
                overlaps[name] = np.sort(np.linalg.svd(
                    work[:, idx].conj().T @ raw[:, plan.keep[idx]], compute_uv=False))
        dev_after = _partner_deviation(work)

        # (5) back to the target AO basis, where every driver takes its orbitals.
        coeff = np.empty((2 * nao_t, n_target), dtype=np.complex128)
        coeff[:nao_t] = target.x @ work[:nw]
        coeff[nao_t:] = target.x @ work[nw:]

    result = BasisProjection(coeff=np.ascontiguousarray(coeff), plan=plan,
                             retained=retained_all[plan.keep], block_floor=floors,
                             overlaps=overlaps, scheme=scheme, carry=carry,
                             n_carried=int(held.size), complement_gap=complement_gap,
                             partner_deviation=(dev_before, dev_after),
                             repaired=bool(repaired))
    if report:
        result.report()
    _warn(result, carried, retained_all, plan)
    return result


def _degenerate_groups(values: np.ndarray, rtol: float = COMPLETION_GROUP_RTOL):
    """Consecutive index runs of ``values`` (ascending) with no relative gap above ``rtol``.

    The same grouping rule the rest of the project cuts a spectrum by
    (:mod:`kuiva.util.degeneracy`), reused here for a different purpose: not to *cut* a
    spectrum but to find the subspaces inside which a basis is arbitrary and may therefore be
    rebuilt in Kramers pairs at no cost.
    """
    groups, start = [], 0
    for i in range(1, int(values.size)):
        # ⚠ relative_gap is signed and expects the larger value first; these are ascending.
        if relative_gap(float(values[i]), float(values[i - 1])) > rtol:
            groups.append(np.arange(start, i, dtype=int))
            start = i
    if values.size:
        groups.append(np.arange(start, int(values.size), dtype=int))
    return groups


def _complete(work: np.ndarray, held: np.ndarray, free: np.ndarray, complete_with,
              complete_energy, target: OrthonormalBasis):
    """Fill the ``free`` columns of ``work`` with the complement of the ``held`` ones.

    In place. ``free`` is ascending and the complement is delivered in ascending
    pseudo-canonical energy, so the lowest orbitals land in the lowest free positions — which
    is the aufbau assignment, and is what makes ``carry="active"`` give the target's own SCF
    core rather than an arbitrary basis of the space that is left over.

    Returns the ``(smallest kept, largest discarded)`` singular value of the complement
    construction — the evidence that what was kept really is the complement and not a
    numerically ambiguous slice of it.
    """
    nw = work.shape[0] // 2
    n_new = int(free.size)
    q = work[:, held]
    if complete_with is None:
        seed = np.eye(2 * nw, dtype=np.complex128)
        energy = None
    else:
        g = np.ascontiguousarray(complete_with, dtype=np.complex128)
        nao_t = target.nao
        if g.shape[0] != 2 * nao_t:
            raise ValueError("complete_with has {} rows for a {}-function target basis"
                             .format(g.shape[0], nao_t))
        seed = np.empty((2 * nw, g.shape[1]), dtype=np.complex128)
        seed[:nw] = target.x_dag @ g[:nao_t]
        seed[nw:] = target.x_dag @ g[nao_t:]
        energy = None if complete_energy is None else np.asarray(complete_energy, dtype=float)
        if energy is not None and energy.size != seed.shape[1]:
            raise ValueError("complete_energy has {} entries for {} seed spinors"
                             .format(energy.size, seed.shape[1]))

    r = seed - q @ (q.conj().T @ seed)
    u, sv, _vt = np.linalg.svd(r, full_matrices=False)
    if sv.size < n_new or sv[n_new - 1] <= 1e-6:
        raise ValueError(
            "the completion set spans only {} of the {} dimensions the target basis adds "
            "beyond the source; pass complete_with= a set covering the whole target space "
            "(the target reference's own guess spinors do)"
            .format(int(np.count_nonzero(sv > 1e-6)), n_new))
    gap = (float(sv[n_new - 1]), float(sv[n_new]) if sv.size > n_new else 0.0)
    new = np.ascontiguousarray(u[:, :n_new])

    # Pseudo-canonicalize inside the complement against the guess energy operator
    # F = sum_p |g_p> eps_p <g_p|, so the added columns come out as an orbital set
    # (energy-ordered, in the place a virtual of that energy belongs) rather than as an
    # arbitrary basis of a subspace. eigh's eigenvalues are ascending, which is the order
    # a virtual space is read in.
    groups = None
    if energy is not None:
        gq = seed.conj().T @ new                                    # (m, n_new)
        f = (gq.conj().T * energy) @ gq
        f = 0.5 * (f + f.conj().T)
        w, v = np.linalg.eigh(f)
        new = new @ v
        groups = _degenerate_groups(w)

    new = new - q @ (q.conj().T @ new)                              # numerical guard
    new, _floor = _lowdin(new)

    # ⚠ F is time-reversal even, so every group above is an even-dimensional subspace whose
    # basis eigh returned arbitrarily. Rebuilding it as explicit Kramers pairs costs nothing
    # and changes nothing that was determined. An odd group means the complement is not
    # time-reversal closed — an unrestricted source — and the completion is then left as it
    # is, which is the right answer for an orbital set that is not Kramers paired.
    if groups is None:
        groups = [np.arange(n_new, dtype=int)]
    if all(g.size % 2 == 0 for g in groups):
        try:
            new = nearest_kramers_paired(new, groups)
        except ValueError as exc:                    # not time-reversal closed after all
            log.debug("the completed virtual columns were left unpaired (%s)", exc)
    else:
        log.debug("the completion spectrum has odd degenerate group(s); the added columns "
                  "are delivered unpaired (an unrestricted source is the normal cause)")
    work[:, free] = new
    return gap


def _warn(result: BasisProjection, carried, retained_all: np.ndarray,
          plan: ColumnPlan) -> None:
    """The things a user must be told about a projection, at WARNING."""
    for name, src in carried:
        if name == "virtual" or not src.size:
            continue
        lost = retained_all[plan.keep[src]]
        low = src[lost < RETAINED_WARN]
        if low.size:
            worst = int(low[int(np.argmin(lost[lost < RETAINED_WARN]))])
            log.warning("%d %s orbital(s) keep less than %.3f of their norm in the target "
                        "basis (worst: spinor %d at %.4f); the target basis does not contain "
                        "them, so this guess describes a different orbital space than the "
                        "calculation it came from", int(low.size), name, RETAINED_WARN,
                        worst, float(retained_all[plan.keep[worst]]))
    if result.fidelity < FIDELITY_WARN:
        log.warning("the projected inactive/active space overlaps the one it came from by "
                    "only %.4f (smallest principal overlap); this is a guess for a "
                    "recognisably different calculation, and the active space should be "
                    "re-inspected rather than trusted", result.fidelity)
    if plan.n_dropped:
        log.warning("%d virtual orbital(s) were dropped: the target basis holds %d spinors "
                    "against the source's %d. Projecting onto a smaller basis discards a "
                    "variational space rather than reproducing it.",
                    plan.n_dropped, plan.n_target, plan.n_source)


__all__ = ["BasisProjection", "ColumnPlan", "plan_columns", "project_spinors",
           "SCHEMES", "DEFAULT_SCHEME", "CARRY", "DEFAULT_CARRY", "MIN_BLOCK_EIGENVALUE",
           "FIDELITY_WARN", "RETAINED_WARN", "COMPLETION_GROUP_RTOL"]
