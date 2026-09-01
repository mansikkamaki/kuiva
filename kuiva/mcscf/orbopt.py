"""Shared state-averaged orbital optimizer.

**The contract is the point of this module.** One optimizer, consumed identically by the cheap
CI, by conventional-CI CASSCF and by DMRG-CASSCF: *reduced density
matrices in, orbital rotation out*. The CI method enters only as a callback
(:func:`optimize_orbitals`), so nothing here knows or cares whether the RDMs came from a
truncated determinant list, a full CI or a matrix-product state. Never bypass it.

The energy and its gradient
---------------------------
With inactive (fully occupied) spinors ``i``, active ``t``, virtual ``a``::

    F^I_pq = h_pq + sum_i [(pq|ii) - (pi|iq)]                     inactive Fock
    F^A_pq = sum_tu gamma_tu [(pq|tu) - (pu|tq)]                  active Fock
    E      = E_nuc + 1/2 sum_i (h_ii + F^I_ii)
             + sum_tu F^I_tu gamma_tu + 1/2 sum_tuvw (tu|vw) Gamma_tuvw

Orbitals rotate as ``C -> C exp(kappa)`` with ``kappa`` **anti-Hermitian**. Expanding the
energy to first order gives ``dE = 2 Re sum_{p>q} G_pq kappa_pq`` with the anti-Hermitian
gradient matrix ``G = F^dag - F`` built from the generalized Fock

    F_pq = sum_r D_pr h_qr + sum_rst Gamma_prst (qr|st)

which specializes (derivation in the tests, verified against the general form *and* against
finite differences) to

    F_iq = (F^I + F^A)_qi,     F_tq = sum_u gamma_tu F^I_qu + sum_uvw Gamma_tuvw (qu|vw),
    F_aq = 0.

Everything is complex: ``kappa`` has independent real and imaginary parts, and the imaginary
part is not decoration — it is what lets the optimizer build spinors that mix the Kramers
partners, i.e. what makes the wavefunction respond to spin-orbit coupling at all.

Why the Fock matrices are built in the AO basis
-----------------------------------------------
``F^I`` and ``F^A`` are ordinary Coulomb/exchange builds from a density, so they are formed by
:func:`kuiva.integrals.transform.coulomb_exchange` in the **AO** basis and transformed to MO
afterwards. The alternative — a full ``(naux, n, n)`` MO three-index transform — costs
``naux * n^2`` storage against ``nao^2`` here, gigabytes against megabytes for a CASSCF over
all spinors, and it is repeated every macro-iteration. The only three-index quantity that must
be transformed is ``B^P_{p,t}`` with **one active index** (:attr:`CASIntegrals.b_act`), which
is ``n_active/n`` of the full array. This is the single most important performance decision in
the module.

Three step types, one optimizer
------------------------------
``OrbitalOptimizer(mode=...)`` selects how the step is built. All three share the exact
gradient, the trust region and the accept/reject logic; they differ only in the curvature
model:

* ``"quasi-newton"`` — self-scaled L-BFGS preconditioned by an approximate diagonal Hessian.
  No Hessian-vector products at all. **Cheapest by far where it works**: measured at 3-4x
  less total work than the second-order step on problems that converge.
* ``"second-order"`` — the exact orbital Hessian (:class:`OrbitalHessian`) through an
  augmented-Hessian eigenproblem (:func:`augmented_hessian_step`). Converges cases where the
  quasi-Newton step does not converge at all, which is not a speed difference but the
  difference between an answer and none.
* ``"auto"`` (default) — start cheap, escalate when the **gradient trajectory** shows the
  cheap step is not getting anywhere. The robust choice, not the cheapest one.

Keeping the orbitals Kramers paired
-----------------------------------
``kappa`` is complex and anti-Hermitian, and nothing in that holds the orbital *spaces* closed
under time reversal. For a time-reversal-symmetric Hamiltonian and an ensemble its symmetry
leaves invariant, ``E(kappa) = E(Theta kappa)`` identically, so the exact gradient is
time-reversal **even** and the exact step would preserve closure — but the quasi-Newton and
augmented-Hessian steps carry curvature along directions the energy barely resists, and the
roundoff-level asymmetry they inject there is **amplified rather than damped**. Measured on a
UF3 CAS(3, 14 spinors) SA-10 reference before the constraint below existed: the relative
time-reversal breach of the active integrals grew 7e-21 -> 5e-7 over thirteen solves, roughly
a factor of ten every few iterations; the Kramers splitting of the odd-electron spectrum
tracked it exactly (0 -> 1.6e-6 -> 0.13 cm^-1); and the state-averaging gate then refused
inside a *trial* evaluation, which ended the optimization.

⚠ **An odd electron count is how that drift is DETECTED, not who it harms.** A rotation
*within* the active space cannot move a CI eigenvalue, so a split Kramers pair proves the
**subspace** drifted, not merely the alignment inside it — and an even-electron run drifts
identically with no enforced degeneracy for anything to notice (an N2 CAS(6,8) converged with
a 2.6e-04 closure defect and said nothing at all). Kramers' theorem is the tripwire, not the
victim.

The remedy is a constraint on the **step** and not a repair of the orbitals after it
(``kramers_rotation=``, resolved by :func:`resolve_kramers_rotation`, applied by
:meth:`OrbitalOptimizer._project`). Time reversal acts on a spinor *index* as ``pbar = p ^ 1``
with the sign ``t_p = (-1)^p``, and ``exp`` preserves the relation

    kappa_pq = t_p t_q conj(kappa[pbar, qbar])

because the map is multiplicative and the exponential series is real — so a ``kappa`` obeying
it generates a rotation that takes Kramers pairs to Kramers pairs *exactly*, and the spans
cannot leave the closed manifold at all. Imposing it is an **orthogonal projection of the real
parameter space** (:func:`kramers_project`), so what is optimized is the restriction of the
same problem rather than a modified one, and what the projection removes is the roundoff the
Fock builds put into the odd directions — the exact gradient has none there.

⚠ **It is a constraint, and it carries the irrep mask's caveat unchanged**: the energy it
converges to is the lowest *time-reversal-symmetric* one, which is not the unconstrained
optimum wherever that symmetry is spontaneously broken. ⚠ **And it presumes the incoming
columns really are ``(psi, T psi)`` pairs**, which is why ``"auto"`` measures them rather than
trusting a flag: an unrestricted reference's spinors are legitimately not paired, and neither
are the active orbitals a previous *unconstrained* run converged to.

Releasing it: the constrained solution is tested, not assumed
-------------------------------------------------------------
Spontaneous breaking is not hypothetical at an even active electron count — N2 CAS(6,8)
converges 0.64 Eh **below** its symmetric stationary point with 42% of its density time-odd —
and no *static* quantity separates that from the drift: both leave the symmetric point along a
time-odd direction and both descend monotonically. What separates them is a **measurement at
the converged point**: whether the symmetric solution is a minimum of the unconstrained
problem, i.e. whether the orbital Hessian has negative curvature in the directions the
constraint projected out. That is :func:`measure_time_odd_curvature` — the lowest eigenvalue
of ``P_odd H P_odd`` by Davidson, a few tens of Hessian-vector products once per run — and
where it is negative, :func:`optimize_orbitals` releases the constraint, steps off the saddle
along the offending direction and continues unconstrained. The orbital analogue of the SCF's
``stability="follow"``, and the reason the constraint can be the default at **both**
parities: a broken solution that is real is still reached, and one that is only drift never
is. ⚠ Measured on the two characterized cases: N2 CAS(6,8) releases at -0.34 Eh/rad^2 and
recovers the unconstrained energy to 1.3e-09 Eh; a healthy run reports a positive curvature
and finishes where it was.

⚠ **The release is even-electron only, and that is what parity still decides.** At an odd
count a time-reversal-broken solution has no Kramers degeneracy and the state-averaging gate
refuses it downstream, so there is nothing to release *to* — the verdict is reported and the
constraint kept.

⚠ **The remedy that does not work, recorded so it is not tried again**: repairing the
*orbitals* after each step by projecting every space onto its nearest time-reversal-closed
span (:func:`kuiva.interface.api.kramers_span_repair`, still present and still wired to
nothing). It removes the drift exactly and is wrong anyway — "nearest closed span" stops
meaning "almost the same span" once a block has drifted, so it silently **re-selects the
active space**, measured as an N2 CAS(6,8) converging 0.6 Eh *above its own SCF energy*.

⚠ Stability notes. The diagonal Hessian is a **preconditioner**, not a curvature
claim: it is floored away from zero, its anisotropy is clamped (measured up to 10x too small
on soft modes), and a persistently negative diagonal is reported rather than silently damped.
The L-BFGS curvature is accumulated on the **CI-optimized** surface — legitimate, because a
re-solved CI makes ``E(kappa)`` smooth with precisely the orbital gradient — but the pairs are
transported between charts without correction, which is exact only to first order in the step.

⚠ The scheme is **two-step**: the orbital-CI coupling is not in the Hessian, so the asymptotic
rate is linear rather than quadratic. Demonstrated cleanly on a stiff case — with the RDMs
frozen (a pure orbital minimization, where the Hessian is exact for the surface) the same
optimizer converges to ``|g| = 1.1e-8``; with the CI re-solved each macro-iteration it settles
at ``6e-4``. That gap *is* the neglected coupling. Including it would require applying the
Hessian to CI vectors, which would break the ci_solver contract that lets a truncated CI, a full CI
and a DMRG plug in identically.

References
----------
* CASSCF orbital gradients, the generalized Fock matrix and the exponential parametrization:
  T. Helgaker, P. Jorgensen, J. Olsen, "Molecular Electronic-Structure Theory", Wiley (2000),
  ch. 10 and 12; B. O. Roos, P. R. Taylor, P. E. M. Siegbahn, Chem. Phys. 48, 157 (1980),
  doi:10.1016/0301-0104(80)80045-0; P. E. M. Siegbahn, J. Almlof, A. Heiberg, B. O. Roos,
  J. Chem. Phys. 74, 2384 (1981), doi:10.1063/1.441359.
* Second-order / augmented-Hessian MCSCF: H. J. Aa. Jensen, H. Agren, Chem. Phys. Lett. 110,
  140 (1984), doi:10.1016/0009-2614(84)80166-1; H. J. Aa. Jensen, P. Jorgensen, J. Chem.
  Phys. 80, 1204 (1984), doi:10.1063/1.446797; D. A. Kreplin, P. J. Knowles, H.-J. Werner,
  J. Chem. Phys. 150, 194106 (2019), doi:10.1063/1.5094644.
* One-index transformed Fock matrices, which make a Hessian-vector product cost the same
  order as a gradient: P. Jorgensen, P. Swanstrom, D. L. Yeager, J. Chem. Phys. 78, 347
  (1983), doi:10.1063/1.444508; Helgaker, Jorgensen & Olsen (2000), section 10.8.
* Inexact-Newton forcing sequences for the augmented-Hessian solve: S. C. Eisenstat,
  H. F. Walker, SIAM J. Sci. Comput. 17, 16 (1996), doi:10.1137/0917003; R. S. Dembo,
  S. C. Eisenstat, T. Steihaug, SIAM J. Numer. Anal. 19, 400 (1982), doi:10.1137/0719025.
* Level shifting to enforce a trust radius: T. Helgaker, Chem. Phys. Lett. 182, 503 (1991),
  doi:10.1016/0009-2614(91)90115-P.
* L-BFGS: J. Nocedal, Math. Comput. 35, 773 (1980), doi:10.1090/S0025-5718-1980-0572855-7;
  D. C. Liu, J. Nocedal, Math. Program. 45, 503 (1989), doi:10.1007/BF01589116. Trust-region
  globalization: J. Nocedal, S. J. Wright, "Numerical Optimization", 2nd ed., Springer (2006).
* Two-component / relativistic MCSCF, where the rotation parameters are complex:
  H. J. Aa. Jensen, K. G. Dyall, T. Saue, K. Fægri, J. Chem. Phys. 104, 4083 (1996),
  doi:10.1063/1.471644; T. Fleig, J. Olsen, C. M. Marian, J. Chem. Phys. 114, 4775 (2001),
  doi:10.1063/1.1349076.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..integrals.transform import (ThreeIndexAO, assemble_4c, coulomb_exchange,
                                   diagonal_pair_blocks, half_transform,
                                   half_transform_memory_gb, mo_block_memory_gb,
                                   transform_3c)
from ..util import output as out
from ..util import resources
from ..util import threads
from ..util.errors import SolverFailure
from ..util.logging import get_logger
from ..util.timing import timer

log = get_logger(__name__)

#: Largest single rotation angle [rad] allowed in one macro-iteration.
DEFAULT_MAX_STEP = 0.20
#: Gradient norm below which ``mode="auto"`` escalates to the second-order step. **Zero by
#: default: escalation is driven by failure, not by proximity.** Measured: where the
#: quasi-Newton step converges at all it costs 3-4x less total work than the second-order one
#: (118 vs 454 gradient-equivalents on an easy case), because a Hessian-vector product is not
#: free and an augmented-Hessian solve needs many. Raise it when a *macro-iteration* is the
#: expensive thing — a DMRG-CASSCF, where one CI solve dwarfs any number of Fock builds, wants
#: the fewest possible iterations and should set this high or use ``mode="second-order"``.
SECOND_ORDER_START = 0.0
#: Cost of one Hessian-vector product relative to one macro-iteration, for
#: :attr:`CASSCFResult.work_units`. See that docstring: system dependent, measured ~2.9 on
#: TiCl3 with the cheap CI.
HVP_WORK_WEIGHT = 1.5
#: Iterations over which gradient progress is judged, and the reduction expected across that
#: window. Escalation is driven by the *gradient trajectory*, not the energy: an energy that
#: has stopped moving cannot distinguish "converging slowly" from "stuck", because a
#: quasi-Newton run does both with the same tiny per-iteration energy change. A gradient that
#: keeps falling — however slowly — is converging and does not need rescuing.
STALL_WINDOW = 15
STALL_FACTOR = 2.0
#: Absolute floor on the diagonal Hessian preconditioner [Eh]; see the stability note above.
HESSIAN_FLOOR = 1.0e-3
#: Relative pairing defect below which the incoming orbitals count as Kramers paired, and
#: ``kramers_rotation="auto"`` therefore constrains the rotation
#: (:func:`kuiva.spinor.expand.kramers_pairing_defect`). ⚠ **A gate, not a tolerance**: it
#: says where roundoff stops, not how much breach is acceptable. A guess built by
#: :func:`kuiva.spinor.expand.expand_scalar_mos` is paired to 0, a constrained optimization
#: leaves 1e-15..1e-13 after hundreds of steps, and everything the constraint cannot be
#: applied to — an unrestricted reference, a converged Kramers-unrestricted CASSCF's active
#: orbitals — is O(0.1) or worse. There is nothing in between to get wrong.
KRAMERS_PAIRING_TOL = 1.0e-10
#: Curvature [Eh/rad^2] below which a converged Kramers-constrained solution counts as a
#: **saddle** in the time-reversal-odd rotations the constraint forbids
#: (:func:`measure_time_odd_curvature`). ⚠ **A gate, not a tolerance**, and the two sides of
#: it are orders apart rather than adjacent: a genuine symmetry-breaking instability is
#: O(0.1-1) Eh/rad^2 (N2 CAS(6,8): -0.53), while a stable solution's lowest time-odd
#: curvature is a positive number of the same size as its stiffest even ones. What this
#: value has to exclude is only the residual of an *imperfectly* converged optimization,
#: whose gradient is conv_grad and whose curvature error is smaller.
KRAMERS_STABILITY_TOL = 1.0e-3
#: Largest rotation angle [rad] of the step that leaves a time-odd saddle when the constraint
#: is released (:func:`kramers_release_rotation`). It only has to break the symmetry — the
#: instability supplies the rest of the descent — so it is deliberately well inside
#: :data:`DEFAULT_MAX_STEP` and is not a tuned quantity.
KRAMERS_RELEASE_STEP = 0.10
#: Relative floor, as a fraction of the median diagonal. The approximate diagonal is a poor
#: estimate of curvature for *soft* modes specifically — measured against a numerical diagonal
#: Hessian it is a factor 0.67 too small on average but up to 10x too small on the softest
#: rotations, which makes the preconditioned step there far too long and costs rejections.
#: Clamping the anisotropy keeps the reliable information (virtual-inactive rotations really
#: are stiffer than active ones) and discards the unreliable extreme. Measured effect on a
#: stiff synthetic CASSCF: 300 macro-iterations to |g| < 1e-5 without it, 156 with.
HESSIAN_RELATIVE_FLOOR = 0.3


# --- Orbital spaces --------------------------------------------------------------------

@dataclass(frozen=True)
class OrbitalSpaces:
    """Partition of the spinors into inactive / active / virtual.

    Index arrays rather than counts, because the active space is selected by orbital
    *character* and need not be contiguous (necessarily so for an unrestricted
    reference).
    """

    inactive: np.ndarray
    active: np.ndarray
    virtual: np.ndarray
    n_orb: int

    def __post_init__(self) -> None:
        for name in ("inactive", "active", "virtual"):
            object.__setattr__(self, name,
                               np.asarray(getattr(self, name), dtype=int).ravel())
        allidx = np.concatenate([self.inactive, self.active, self.virtual])
        if allidx.size != self.n_orb or np.unique(allidx).size != self.n_orb:
            raise ValueError(
                "the three spaces must partition the {} spinors exactly; got {} indices "
                "({} unique)".format(self.n_orb, allidx.size, np.unique(allidx).size))

    @classmethod
    def from_counts(cls, n_inactive: int, n_active: int, n_orb: int) -> "OrbitalSpaces":
        """Contiguous partition — valid for a Kramers-paired restricted reference."""
        i = np.arange(n_inactive)
        a = np.arange(n_inactive, n_inactive + n_active)
        v = np.arange(n_inactive + n_active, n_orb)
        return cls(inactive=i, active=a, virtual=v, n_orb=n_orb)

    @property
    def n_inactive(self) -> int:
        return int(self.inactive.size)

    @property
    def n_active(self) -> int:
        return int(self.active.size)

    @property
    def n_virtual(self) -> int:
        return int(self.virtual.size)

    def rotation_pairs(self, active_active: bool = False,
                       labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Non-redundant rotation pairs ``(p, q)``, ``p`` from the higher space.

        Inactive-inactive and virtual-virtual rotations are **exactly** redundant (they
        change no observable) and are always excluded. Active-active rotations are redundant
        with the CI parameters for an exact CAS and are excluded by default; they are *not*
        redundant for a truncated CI, which is why the flag exists — but the
        preferred way to fix the active-space basis there is the natural-spinor rotation, not
        an energy gradient that a truncated CI cannot make stationary.

        ``labels`` is an ``(n_orb, width)`` array of irrep labels (:mod:`kuiva.symm`). Given
        it, pairs joining **different** irreps are dropped, so ``kappa`` is block diagonal over
        the irreps and ``exp(kappa)`` cannot mix them: the orbitals stay symmetry-pure
        *exactly*, at every iteration, by construction rather than to a tolerance. That is
        what makes an irrep label still mean something at convergence, and it is the same
        mechanism the redundant rotations above are removed by — a mask on the parameter list,
        with the rest of the optimizer untouched.

        ⚠ It is a **constraint**, and a constrained optimum is not the unconstrained one
        wherever the symmetry is spontaneously broken. The energy it converges to is the lowest
        *symmetric* one; where that matters the answer is to run without the mask and read the
        drift the CI measures, not to loosen the mask.
        """
        rows, cols = [], []
        for hi, lo in ((self.active, self.inactive), (self.virtual, self.inactive),
                       (self.virtual, self.active)):
            if hi.size and lo.size:
                rows.append(np.repeat(hi, lo.size))
                cols.append(np.tile(lo, hi.size))
        if active_active and self.active.size > 1:
            t = self.active
            lo_pos, hi_pos = np.triu_indices(t.size, k=1)
            rows.append(t[hi_pos])
            cols.append(t[lo_pos])
        if not rows:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        r, c = np.concatenate(rows), np.concatenate(cols)
        if labels is not None:
            lab = np.atleast_2d(np.asarray(labels, dtype=int))
            if lab.shape[0] != self.n_orb:
                raise ValueError("{} labels for {} spinors".format(lab.shape[0], self.n_orb))
            same = np.all(lab[r] == lab[c], axis=1)
            r, c = r[same], c[same]
        return r, c

    def __repr__(self) -> str:
        return "OrbitalSpaces(inactive={}, active={}, virtual={})".format(
            self.n_inactive, self.n_active, self.n_virtual)


# --- Integrals in the form the gradient needs ------------------------------------------

def cas_integrals_memory_gb(naux: int, n_orb: int, n_active: int) -> float:
    """Size [GB] of the three-index block :class:`CASIntegrals` holds (exact sizing function).

    ``b_act`` is ``B^P_{p,t}`` — every spinor against the **active** ones — and it is the only
    large array the orbital optimizer and the CI drivers own. The square ``B^P_{pq}`` is never
    built on any production path, so a memory plan that budgets one refuses calculations that
    would have run; this is the function such a plan must use instead. The one- and
    two-electron matrices beside it are ``O(n^2)`` and negligible against it.
    """
    return mo_block_memory_gb(naux, n_orb, n_active, np.complex128)


#: Copies of ``b_act`` that a second-order Hessian-vector product holds **on top of** the
#: resident one, at its peak: the bra-side response ``kappa^dag B``, the ket-side transform,
#: and the sum of the two, all of the same shape and all live at the moment of the addition
#: (:meth:`OrbitalHessian.matvec`). ⚠ It is a property of that expression, not a safety
#: factor — change how the response is accumulated and this number changes with it.
HESSIAN_RESPONSE_COPIES = 3


def hessian_response_memory_gb(naux: int, n_orb: int, n_active: int) -> float:
    """Transient [GB] a second-order step adds to the resident ``b_act`` (exact sizing).

    Transient rather than resident: the copies live inside one matrix-vector product and are
    gone before the next. A memory plan still has to carry them, because they are the peak of
    the second-order path and they are three times the array the plan already names — an
    optimizer that escalates (``mode="auto"`` does, on the gradient trajectory) would otherwise
    allocate four times what was budgeted for it.
    """
    return HESSIAN_RESPONSE_COPIES * cas_integrals_memory_gb(naux, n_orb, n_active)


def hessian_square_memory_gb(naux: int, n_orb: int, n_occ: int) -> float:
    """Size [GB] of the fixed-factor route's resident blocks (exact sizing function).

    The full MO three-index square ``B^P_pq`` plus its occupied blocks in their two
    contraction-ready layouts (``B^P_{p,occ}`` twice over, ``B^P_{occ,occ}`` once), all
    ``complex128`` — what :class:`OrbitalHessian` holds for the span of one
    macro-iteration when the integral-free product route engages (see its docstring).
    This is the deliberate, size-gated exception to the rule that no production path
    transforms the square: it is taken only where this size fits the transient budget,
    and the blocked AO route remains the fallback, so a plan carrying
    ``min(this, transient budget)`` never refuses a calculation the fallback would have
    run.
    """
    return (resources.array_gb((naux, n_orb, n_orb))
            + 2.0 * resources.array_gb((naux, n_orb, n_occ))
            + resources.array_gb((naux, n_occ, n_occ)))


@dataclass
class CASIntegrals:
    """The integral quantities the orbital gradient and the CI need, at fixed orbitals.

    Deliberately *not* the full MO integral set: only the one-electron matrix, the inactive
    Fock, and the three-index block with one active index. See the module docstring.
    """

    h: np.ndarray                    # (n, n) one-electron, MO basis
    f_inactive: np.ndarray           # (n, n) inactive Fock, MO basis
    b_act: np.ndarray                # (naux, n, n_active)
    e_core: float                    # inactive energy, including E_nuc
    spaces: OrbitalSpaces
    #: Inactive Fock in the **AO** basis. Kept because :class:`OrbitalHessian` needs it and
    #: rebuilding it costs a J/K build — measured at a third of a second-order step's setup
    #: on a real system, for a matrix that was already in hand.
    f_inactive_ao: Optional[np.ndarray] = None

    @property
    def n_orb(self) -> int:
        return int(self.h.shape[0])

    def active_eri(self) -> np.ndarray:
        """``(tu|vw)`` over the active spinors."""
        b = self.b_act[:, self.spaces.active, :]
        return assemble_4c(b, b)

    def h_active_effective(self) -> np.ndarray:
        """The active-space one-electron Hamiltonian, i.e. ``F^I`` restricted to active."""
        act = self.spaces.active
        return self.f_inactive[np.ix_(act, act)]

    @classmethod
    def build(cls, factors: ThreeIndexAO, h_ao: np.ndarray, c_spinor: np.ndarray,
              spaces: OrbitalSpaces, e_nuc: float = 0.0) -> "CASIntegrals":
        """Transform at the current orbitals. One AO Fock build plus one active transform."""
        with timer("CAS integrals"):
            c = np.ascontiguousarray(c_spinor)
            h_mo = c.conj().T @ h_ao @ c
            # Factorized inactive density: every inactive spinor is singly occupied.
            j, k = coulomb_exchange(factors, np.ascontiguousarray(c[:, spaces.inactive]))
            f_i_ao = h_ao + j - k
            f_i = c.conj().T @ f_i_ao @ c
            inac = np.ix_(spaces.inactive, spaces.inactive)
            e_core = 0.5 * float(np.real(np.trace(h_mo[inac]) + np.trace(f_i[inac]))) + e_nuc
            b_act = transform_3c(factors, c, np.ascontiguousarray(c[:, spaces.active]))
        # ``b_act`` is resident for as long as this object lives — one macro-iteration, or
        # two blocks at a trial step, when the incumbent's and the trial's overlap — so the
        # ledger carries a *reservation* for it, not only ``transform_3c``'s pre-allocation
        # check, or the largest array the optimizer holds would be invisible to every later
        # refusal. Tied to the array's lifetime because this constructor runs inside the
        # driver loops: an explicit release at every accept/reject/failure branch that drops
        # an instance is the leak-prone alternative.
        resources.reserve_owned(
            b_act, "three-index MO block B^P_pt",
            cas_integrals_memory_gb(factors.naux, b_act.shape[1], spaces.n_active),
            note="{} spinors x {} active, naux = {}".format(
                b_act.shape[1], spaces.n_active, factors.naux),
            advice=["reduce the active space: this block carries one index over it",
                    "reduce the auxiliary/Cholesky dimension, which it is linear in"])
        # The ``(n_act)^4`` active ERI block is assembled on demand (:meth:`active_eri`),
        # several times per macro-iteration; its fit is checked once here, where the size is
        # first known, rather than per assembly inside the driver loop.
        resources.require("active four-index integrals",
                          resources.array_gb((spaces.n_active,) * 4, np.complex128),
                          advice=["reduce the active space"])
        return cls(h=h_mo, f_inactive=f_i, b_act=b_act, e_core=e_core, spaces=spaces,
                   f_inactive_ao=f_i_ao)


# --- Energy, generalized Fock, gradient --------------------------------------------------

def cas_energy(ints: CASIntegrals, gamma: np.ndarray, gamma2: np.ndarray) -> float:
    """State-averaged CASSCF energy for the given RDMs [Eh]."""
    act = ints.spaces.active
    e1 = np.einsum("tu,tu->", ints.f_inactive[np.ix_(act, act)], gamma)
    e2 = 0.5 * np.einsum("tuvw,tuvw->", ints.active_eri(), gamma2)
    return float(np.real(e1 + e2)) + ints.e_core


def _active_fock(ints: CASIntegrals, factors: ThreeIndexAO, c_spinor: np.ndarray,
                 gamma: np.ndarray) -> np.ndarray:
    """``F^A`` in the MO basis, built as an AO Coulomb/exchange from the active density.

    The active density ``sum_tu c_t gamma_tu c_u^dag`` is not idempotent, so it is factorized
    through the natural spinors of ``gamma`` — which exist because a 1-RDM is Hermitian
    positive semidefinite — and fed to the same occupied-index-first builder as the inactive
    density. Occupation numbers below zero can only come from rounding and are clamped there.
    """
    return c_spinor.conj().T @ _active_fock_ao(factors, c_spinor, ints.spaces,
                                               gamma) @ c_spinor


def _active_fock_ao(factors: ThreeIndexAO, c_spinor: np.ndarray, spaces: OrbitalSpaces,
                    gamma: np.ndarray) -> np.ndarray:
    """``F^A`` in the AO basis. Kept separate so callers that need both the MO and the AO form
    (the gradient and the Hessian) build it once instead of twice."""
    c_act = np.ascontiguousarray(c_spinor[:, spaces.active])
    # The AO density reproducing sum_tu gamma_tu (pq|tu) is C gamma^T C^dag; with the
    # two-factor builder no eigendecomposition is needed — pass the weight matrix directly.
    j, k = coulomb_exchange(factors, np.ascontiguousarray(c_act @ gamma.T),
                            orbitals_right=c_act)
    return j - k


def averaged_fock(ints: CASIntegrals, factors: ThreeIndexAO, c_spinor: np.ndarray,
                  gamma: np.ndarray) -> np.ndarray:
    """``F^I + F^A`` in the MO basis, for the state-averaged 1-RDM ``gamma`` [Eh].

    The one-body operator whose *diagonal* :func:`fock_diagonal` reports and whose
    inactive/virtual blocks SC-NEVPT2 diagonalizes to define the Dyall zeroth-order orbital
    energies (:func:`kuiva.pt.nevpt2.pseudo_canonicalize`). Exposed as its own function
    only so that the perturbation layer does not have to re-derive it or reach into a private
    name; nothing about the construction is new here.

    ⚠ **Hermitian in exact arithmetic, and the caller should symmetrize rather than assume.**
    ``F^A`` comes from a Coulomb/exchange build on a factorized density, so its Hermiticity is
    accurate to rounding and not exact.
    """
    return ints.f_inactive + _active_fock(ints, factors, c_spinor, gamma)


def fock_diagonal(ints: CASIntegrals, factors: ThreeIndexAO, c_spinor: np.ndarray,
                  gamma: np.ndarray) -> np.ndarray:
    """Diagonal of ``F^I + F^A`` in the current orbital basis — "orbital energies" [Eh].

    What the molden dump and any orbital listing print as ``Ene=``. It is the natural
    generalization of an SCF eigenvalue to an MCSCF orbital set: for an inactive orbital it is
    the usual Fock expectation value, and for an active one it is the average-of-configuration
    energy of that spinor in the field of the converged density.

    ⚠ **It is a label, not a physical excitation energy**, and the orbitals are not
    eigenfunctions of this operator unless they were canonicalized — the off-diagonal elements
    are generally not small. Use it for ordering and identification; the energies that mean
    something are the state energies.

    Real by construction (the diagonal of a Hermitian matrix); the imaginary part is discarded
    after being checked.
    """
    f = averaged_fock(ints, factors, c_spinor, gamma)
    diag = np.diag(f)
    imag = float(np.max(np.abs(diag.imag))) if diag.size else 0.0
    if imag > 1e-8 * max(float(np.max(np.abs(diag.real))), 1.0):
        log.warning("Fock diagonal has a non-negligible imaginary part (max %.2e); the Fock "
                    "matrix should be Hermitian", imag)
    return np.ascontiguousarray(diag.real)


def generalized_fock(ints: CASIntegrals, gamma: np.ndarray, gamma2: np.ndarray,
                     f_active: np.ndarray) -> np.ndarray:
    """The generalized Fock ``F`` (see the module docstring). Virtual rows are zero."""
    sp = ints.spaces
    n = ints.n_orb
    f = np.zeros((n, n), dtype=np.complex128)
    fia = ints.f_inactive + f_active
    # inactive rows: F_iq = (F^I + F^A)_qi
    f[sp.inactive, :] = fia[:, sp.inactive].T
    # active rows: F_tq = sum_u gamma_tu F^I_qu + sum_uvw Gamma_tuvw (qu|vw)
    act = sp.active
    b_act_act = ints.b_act[:, act, :]                          # (naux, nact, nact)
    # As GEMMs over flattened index pairs, not einsum — see OrbitalHessian for why.
    nact = sp.n_active
    naux = b_act_act.shape[0]
    m = (b_act_act.reshape(naux, nact ** 2)
         @ gamma2.reshape(nact ** 2, nact ** 2).T).reshape(naux, nact, nact)
    lhs = np.ascontiguousarray(m.transpose(1, 0, 2)).reshape(nact, -1)
    rhs = np.ascontiguousarray(ints.b_act.transpose(1, 0, 2)).reshape(n, -1)
    f[act, :] = gamma @ ints.f_inactive[:, act].T + lhs @ rhs.T
    return f


def orbital_gradient(ints: CASIntegrals, gamma: np.ndarray, gamma2: np.ndarray,
                     f_active: np.ndarray) -> np.ndarray:
    """The anti-Hermitian gradient matrix ``G = F^dag - F``.

    ``dE = 2 Re sum_{p>q} G_pq kappa_pq`` for an anti-Hermitian step ``kappa``.
    """
    f = generalized_fock(ints, gamma, gamma2, f_active)
    return f.conj().T - f


class OrbitalHessian:
    """Exact orbital-orbital Hessian-vector products at fixed RDMs.

    ``H(kappa) = d/dt grad(C exp(t kappa))|_0``, computed analytically. The energy depends on
    ``kappa`` only through the integrals, so differentiating the gradient means differentiating
    the integrals — a **one-index transformation**. Written in terms of the orbital response
    ``X = C kappa``, every two-electron term collapses into ordinary Coulomb/exchange builds
    with *transition* densities:

        dD^I = X_I C_I^dag + C_I X_I^dag,    dD^A = X_T g^T C_T^dag + C_T g^T X_T^dag

    which are Hermitian but **indefinite**, hence the two-factor
    :func:`~kuiva.integrals.transform.coulomb_exchange`. Each is built **one-sided** —
    ``J/K(A)`` for ``A = X C^dag`` alone, Hermitized afterwards, which is exact because J
    and K are linear in the density and map ``A^dag`` to the Hermitian adjoint over real AO
    integrals — and the kappa-independent right-hand half transforms (of ``C_I`` and
    ``C_T``) are cached across every product of the macro-iteration. Together that removes
    about three quarters of the J/K work a symmetrized two-sided build pays per product. So
    one Hessian-vector product costs two half-width J/K builds plus one three-index
    transform — the same order as a gradient, not the ``O(n^4)`` an explicit Hessian would
    need.

    The fixed-factor route (default where it fits)
    ----------------------------------------------
    That AO build is the **fallback**. Where the full MO factor square ``B^P_pq`` fits the
    transient budget (:func:`hessian_square_memory_gb`; ``mo_square=`` overrides), it is
    built once per macro-iteration and every product becomes **integral-free dense MO
    algebra** (:meth:`_response_square`) — no AO pass, no half transform, no per-product
    ``transform_3c``, and :meth:`exact_diagonal`'s streamed pass reduces to array
    reductions on the held blocks. This adapts the fixed-integral micro-iteration idea of
    Werner & Meyer, J. Chem. Phys. 73, 2342 (1980), and Werner & Knowles, J. Chem. Phys.
    82, 5053 (1985), to three-index factors: their stored ``J^{ij}/K^{ij}`` occupied-pair
    matrices scale as ``n_occ^2 n^2`` — four times each dimension of the scalar case for
    spinors, gigabytes already for a diatomic — where the square is ``naux n^2`` and the
    products still avoid every integral pass. ⚠ **This is the one sanctioned exception to
    the no-square rule** for the three-index factors, taken deliberately, size-gated, and
    with the blocked route always available: the plan carries
    ``min(hessian_square_memory_gb, transient budget)``, so the gate can never refuse a
    calculation the fallback would have run. The two routes agree to roundoff and a test
    holds them together.

    The one economy worth naming: the *bra*-side three-index response needs no new integrals
    at all. Since ``X = C kappa``,

        Bx1^P_{p,t} = sum_mu conj(X_{mu p}) L^P C_{nu t} = sum_r conj(kappa_{rp}) Bc^P_{r,t}

    is a small contraction with the ``b_act`` already held by :class:`CASIntegrals`. Only the
    ket-side response ``Bx2 = transform_3c(L, C, X_act)`` is a genuine transform.

    The chart curvature term
    ------------------------
    The raw derivative of the gradient *field* is **not symmetric**: differentiating
    ``g(kappa')`` — the gradient at the rotated point, expressed in *that point's* frame —
    is not the same as the Hessian of ``f(kappa) = E(C exp(kappa))``. They differ by a
    curvature term of the exponential chart,

        d^2 f . kappa  =  dg/dt  -  1/2 [kappa, conj(G)]

    with ``G`` the gradient matrix. The correction is applied here, so :meth:`matvec` returns
    the **true, symmetric** Hessian-vector product. It was verified by building both operators
    densely on a small case: the identity holds to ``7e-15``, and the corrected operator's
    symmetry to machine precision.

    This is not a cosmetic detail. The correction vanishes at a stationary point and has zero
    quadratic form everywhere — ``<A d, d>`` equals ``d^2f/dt^2`` with or without it — so it is
    invisible to every obvious check. But an asymmetric operator makes Davidson wander: before
    the correction the augmented-Hessian solve hit its iteration limit without converging on
    every step away from the solution, and the optimizer converged *linearly*.

    .. warning::
       This is the **orbital-orbital** block only; the orbital-CI coupling is not included.
       That makes the scheme a *two-step* second-order MCSCF: the CI is re-solved between
       macro-iterations, and near convergence the neglected coupling slows the asymptotic rate
       relative to a fully coupled one-step method. It is also what keeps the solver-agnostic contract
       intact — the optimizer never needs to apply H to a CI vector, so DMRG and a truncated CI
       plug in on equal terms. A one-step method would have to break that.
    """

    def __init__(self, ints: CASIntegrals, factors: ThreeIndexAO, h_ao: np.ndarray,
                 c_spinor: np.ndarray, gamma: np.ndarray, gamma2: np.ndarray,
                 rows: np.ndarray, cols: np.ndarray,
                 f_active_ao: Optional[np.ndarray] = None,
                 mo_square: Optional[bool] = None):
        self.ints = ints
        self.factors = factors
        self.h_ao = h_ao
        self.c = np.ascontiguousarray(c_spinor)
        self.gamma = gamma
        self.gamma2 = gamma2
        self.rows, self.cols = rows, cols
        sp = ints.spaces
        self.sp = sp
        self.n_calls = 0
        # kappa-independent quantities, built once per macro-iteration.
        self.c_i = np.ascontiguousarray(self.c[:, sp.inactive])
        self.c_t = np.ascontiguousarray(self.c[:, sp.active])
        # Reuse the AO Fock matrices the caller already built. Each is a J/K build, and on a
        # real system the *setup* of a Hessian dominates its first few matrix-vector products.
        if ints.f_inactive_ao is not None:
            self.f_i_ao = ints.f_inactive_ao
        else:
            j_i, k_i = coulomb_exchange(factors, self.c_i)
            self.f_i_ao = h_ao + j_i - k_i
        if f_active_ao is not None:
            self.f_a_ao = f_active_ao
        else:
            j_a, k_a = coulomb_exchange(factors, np.ascontiguousarray(self.c_t @ gamma.T),
                                        orbitals_right=self.c_t)
            self.f_a_ao = j_a - k_a
        self.b_act = ints.b_act                                   # (naux, n, nact)
        self.nact = sp.n_active
        self.g2_flat = np.ascontiguousarray(
            gamma2.reshape(self.nact ** 2, self.nact ** 2))
        self.mc = self._contract_gamma2(self.b_act[:, sp.active, :])
        # Half-transform cache for the response J/K builds (see matvec): the right-hand
        # factors of both transition densities are the kappa-independent ``c_i`` and
        # ``c_t``, so their half transforms are paid once here instead of once per
        # Hessian-vector product — in ONE pass over the factors for both, with the two
        # blocks handed out as column views of the combined array. The same array, with
        # its column identity, is what :meth:`exact_diagonal` reuses for its occupied
        # stage. Sized against the transient budget — the same allowance any blocked
        # kernel gets, asked once, outside every loop — and skipped rather than shrunk
        # when it does not fit: the uncached path is exact, just slower.
        naux, nao = factors.naux, factors.nao
        n_i, n_t = self.c_i.shape[1], self.c_t.shape[1]
        n_occ = n_i + n_t
        self._occ_cols = np.concatenate([sp.inactive, sp.active])
        # Route decision: the fixed-factor (integral-free) product when the MO square fits
        # the transient budget, the blocked AO route otherwise — asked once, outside every
        # loop, and never a correctness question: the two routes agree to roundoff.
        square_gb = hessian_square_memory_gb(naux, ints.n_orb, n_occ)
        if mo_square is None:
            mo_square = n_occ > 0 and square_gb <= resources.transient_gb()
        self.mo_square = bool(mo_square)
        # What the kappa-sparse product forms may assume (see _response_square): the
        # occupied orbitals sit at the leading indices, and — for the active response —
        # no active-active rotations are parameters. Detected once; the dense general
        # forms remain the fallback and a test holds the two together.
        sid = np.zeros(ints.n_orb, dtype=int)
        sid[sp.active] = 1
        sid[sp.virtual] = 2
        self._has_act_act = bool(np.any((sid[rows] == 1) & (sid[cols] == 1)))
        self._occ_prefix = bool(np.array_equal(self._occ_cols, np.arange(n_occ)))
        if self.mo_square:
            # The whole of what the fixed-factor route holds; kappa-independent, one
            # transform per macro-iteration, and every Hessian-vector product afterwards
            # is dense MO algebra on these blocks (see _response_square). The blocks are
            # stored in contraction-ready layouts so each term below is ONE large GEMM —
            # batched matmul-and-reduce over the auxiliary index measured 13 GFLOP/s
            # against ~2.5x that for the flat forms, the same lesson as the J/K build's
            # einsum history.
            # ⚠ transform_3c, deliberately: a real-arithmetic square assembly (three real
            # GEMMs per spin, the fourth product free by the symmetry of L) was built,
            # measured and REMOVED — 25% fewer flops, 29% more CPU seconds, because the
            # strided stores into the interleaved complex result and the batched
            # transpose cost more than the flops saved. The wide-ket zgemm path is the
            # measured optimum of the assemblies tried, judged on CPU seconds.
            with timer("MO factor square"):
                n_orb = ints.n_orb
                self.b_full = transform_3c(factors, self.c, self.c)
                b_occ = np.ascontiguousarray(self.b_full[:, :, self._occ_cols])
                #: (n, naux, n_occ) — the occupied columns, aux-middle layout.
                self.b_occ_t = np.ascontiguousarray(b_occ.transpose(1, 0, 2))
                b_oo = b_occ[:, self._occ_cols, :]
                #: ((naux n_i), n_occ) and ((naux n_t), n_occ) — occupied-row blocks.
                self.b_i_occ_flat = np.ascontiguousarray(
                    b_oo[:, :n_i, :]).reshape(naux * n_i, n_occ)
                self.b_a_occ_flat = np.ascontiguousarray(
                    b_oo[:, n_i:, :]).reshape(naux * n_t, n_occ)
                #: (n, (naux n_i)) and (n, (naux n_t)) — occupied-column blocks.
                self.b_p_i_flat = np.ascontiguousarray(
                    b_occ[:, :, :n_i].transpose(1, 0, 2)).reshape(n_orb, naux * n_i)
                self.b_p_a_flat = np.ascontiguousarray(
                    b_occ[:, :, n_i:].transpose(1, 0, 2)).reshape(n_orb, naux * n_t)
            self._w_occ = None
            self._w_i = self._w_t = None
        else:
            self.b_full = self.b_occ_t = None
            self.b_i_occ_flat = self.b_a_occ_flat = None
            self.b_p_i_flat = self.b_p_a_flat = None
            want_gb = half_transform_memory_gb(naux, nao, n_occ)
            if want_gb <= resources.transient_gb():
                self._w_occ = half_transform(
                    factors, np.concatenate([self.c_i, self.c_t], axis=1))
                self._w_i = [w[:, :, :n_i] for w in self._w_occ] if n_i else None
                self._w_t = [w[:, :, n_i:] for w in self._w_occ] if n_t else None
            else:
                self._w_occ = None
                self._w_i = self._w_t = None
                log.debug("half-transform cache (%.2f GB) exceeds the transient budget; "
                          "each Hessian-vector product recomputes the right-hand half "
                          "transform", want_gb)
        # Gradient matrix, for the chart-curvature correction (see the class docstring).
        f_active = self.c.conj().T @ self.f_a_ao @ self.c
        self.f_act_mo = f_active
        self.g_dual = np.conj(orbital_gradient(ints, gamma, gamma2, f_active))
        self._exact_diag: Optional[np.ndarray] = None

    def _fock_mo(self, f_ao: np.ndarray, x: np.ndarray) -> np.ndarray:
        """``X^dag F C + C^dag F X`` — the one-electron part of a one-index transformation."""
        return x.conj().T @ f_ao @ self.c + self.c.conj().T @ f_ao @ x

    def _contract_gamma2(self, b_aa: np.ndarray) -> np.ndarray:
        """``M^P_{tu} = sum_vw Gamma_tuvw B^P_{vw}`` as one GEMM over the flattened pairs."""
        naux = b_aa.shape[0]
        flat = b_aa.reshape(naux, self.nact ** 2)
        return (flat @ self.g2_flat.T).reshape(naux, self.nact, self.nact)

    @staticmethod
    def _pair_contract(b_gen: np.ndarray, m_act: np.ndarray) -> np.ndarray:
        """``R_{tq} = sum_{P,u} B^P_{qu} M^P_{tu}`` as one GEMM over the flattened ``(P, u)``."""
        naux, n_gen, nact = b_gen.shape
        lhs = np.ascontiguousarray(m_act.transpose(1, 0, 2)).reshape(m_act.shape[1], -1)
        rhs = np.ascontiguousarray(b_gen.transpose(1, 0, 2)).reshape(n_gen, -1)
        return lhs @ rhs.T

    def matvec(self, kappa_vec: np.ndarray) -> np.ndarray:
        """Hessian applied to a packed rotation vector, in the gradient's own representation."""
        self.n_calls += 1
        sp = self.sp
        n = self.ints.n_orb
        kappa = _unpack(kappa_vec, self.rows, self.cols, n)
        if self.mo_square:
            with timer("Hessian-vector product"):
                d_f = self._response_square(kappa)
            d_g = d_f.conj().T - d_f
            raw = 2.0 * np.conj(_pack(d_g, self.rows, self.cols))
            curv = -0.5 * (kappa @ self.g_dual - self.g_dual @ kappa)
            return raw - 2.0 * _pack(curv, self.rows, self.cols)
        x = self.c @ kappa                                        # orbital response

        with timer("Hessian-vector product"):
            x_i = np.ascontiguousarray(x[:, sp.inactive])
            x_t = np.ascontiguousarray(x[:, sp.active])
            # Transition densities, as ONE-SIDED low-rank products, Hermitized afterwards:
            # the response density is A + A^dag with A = X C^dag, and J/K are linear maps
            # taking A^dag to the Hermitian adjoint of what they give A (real AO
            # integrals), so building A's half and adding the adjoint costs half the half
            # transforms and half the pair contraction of building the symmetrized density
            # [X | C], [C | X] directly. The active A uses gamma Hermitian (the rdm gate):
            # c_t gamma^T x_t^dag = (x_t gamma^T c_t^dag)^dag. The kappa-independent
            # right-hand half transforms come from the cache built once in __init__.
            j_i, k_i = coulomb_exchange(self.factors, x_i, orbitals_right=self.c_i,
                                        right_half=self._w_i)
            j_a, k_a = coulomb_exchange(self.factors,
                                        np.ascontiguousarray(x_t @ self.gamma.T),
                                        orbitals_right=self.c_t, right_half=self._w_t)
            g_i = j_i - k_i
            g_i = g_i + g_i.conj().T
            g_a = j_a - k_a
            g_a = g_a + g_a.conj().T
            d_f_i = self._fock_mo(self.f_i_ao, x) + self.c.conj().T @ g_i @ self.c
            d_f_a = self._fock_mo(self.f_a_ao, x) + self.c.conj().T @ g_a @ self.c

            # Three-index response. Bra side: free from b_act — but it must be a *batched
            # GEMM*, not an einsum. `einsum("rp,Prt->Ppt", ...)` is 1.2e9 flops that NumPy
            # will not dispatch to BLAS, and it dominated the Hessian-vector product
            # (measured at 16 s per product on a real system before this was written as
            # matmul). Same trap as the J/K build; coefficients and densities transform oppositely.
            # ⚠ Three arrays of b_act's shape are live at the addition below — this one, the
            # transform's result, and the sum. That peak is what
            # ``hessian_response_memory_gb`` states and what a memory plan carries for the
            # second-order path; keep the two in step if this expression changes.
            bx = np.matmul(kappa.conj().T, self.b_act)
            bx = bx + transform_3c(self.factors, self.c, x_t)

            d_f = np.zeros((n, n), dtype=np.complex128)
            d_f[sp.inactive, :] = (d_f_i + d_f_a)[:, sp.inactive].T
            act = sp.active
            mx = self._contract_gamma2(bx[:, act, :])
            d_f[act, :] = (self.gamma @ d_f_i[:, act].T
                           + self._pair_contract(bx, self.mc)
                           + self._pair_contract(self.b_act, mx))
        d_g = d_f.conj().T - d_f
        raw = 2.0 * np.conj(_pack(d_g, self.rows, self.cols))
        # Remove the chart-curvature term: it *is* the antisymmetric part of the raw operator,
        # so subtracting it leaves the true (symmetric) Hessian. Already a dual element, so it
        # is packed without the extra conjugation the primal gradient needs.
        curv = -0.5 * (kappa @ self.g_dual - self.g_dual @ kappa)
        return raw - 2.0 * _pack(curv, self.rows, self.cols)

    __call__ = matvec

    def _response_square(self, kappa: np.ndarray) -> np.ndarray:
        """The response Fock matrix ``d_f``, built from the held MO factor square.

        The fixed-factor formulation (the adaptation of Werner & Meyer's and Werner &
        Knowles' fixed-integral micro-iterations — J. Chem. Phys. 73, 2342 (1980);
        J. Chem. Phys. 82, 5053 (1985) — to three-index factors; see the class docstring):
        with ``B^P_pq`` held, the MO-projected Coulomb and exchange responses of a density
        perturbation ``dD`` are

            J(dD)_pq = sum_P B^P_pq Tr(B^P dD),      K(dD)_pq = sum_P (B^P dD B^P)_pq,

        and the two response densities are commutators, ``dD_I = [kappa, P_I]`` and
        ``dD_A = [kappa, Ge]`` with ``P_I`` the projector on the inactive columns and
        ``Ge`` gamma^T embedded in the active block (gamma Hermitian, the rdm gate). The
        response Focks are needed only at **occupied columns** — the generalized Fock has
        no virtual rows — which is what keeps every contraction here at
        ``O(naux n^2 n_occ)`` dense GEMMs with no AO pass and no half transform. The
        ket-side three-index response is ``B kappa`` restricted to active columns, so the
        per-product ``transform_3c`` of the AO route disappears too.
        """
        sp = self.sp
        n = self.ints.n_orb
        inact, act = sp.inactive, sp.active
        n_i, n_t = inact.size, act.size
        n_occ = n_i + n_t
        b = self.b_full
        naux = b.shape[0]
        b_occ_t = self.b_occ_t                                    # (n, naux, n_occ)

        # --- inactive response: dD_I = [kappa, P_I] ---------------------------------
        # kappa's inactive-inactive block is never a parameter, so with the occupied
        # orbitals at the leading indices the contractions can skip the zero rows and
        # columns — about 40% of these two GEMMs on a typical inactive fraction. The
        # dense forms below them are the general fallback (arbitrary index layout).
        if self._occ_prefix:
            b2 = b.reshape(naux * n, n)
            y_i = (b2[:, n_i:] @ np.ascontiguousarray(kappa[n_i:, :n_i])
                   ).reshape(naux, n, n_i)
            z_i = (np.ascontiguousarray(kappa[:n_i, n_i:])
                   @ b_occ_t[n_i:].reshape(n - n_i, naux * n_occ)
                   ).reshape(n_i, naux, n_occ)
        else:
            k_i = np.ascontiguousarray(kappa[:, inact])           # (n, n_i)
            y_i = (b.reshape(-1, n) @ k_i).reshape(naux, n, n_i)  # (B kappa P_I)
            z_i = (np.ascontiguousarray(kappa[inact, :])
                   @ b_occ_t.reshape(n, naux * n_occ)).reshape(n_i, naux, n_occ)
        y_i_t = np.ascontiguousarray(y_i.transpose(1, 0, 2)).reshape(n, naux * n_i)
        t_i = (y_i[:, inact, np.arange(n_i)].sum(axis=1)
               - z_i[np.arange(n_i), :, np.arange(n_i)].sum(axis=0))   # Tr(B^P dD_I)
        z_i_flat = np.ascontiguousarray(z_i.transpose(1, 0, 2)).reshape(naux * n_i, n_occ)
        m_i = ((b_occ_t * t_i[None, :, None]).sum(axis=1)         # J
               - y_i_t @ self.b_i_occ_flat                        # - K, first commutator half
               + self.b_p_i_flat @ z_i_flat)                      # - K, second half

        # --- active response: dD_A = [kappa, Ge] ------------------------------------
        if self._occ_prefix and not self._has_act_act:
            # kappa's active-active block is zero too: the active columns' nonzero rows
            # are the inactive prefix and the virtual tail, each a contiguous slab.
            b2 = b.reshape(naux * n, n)
            y_a = (b2[:, :n_i] @ np.ascontiguousarray(kappa[:n_i, n_i:n_occ])
                   + b2[:, n_occ:] @ np.ascontiguousarray(kappa[n_occ:, n_i:n_occ])
                   ).reshape(naux, n, n_t)
            z_a = (np.ascontiguousarray(kappa[n_i:n_occ, :n_i])
                   @ b_occ_t[:n_i].reshape(n_i, naux * n_occ)
                   + np.ascontiguousarray(kappa[n_i:n_occ, n_occ:])
                   @ b_occ_t[n_occ:].reshape(n - n_occ, naux * n_occ)
                   ).reshape(n_t, naux, n_occ)
        else:
            k_a = np.ascontiguousarray(kappa[:, act])             # (n, n_t)
            y_a = (b.reshape(-1, n) @ k_a).reshape(naux, n, n_t)  # (B kappa P_A)
            z_a = (np.ascontiguousarray(kappa[act, :])
                   @ b_occ_t.reshape(n, naux * n_occ)).reshape(n_t, naux, n_occ)
        y_ag = y_a @ self.gamma.T                                 # y_a doubles as bx below
        y_ag_t = np.ascontiguousarray(y_ag.transpose(1, 0, 2)).reshape(n, naux * n_t)
        z_ag = np.tensordot(self.gamma.T, z_a, axes=([1], [0]))   # (n_t, naux, n_occ)
        t_a = (y_ag[:, act, np.arange(n_t)].sum(axis=1)
               - z_ag[np.arange(n_t), :, n_i + np.arange(n_t)].sum(axis=0))
        z_ag_flat = np.ascontiguousarray(
            z_ag.transpose(1, 0, 2)).reshape(naux * n_t, n_occ)
        m_a = ((b_occ_t * t_a[None, :, None]).sum(axis=1)
               - y_ag_t @ self.b_a_occ_flat
               + self.b_p_a_flat @ z_ag_flat)

        # One-electron parts: [F^MO, kappa], with the MO Fock matrices already in hand.
        occ = self._occ_cols
        d_f_i = np.zeros((n, n), dtype=np.complex128)
        d_f_a = np.zeros((n, n), dtype=np.complex128)
        d1_i = self.ints.f_inactive @ kappa - kappa @ self.ints.f_inactive
        d1_a = self.f_act_mo @ kappa - kappa @ self.f_act_mo
        d_f_i[:, occ] = d1_i[:, occ] + m_i
        d_f_a[:, occ] = d1_a[:, occ] + m_a

        # Three-index response of b_act: bra side from b_act, ket side is (B kappa)_act.
        bx = np.matmul(kappa.conj().T, self.b_act) + y_a

        d_f = np.zeros((n, n), dtype=np.complex128)
        d_f[inact, :] = (d_f_i + d_f_a)[:, inact].T
        mx = self._contract_gamma2(bx[:, act, :])
        d_f[act, :] = (self.gamma @ d_f_i[:, act].T
                       + self._pair_contract(bx, self.mc)
                       + self._pair_contract(self.b_act, mx))
        return d_f

    def exact_diagonal(self) -> np.ndarray:
        """The exact diagonal of this Hessian over the rotation pairs, as a preconditioner.

        The Fock-difference model (:func:`diagonal_hessian`) is a *preconditioner*, measured
        ~0.67x too small on average and up to 10x on soft modes — and the Davidson expansion
        count is set by exactly those modes. This is the analytic diagonal, derived as the
        Wirtinger curvature ``2 d^2E / dkappa_pq dconj(kappa_pq)`` (the complex-linear
        diagonal of :meth:`matvec`, i.e. the mean of the two real directions' curvatures;
        the anti-linear part is not representable in a one-number-per-pair preconditioner).
        Per pair class, with ``f = F^I + F^A``, ``F`` the generalized Fock and all values
        real parts:

            virtual-inactive (a,i):  2 [ f_aa - f_ii - (aa|ii) + (ia|ai) ]
            active-inactive  (t,i):  2 [ f_tt - f_ii + gamma_tt F^I_ii - F_tt
                                          - (tt|ii) + (it|ti)
                                          + sum_uv Gamma_ttuv (ii|uv)
                                          + sum_vw Gamma_vttw (vi|iw)
                                          + 2 Re sum_u gamma_tu (ii|tu)
                                          - 2 Re sum_s gamma_ts (is|ti) ]
            virtual-active   (a,t):  2 [ gamma_tt F^I_aa - F_tt
                                          + sum_uv Gamma_ttuv (aa|uv)
                                          + sum_vw Gamma_vttw (va|aw) ]

        (RDM sums over active spinors only; the inactive contractions are already folded
        into ``F^I`` and the explicit ``(..|ii)`` terms.) The derivation is pinned by
        ``tests/test_mcscf.py``, which extracts the complex-linear diagonal from
        :meth:`matvec` on unit vectors and demands agreement to near machine precision —
        the matvec is verified independently against the numerical Hessian, so an error in
        these formulas cannot pass. Active-active pairs (opt-in, redundant for an exact
        CAS) fall back to the unfloored Fock-difference value.

        Every two-electron integral above reduces to the diagonal three-index block and
        exchange-type sums (:func:`~kuiva.integrals.transform.diagonal_pair_blocks`) plus
        contractions of ``b_act`` the Hessian already holds; the streamed pass costs one to
        two matrix-vector products, once per macro-iteration, cached on the instance.
        Values may legitimately be negative far from a minimum; the augmented-Hessian
        solver's floored denominators absorb that, so nothing is clamped here.
        """
        if self._exact_diag is not None:
            return self._exact_diag
        with timer("exact Hessian diagonal"):
            sp, ints = self.sp, self.ints
            n = ints.n_orb
            act, inact = sp.active, sp.inactive
            nact = self.nact
            rows, cols = self.rows, self.cols
            fia_d = np.real(np.diag(ints.f_inactive + self.f_act_mo))
            fi_d = np.real(np.diag(ints.f_inactive))
            f_gen = generalized_fock(ints, self.gamma, self.gamma2, self.f_act_mo)
            fg_d = np.real(np.diag(f_gen))
            g_tt = np.clip(np.real(np.diag(self.gamma)), 0.0, None)
            occ = np.zeros(n)
            occ[inact] = 1.0
            occ[act] = np.real(np.diag(self.gamma))

            if self.mo_square:
                # Everything the streamed pass would compute is a reduction over blocks
                # the fixed-factor route already holds.
                b_diag = np.ascontiguousarray(
                    np.real(self.b_full[:, np.arange(n), np.arange(n)]))
                s2v = (np.abs(self.b_occ_t[sp.virtual, :, :inact.size]) ** 2).sum(axis=1)
            else:
                # The streamed pass reuses the occupied half transforms the response
                # builds already cached, and computes the exchange sums only for the
                # virtual rows — the active rows' exchange sums come from the resident
                # b_act below, and the occupied columns' half transforms would be
                # recomputation.
                half = (self._occ_cols, self._w_occ) if self._w_occ is not None else None
                b_diag, s2v = diagonal_pair_blocks(self.factors, self.c, inact,
                                                   rows=sp.virtual, half=half)
            # Exchange-type sums against the active orbitals, from the resident b_act.
            s2t = (np.abs(self.b_act) ** 2).sum(axis=0)              # (n, nact)
            # sum_uv Gamma_ttuv (qq|uv) = sum_P B^P_qq M^P_tt, with M the Gamma-contracted
            # intermediate the matvec already holds.
            mcd = self.mc[:, np.arange(nact), np.arange(nact)]       # (naux, nact)
            g1 = b_diag.T @ mcd                                      # (n, nact)
            # sum_vw Gamma_vttw (vq|qw) = sum_{P,vw} Gamma_vttw conj(B^P_qv) B^P_qw.
            n_pair = np.empty((n, nact), dtype=np.float64)
            for t in range(nact):
                g2t = np.ascontiguousarray(self.gamma2[:, t, t, :])  # Gamma_{v t t w}
                y = self.b_act @ g2t.T                               # (naux, n, nact)
                n_pair[:, t] = np.real((np.conj(self.b_act) * y).sum(axis=(0, 2)))
            # 2 Re sum_u gamma_tu (ii|tu) and -2 Re sum_s gamma_ts (is|ti).
            bta = self.b_act[:, act, :]                              # (naux, nact, nact)
            t_p = (self.gamma[None, :, :] * bta).sum(axis=-1)        # (naux, nact)
            term_tt = 2.0 * np.real(b_diag[:, inact].T @ t_p)        # (n_inact, nact)
            b_i = self.b_act[:, inact, :]                            # (naux, n_inact, nact)
            v = b_i @ self.gamma.T
            term_u = -2.0 * np.real((v * np.conj(b_i)).sum(axis=0))  # (n_inact, nact)

            loc = np.zeros(n, dtype=int)
            loc[inact] = np.arange(inact.size)
            loc[act] = np.arange(act.size)
            vloc = np.zeros(n, dtype=int)
            vloc[sp.virtual] = np.arange(sp.virtual.size)
            sid = np.zeros(n, dtype=int)
            sid[act] = 1
            sid[sp.virtual] = 2

            a_diag = np.empty(rows.size, dtype=np.float64)
            ppqq = (b_diag[:, rows] * b_diag[:, cols]).sum(axis=0)
            m_ai = (sid[rows] == 2) & (sid[cols] == 0)
            m_ti = (sid[rows] == 1) & (sid[cols] == 0)
            m_at = (sid[rows] == 2) & (sid[cols] == 1)
            m_tt = (sid[rows] == 1) & (sid[cols] == 1)
            r, c = rows[m_ai], cols[m_ai]
            a_diag[m_ai] = (fia_d[r] - fia_d[c] - ppqq[m_ai]
                            + s2v[vloc[r], loc[c]])
            r, c = rows[m_ti], cols[m_ti]
            lt, li = loc[r], loc[c]
            a_diag[m_ti] = (fia_d[r] - fia_d[c] + g_tt[lt] * fi_d[c] - fg_d[r]
                            - ppqq[m_ti] + s2t[c, lt]
                            + np.real(g1[c, lt]) + n_pair[c, lt]
                            + term_tt[li, lt] + term_u[li, lt])
            r, c = rows[m_at], cols[m_at]
            lt = loc[c]
            a_diag[m_at] = (g_tt[lt] * fi_d[r] - fg_d[c]
                            + np.real(g1[r, lt]) + n_pair[r, lt])
            if np.any(m_tt):                       # opt-in active-active: model value only
                r, c = rows[m_tt], cols[m_tt]
                a_diag[m_tt] = (occ[c] - occ[r]) * (fia_d[r] - fia_d[c])
            self._exact_diag = 2.0 * a_diag
        return self._exact_diag


def diagonal_hessian(ints: CASIntegrals, gamma: np.ndarray, f_active: np.ndarray,
                     rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Approximate diagonal Hessian for the rotation pairs — a **preconditioner** (see above).

    ``H_pq ~ 2 (D_pp - D_qq) (f_qq - f_pp)`` with ``f = F^I + F^A`` and ``D`` the total
    occupations. Positive whenever the orbital ordering is sane, which is exactly the
    condition under which a Newton-like step is meaningful.
    """
    sp = ints.spaces
    n = ints.n_orb
    occ = np.zeros(n)
    occ[sp.inactive] = 1.0
    occ[sp.active] = np.clip(np.real(np.diag(gamma)), 0.0, 1.0)
    fdiag = np.real(np.diag(ints.f_inactive + f_active))
    hdiag = 2.0 * (occ[cols] - occ[rows]) * (fdiag[rows] - fdiag[cols])
    n_bad = int(np.count_nonzero(hdiag < 0.0))
    if n_bad:
        log.debug("%d of %d rotation pairs have a negative approximate curvature; the step "
                  "is preconditioned with the floored magnitude", n_bad, hdiag.size)
    hdiag = np.abs(hdiag)
    floor = HESSIAN_FLOOR
    if hdiag.size:
        floor = max(floor, HESSIAN_RELATIVE_FLOOR * float(np.median(hdiag)))
    return np.maximum(hdiag, floor)


# --- Rotations ----------------------------------------------------------------------------

@dataclass
class AHResult:
    """Outcome of an augmented-Hessian solve.

    The step solves ``(H + shift - eigenvalue) x = -g``: ``eigenvalue`` is the automatic level
    shift from the augmented formulation, ``shift`` the additional one imposed to respect the
    trust radius. Both are reported so the model that was actually solved is checkable rather
    than implicit — a step is only meaningful together with the model it came from.
    """
    step: np.ndarray
    eigenvalue: float
    residual: float
    n_matvec: int
    converged: bool
    shift: float = 0.0
    #: The Krylov subspace the solve built, as ``(basis, applied)`` lists of
    #: ``(alpha, x)`` pairs. Returned so a caller that must re-solve **at the same point**
    #: with a different trust radius (a trust-region rejection retry) can pass it back as
    #: ``subspace=`` and pay zero Hessian-vector products. It is a statement about one
    #: gradient and one Hessian; reusing it after the orbitals have moved is wrong.
    subspace: Optional[tuple] = None
    #: The lowest few Ritz vectors of the final subspace (normalized ``x`` parts), when
    #: ``n_ritz_out`` asked for them. These are the soft directions a diagonal
    #: preconditioner is worst at rediscovering; the Hessian moves slowly between
    #: macro-iterations, so seeding the *next* solve with them (``recycle=``) trades one
    #: Hessian-vector product per seed for the expansions that would have re-found them.
    ritz_vectors: Optional[list] = None


def augmented_hessian_step(grad: np.ndarray, hvp: Callable[[np.ndarray], np.ndarray],
                           hdiag: np.ndarray, *, max_iter: int = 24, tol: float = 1e-3,
                           max_subspace: int = 100,
                           trust: Optional[float] = None,
                           guess: Optional[np.ndarray] = None,
                           step_tol: float = 0.1,
                           subspace: Optional[tuple] = None,
                           recycle: Optional[Sequence[np.ndarray]] = None,
                           n_ritz_out: int = 0) -> AHResult:
    """Newton step from the augmented-Hessian eigenproblem, solved by Davidson.

    Solves for the lowest eigenpair of

        [ 0    g^T ] [ 1 ]        [ 1 ]
        [ g    H   ] [ x ]  = mu  [ x ]

    whose eigenvector gives the step ``x``. The augmented formulation is preferred over
    solving ``H x = -g`` directly because the eigenvalue ``mu`` acts as an **automatic level
    shift**: it lies below the lowest eigenvalue of ``H``, so the step is well defined and
    downhill even when ``H`` is indefinite — which it is whenever the orbitals are far from a
    minimum, exactly when a raw Newton step would run to a saddle.

    Only Hessian-vector products are used, so nothing of size ``n_param^2`` is ever formed.
    The projected matrix is **symmetrized** before the Ritz step, which is what makes this
    exact for the true Hessian despite the asymmetry of :class:`OrbitalHessian` — see its
    docstring.

    ``hdiag`` preconditions the expansion, as in any Davidson.

    ``trust`` caps the largest rotation angle. It is enforced by **raising the level shift**,
    not by scaling the converged step down: a truncated Newton step is no longer the solution
    of any model, whereas ``(H - mu - sigma) x = -g`` for a larger ``sigma`` is exactly the
    Newton step of a more conservative model, and stays second-order informed. The shift is
    found by bisection **inside the Krylov subspace already built**, so it costs no additional
    Hessian-vector products — the projected matrix is ``S + sigma P`` with ``P`` the Gram
    matrix of the step components, and only the small eigenproblem is re-solved.

    ``tol`` is **relative to the gradient norm**. An absolute tolerance is actively harmful
    here: the initial residual is ``|g|`` itself, so once the gradient is small any absolute
    threshold is met before a single Hessian-vector product is taken, the solver returns a
    **zero step**, and the optimization stalls at precisely the point where Newton should be
    taking over. (Observed, before this was fixed: the gradient parked at 4e-4 while the trust
    region collapsed through five successive rejections.) The caller chooses ``tol``;
    :class:`OrbitalOptimizer` supplies an inexact-Newton forcing sequence.

    ``step_tol`` terminates a **trust-capped** solve early. While the Ritz step exceeds the
    trust radius, what the caller will actually take is the *shifted* step, and refining the
    Krylov subspace beyond the point where that shifted step has stopped moving only polishes
    the part of the Newton direction the shift then discards — measured on a heavy-element
    CASSCF far from convergence, most of the Hessian-vector products of the early
    macro-iterations went exactly there. So once two successive shifted steps agree to
    ``step_tol`` (relative), the solve stops and returns the shifted step: still the exact
    Newton step of the more conservative shifted model, at a fraction of the products. The
    eigen-residual test is untouched, so behaviour near convergence — where the cap is
    inactive — is unchanged. ``step_tol=0`` disables the truncation.

    ``subspace`` re-solves in a previously returned ``AHResult.subspace`` with **no**
    expansions and no Hessian-vector products: the projected problem is rebuilt, shifted to
    the (new) trust radius, and returned. This is for the one situation where it is exact —
    the caller rejected the step and retries at the *same* orbitals and RDMs with a smaller
    trust radius, where the gradient and Hessian are bitwise those the subspace was built
    from.

    ``recycle`` seeds the subspace with additional vectors — the previous macro-iteration's
    lowest Ritz vectors (``AHResult.ritz_vectors``, requested via ``n_ritz_out``). Unlike
    ``subspace`` this is **not** a replay: the Hessian has moved with the orbitals, so each
    seed is re-applied (one Hessian-vector product) and enters as an ordinary expansion.
    The point is which directions get into the subspace first: the softest curvature
    directions dominate the Newton step, move slowly between macro-iterations, and are
    precisely what a diagonal preconditioner rediscovers most slowly. Seeds nearly parallel
    to what is already in the basis are dropped by the orthogonalization, so a redundant
    seed costs nothing.
    """
    m = grad.size
    gnorm = float(np.linalg.norm(grad))
    if m == 0 or gnorm == 0.0:
        return AHResult(np.zeros_like(grad), 0.0, 0.0, 0, True)
    target = tol * gnorm

    def dot(a1, x1, a2, x2):
        return float(a1 * a2 + np.real(np.vdot(x1, x2)))

    def ritz(basis, applied):
        """Lowest Ritz pair of the symmetrized projection: ``(lam, y_a, y_x, resid)``."""
        k = len(basis)
        s = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                s[i, j] = dot(basis[i][0], basis[i][1], applied[j][0], applied[j][1])
        s = 0.5 * (s + s.T)                           # exact projection of the true Hessian
        vals, vecs = np.linalg.eigh(s)
        c = vecs[:, 0]
        lam = float(vals[0])
        y_a = sum(c[i] * basis[i][0] for i in range(k))
        y_x = sum(c[i] * basis[i][1] for i in range(k))
        ay_a = sum(c[i] * applied[i][0] for i in range(k))
        ay_x = sum(c[i] * applied[i][1] for i in range(k))
        r_a = ay_a - lam * y_a
        r_x = ay_x - lam * y_x
        resid = float(np.sqrt(r_a ** 2 + np.real(np.vdot(r_x, r_x))))
        return lam, y_a, y_x, resid, r_a, r_x

    if subspace is not None:
        # Rejection retry: same point, smaller trust. Zero Hessian-vector products.
        basis, applied = subspace
        lam, y_a, y_x, resid, _, _ = ritz(basis, applied)
        step = y_x / y_a if abs(y_a) > 1e-12 else np.zeros_like(grad)
        shift = 0.0
        if trust is not None and step.size and float(np.max(np.abs(step))) > trust:
            step, lam, shift = _shift_to_trust(basis, applied, trust, lam)
        return AHResult(step=step, eigenvalue=lam, residual=resid, n_matvec=0,
                        converged=True, shift=shift, subspace=(basis, applied))

    # Start from the "no step" direction; the first expansion is the preconditioned gradient.
    basis = [(1.0, np.zeros_like(grad))]
    applied = [(0.0, grad.copy())]                    # A applied to the first basis vector
    n_matvec = 0
    if guess is not None and np.any(guess):
        # Warm start from the previous macro-iteration's step. The solution moves slowly
        # between iterations, so this seeds the subspace with the soft directions that are
        # expensive to rediscover — and those are exactly the ones a diagonal preconditioner
        # is worst at finding. Measured: it is the difference between the Davidson converging
        # and hitting its iteration limit on an ill-conditioned Hessian.
        t_x = np.asarray(guess, dtype=np.complex128).copy()
        nrm = np.sqrt(np.real(np.vdot(t_x, t_x)))
        if nrm > 1e-14:
            t_x /= nrm
            basis.append((0.0, t_x))
            applied.append((float(np.real(np.vdot(grad, t_x))), hvp(t_x)))
            n_matvec += 1
    if recycle is not None:
        # Ritz-vector recycling (see the docstring). Each surviving seed costs one product.
        for vec in recycle:
            t_a, t_x = 0.0, np.asarray(vec, dtype=np.complex128).copy()
            for (b_a, b_x) in basis:                  # modified Gram-Schmidt
                ov = dot(b_a, b_x, t_a, t_x)
                t_a -= ov * b_a
                t_x -= ov * b_x
            nrm = np.sqrt(t_a ** 2 + np.real(np.vdot(t_x, t_x)))
            if nrm < 1e-6:                            # already represented: free
                continue
            t_a, t_x = t_a / nrm, t_x / nrm
            basis.append((t_a, t_x))
            applied.append((float(np.real(np.vdot(grad, t_x))), t_a * grad + hvp(t_x)))
            n_matvec += 1
    lam, step, resid = 0.0, np.zeros_like(grad), gnorm
    converged = False
    capped_shift = None                               # set when step_tol truncates the solve
    prev_capped = None                                # previous iteration's shifted step

    for it_ah in range(max_iter):
        k = len(basis)
        lam, y_a, y_x, resid, r_a, r_x = ritz(basis, applied)
        if abs(y_a) > 1e-12:
            step = y_x / y_a
        # At least one expansion always happens: the k == 1 subspace contains only the
        # "no step" vector, so accepting it would return a zero step and stall the caller.
        if (it_ah > 0 and resid < target) or k >= max_subspace:
            converged = resid < target
            break
        # Trust-radius-aware truncation (see the docstring): while the step is capped, stop
        # as soon as the *shifted* step — the one the caller will actually take — has
        # stabilized. The shift search runs on the projected matrices only, so probing it
        # every capped iteration costs no Hessian-vector products.
        if (step_tol > 0.0 and trust is not None and it_ah > 0 and step.size
                and float(np.max(np.abs(step))) > trust):
            st, lm, sg = _shift_to_trust(basis, applied, trust, lam)
            st_norm = float(np.linalg.norm(st))
            if (prev_capped is not None and st_norm > 0.0
                    and float(np.linalg.norm(st - prev_capped)) <= step_tol * st_norm):
                step, lam, capped_shift = st, lm, sg
                converged = True
                break
            prev_capped = st
        # Davidson preconditioning with the diagonal Hessian.
        denom = hdiag - lam
        denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
        t_a = r_a / (-lam if abs(lam) > 1e-8 else -1e-8)
        t_x = r_x / denom
        for (b_a, b_x) in basis:                      # modified Gram-Schmidt
            ov = dot(b_a, b_x, t_a, t_x)
            t_a -= ov * b_a
            t_x -= ov * b_x
        nrm = np.sqrt(t_a ** 2 + np.real(np.vdot(t_x, t_x)))
        if nrm < 1e-10:
            converged = True
            break
        t_a, t_x = t_a / nrm, t_x / nrm
        basis.append((t_a, t_x))
        applied.append((float(np.real(np.vdot(grad, t_x))), t_a * grad + hvp(t_x)))
        n_matvec += 1

    if capped_shift is not None:
        shift = capped_shift                          # the step is already the shifted one
    else:
        shift = 0.0
        if trust is not None and step.size and float(np.max(np.abs(step))) > trust:
            step, lam, shift = _shift_to_trust(basis, applied, trust, lam)
    ritz_out = None
    if n_ritz_out > 0 and len(basis) > 1:
        # Lowest Ritz vectors of the final subspace, for the next solve's recycle= seeds.
        # Projected matrices only — no products. The lowest one is essentially the step
        # direction the caller's warm start already carries; the orthogonalization there
        # makes handing it over anyway harmless.
        k = len(basis)
        s = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                s[i, j] = dot(basis[i][0], basis[i][1], applied[j][0], applied[j][1])
        s = 0.5 * (s + s.T)
        _, vecs = np.linalg.eigh(s)
        ritz_out = []
        for kk in range(min(int(n_ritz_out), k)):
            y = sum(vecs[i, kk] * basis[i][1] for i in range(k))
            nrm = float(np.sqrt(np.real(np.vdot(y, y))))
            if nrm > 1e-12:
                ritz_out.append(y / nrm)
    return AHResult(step=step, eigenvalue=lam, residual=resid, n_matvec=n_matvec,
                    converged=converged, shift=shift, subspace=(basis, applied),
                    ritz_vectors=ritz_out)


def _shift_to_trust(basis, applied, trust: float, lam0: float):
    """Raise the level shift until the augmented-Hessian step fits the trust radius.

    Operates entirely on the small projected matrices, so it is free in Hessian-vector
    products. Returns ``(step, eigenvalue, shift)``.
    """
    k = len(basis)

    def dot(a1, x1, a2, x2):
        return float(a1 * a2 + np.real(np.vdot(x1, x2)))

    s = np.zeros((k, k))
    p = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            s[i, j] = dot(basis[i][0], basis[i][1], applied[j][0], applied[j][1])
            p[i, j] = float(np.real(np.vdot(basis[i][1], basis[j][1])))
    s = 0.5 * (s + s.T)

    def solve(sigma):
        vals, vecs = np.linalg.eigh(s + sigma * p)
        c = vecs[:, 0]
        y_a = sum(c[i] * basis[i][0] for i in range(k))
        y_x = sum(c[i] * basis[i][1] for i in range(k))
        if abs(y_a) < 1e-12:
            return None, float(vals[0])
        return y_x / y_a, float(vals[0])

    lo, hi = 0.0, 1.0
    for _ in range(40):                                # bracket a shift that is large enough
        st, _ = solve(hi)
        if st is not None and float(np.max(np.abs(st))) <= trust:
            break
        hi *= 4.0
    best, best_lam = solve(hi)
    best_sigma = hi
    for _ in range(40):                                # bisect to the largest usable step
        mid = 0.5 * (lo + hi)
        st, lm = solve(mid)
        if st is not None and float(np.max(np.abs(st))) <= trust:
            best, best_lam, best_sigma, hi = st, lm, mid, mid
        else:
            lo = mid
        if hi - lo < 1e-3 * max(hi, 1.0):
            break
    if best is None:
        return np.zeros_like(basis[0][1]), lam0, 0.0
    return best, best_lam, best_sigma


def unitary_from_antihermitian(kappa: np.ndarray) -> np.ndarray:
    """``exp(kappa)`` for anti-Hermitian ``kappa``, unitary to machine precision.

    Via the Hermitian eigendecomposition of ``i*kappa`` rather than a Pade ``expm``: the
    result is a product of unitary factors and is unitary *exactly*, where ``expm`` leaves an
    error that accumulates over macro-iterations and slowly destroys orbital orthonormality —
    a silent corruption of everything downstream.
    """
    kappa = np.asarray(kappa)
    if kappa.size == 0:
        return np.eye(0, dtype=np.complex128)
    a = 1j * kappa                                   # Hermitian if kappa is anti-Hermitian
    herm_err = float(np.max(np.abs(a - a.conj().T)))
    if herm_err > 1e-10:
        log.warning("rotation generator is not anti-Hermitian (deviation %.2e); the "
                    "resulting transformation would not be unitary", herm_err)
        a = 0.5 * (a + a.conj().T)
    w, v = np.linalg.eigh(a)
    return (v * np.exp(-1j * w)) @ v.conj().T


def _as_optional_vector(value) -> Optional[np.ndarray]:
    """A stored complex vector, or ``None``. HDF5 has no null array, so an absent optional is
    written as an empty one and read back as ``None`` — otherwise a restored optimizer would
    warm start its augmented Hessian from a zero-length guess instead of from nothing."""
    if value is None:
        return None
    array = np.ascontiguousarray(value, dtype=np.complex128)
    return None if array.size == 0 else array


def kramers_parameter_map(rows: np.ndarray, cols: np.ndarray,
                          n_orb: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``(mirror, sign, conjugate)`` mapping each rotation parameter to its time-reversed one.

    Time reversal acts on a spinor *index* as ``pbar = p ^ 1`` with the sign ``t_p = (-1)^p``
    (:func:`kuiva.spinor.expand.time_reversal_index_signs`), so on the rotation generator it
    is the antilinear involution ``(Theta kappa)_pq = t_p t_q conj(kappa[pbar, qbar])``. This
    expresses it on the **packed** parameter vector, which needs three arrays rather than one
    permutation because ``(pbar, qbar)`` may be stored as its anti-Hermitian partner
    ``(qbar, pbar)`` — an active-active pair, where only the upper triangle is a parameter::

        Theta(v)[k] = sign[k] * (conj(v[mirror[k]]) if conjugate[k] else v[mirror[k]])

    Returns ``None`` when the parameter list is **not closed** under the index swap, which is
    the structural precondition for the constraint to exist at all: it holds whenever every
    orbital space contains whole Kramers pairs — the rule that no orbital space may split a
    Kramers pair, so always for a paired reference — and the irrep mask, if any, keeps a
    pair together. It fails loudly rather than silently constraining half of a list.

    ⚠ Nothing here checks that the *orbitals* are Kramers paired; that is a property of an
    array, not of an index list, and is :func:`kuiva.spinor.expand.kramers_pairing_defect`.
    """
    rows = np.asarray(rows, dtype=int).ravel()
    cols = np.asarray(cols, dtype=int).ravel()
    if rows.size == 0:
        return (np.zeros(0, dtype=int), np.zeros(0), np.zeros(0, dtype=bool))
    if int(n_orb) % 2:
        return None
    n = int(n_orb)
    keys = rows * n + cols
    order = np.argsort(keys)
    sorted_keys = keys[order]
    if np.any(np.diff(sorted_keys) == 0):                    # pragma: no cover - defensive
        raise ValueError("the rotation parameter list contains a duplicated (p, q) pair")

    def locate(target):
        pos = np.searchsorted(sorted_keys, target)
        safe = np.clip(pos, 0, sorted_keys.size - 1)
        return np.where(sorted_keys[safe] == target, order[safe], -1)

    rbar, cbar = rows ^ 1, cols ^ 1
    direct = locate(rbar * n + cbar)
    flipped = locate(cbar * n + rbar)
    mirror = np.where(direct >= 0, direct, flipped)
    if np.any(mirror < 0):
        return None
    conjugate = direct >= 0
    # t_p t_q, and the extra minus the anti-Hermitian storage of a flipped pair carries.
    phase = np.where((rows % 2) == (cols % 2), 1.0, -1.0)
    sign = np.where(conjugate, phase, -phase)
    return mirror, sign, conjugate


def kramers_project(vec: np.ndarray,
                    kmap: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Project a packed rotation vector onto the time-reversal-**even** subspace.

    ``(v + Theta v) / 2`` through :func:`kramers_parameter_map`. ``Theta`` is orthogonal for
    the real inner product :meth:`OrbitalOptimizer._dot` uses, so this is an orthogonal
    projection of the real parameter space and the constrained optimization is an ordinary
    restriction of the unconstrained one — not a modification of the step.
    """
    mirror, sign, conjugate = kmap
    if vec.size == 0:
        return vec
    partner = vec[mirror]
    return 0.5 * (vec + sign * np.where(conjugate, np.conj(partner), partner))


def kramers_mirror_mean(vec: np.ndarray,
                        kmap: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Average a **real, positive** packed quantity over time-reversed parameter pairs.

    For the diagonal Hessian preconditioner, whose entries are equal between a parameter and
    its time-reversed partner for a time-reversal-symmetric problem and differ from it only by
    roundoff. Averaging costs nothing and keeps the preconditioned direction inside the even
    subspace exactly, where an unaveraged one would leak a little of what the projection is
    there to remove back in at every scaling.
    """
    mirror, _sign, _conjugate = kmap
    return vec if vec.size == 0 else 0.5 * (vec + vec[mirror])


def resolve_kramers_rotation(request: Any, c_spinor: np.ndarray, spaces: OrbitalSpaces, *,
                             active_active: bool = False,
                             labels: Optional[np.ndarray] = None) -> Tuple[bool, float]:
    """Decide whether to constrain the rotation, and say what the decision was made on.

    ``request`` is ``"auto"`` (the default), ``True`` or ``False``; the answer is
    ``(constrained, pairing defect)``.

    ⚠ **"auto" measures the orbitals rather than trusting a flag, and that is the whole
    point.** The constraint is expressed on the index convention ``(2p, 2p+1)`` = ``(psi,
    T psi)``, so it is the right constraint exactly when the incoming columns *are* those
    pairs and is an arbitrary restriction of the variational space when they are not — which
    is not an edge case: an unrestricted reference's spinors are never paired (see
    :func:`kuiva.spinor.expand.kramers_pairing_defect`), and neither are the active orbitals
    a previous *unconstrained* CASSCF converged to. A flag would be right about the first
    calculation and wrong about the restart.

    ⚠ **The active electron count no longer decides this** (it did between v0.36.0 and
    v0.37.0, when the constraint was imposed on odd counts only). A time-reversal-broken
    solution is a legitimate variational answer at an even count — measured on N2 CAS(6,8),
    0.64 Eh **below** the symmetric stationary point with its density 42% time-odd — so the
    constraint could not simply be imposed there; but declining it left every even-electron
    run exposed to the drift it exists to remove, and the parity of the electron count is
    evidence about neither. What decides it now is a **measurement at the converged point**
    (:func:`measure_time_odd_curvature`): the optimization is constrained, its solution is
    tested for negative curvature along the time-odd rotations the constraint forbade, and
    the constraint is *released* where that instability is real. Parity survives only as the
    rule for when a release is on the table (``optimize_orbitals``): at an odd count a broken
    solution has no Kramers degeneracy and the state-averaging gate refuses it downstream
    anyway, so there is nothing to release to.

    ``True`` refuses rather than constrains an unpaired set: an explicit request that quietly
    did nothing (or quietly did the wrong thing) is worse than an error.
    """
    from ..spinor.expand import kramers_pairing_defect

    if request is False:
        return False, float("nan")
    if request is not True and request != "auto":
        raise ValueError("kramers_rotation must be True, False or 'auto'; got {!r}"
                         .format(request))
    c = np.asarray(c_spinor)
    paired = c.ndim == 2 and c.shape[1] % 2 == 0 and c.shape[0] % 2 == 0
    pairing = kramers_pairing_defect(c) if paired else float("inf")
    rows, cols = spaces.rotation_pairs(active_active, labels=labels)
    kmap = kramers_parameter_map(rows, cols, spaces.n_orb) if paired else None
    usable = pairing <= KRAMERS_PAIRING_TOL and kmap is not None
    if usable:
        return True, pairing
    if request is True:
        if kmap is None:
            raise ValueError(
                "kramers_rotation=True needs a rotation parameter list closed under the "
                "Kramers index swap: an orbital space holds half a pair, or an irrep mask "
                "separates the partners of one")
        raise ValueError(
            "kramers_rotation=True needs Kramers-paired orbitals, and these are paired only "
            "to {:.2e} (against {:.1e}): with an unrestricted reference they never are, and a "
            "set converged by an unconstrained optimization is not either. Pass "
            "kramers_rotation='auto', which measures this and leaves the rotation "
            "unconstrained where the constraint would be meaningless"
            .format(pairing, KRAMERS_PAIRING_TOL))
    log.debug("the rotation is left unconstrained: the orbitals are Kramers paired only to "
              "%.2e (against %.1e)%s", pairing, KRAMERS_PAIRING_TOL,
              "" if kmap is not None else ", and the parameter list is not pair closed")
    return False, pairing


def kramers_rotation_note(kramers: bool, pairing: float) -> str:
    """One phrase for the reported rotation line: what the constraint decision rests on.

    The two ways a run comes back unconstrained are different statements and are never
    printed the same way: it was asked for, or the orbitals are not Kramers pairs to begin
    with. Whether a *constrained* run then keeps its constraint is a separate line, written
    at the end of the optimization from :class:`TimeOddCurvature` — it is a measurement at
    the converged point and cannot be known here.
    """
    if kramers:
        return "orbitals paired to {:.0e}; spaces stay time-reversal closed".format(pairing)
    if pairing != pairing:                                    # NaN: switched off explicitly
        return "unconstrained by request"
    return ("orbitals are not Kramers paired ({:.0e})".format(pairing)
            if pairing < np.inf else "not a Kramers-paired orbital set")


def kramers_odd_project(vec: np.ndarray,
                        kmap: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Project a packed rotation vector onto the time-reversal-**odd** subspace.

    ``(v - Theta v) / 2``, the orthogonal complement of :func:`kramers_project` in the same
    real inner product. It is the subspace the constraint removes, and therefore the one a
    stability test has to look in: a constrained optimization is stationary against every
    even rotation by construction, so whether its solution is a minimum of the *unconstrained*
    problem is decided entirely here.
    """
    mirror, sign, conjugate = kmap
    if vec.size == 0:
        return vec
    partner = vec[mirror]
    return 0.5 * (vec - sign * np.where(conjugate, np.conj(partner), partner))


@dataclass
class TimeOddCurvature:
    """Lowest curvature of the orbital Hessian along time-reversal-**odd** rotations.

    ``value`` is a Ritz value and therefore an **upper bound** on the true lowest eigenvalue,
    which is what makes the verdict one-sided and safe: a negative ``value`` *proves* negative
    curvature whether or not the eigensolve converged, while a non-negative one establishes
    stability only when ``converged`` is true. Both readings are reported as what they are.
    """

    value: float
    direction: np.ndarray
    converged: bool
    n_matvec: int

    @property
    def unstable(self) -> bool:
        """The symmetric solution is a saddle: a time-odd rotation lowers the energy."""
        return self.value < -KRAMERS_STABILITY_TOL

    @property
    def verdict(self) -> str:
        """One phrase for the reported line — never "stable" on an unconverged solve."""
        if self.unstable:
            return "SADDLE: a time-odd rotation lowers the energy"
        if not self.converged:
            return "not established: the eigensolve did not converge"
        return "the time-reversal-symmetric solution is a minimum"


def lowest_projected_curvature(hvp: Callable[[np.ndarray], np.ndarray], hdiag: np.ndarray,
                               project: Callable[[np.ndarray], np.ndarray], *,
                               tol: float = 1e-3, max_iter: int = 50,
                               max_subspace: int = 64, n_seed: int = 3
                               ) -> Tuple[float, np.ndarray, int, bool]:
    """Lowest eigenpair of a symmetric operator **restricted to a subspace**, by Davidson.

    ``project`` is an orthogonal projector of the real parameter space (here
    :func:`kramers_odd_project`); every basis vector and every product is passed through it,
    so what is diagonalized is ``P H P`` on ``range(P)`` — a restriction of the same operator,
    not a modified one. Returns ``(eigenvalue, eigenvector, matvecs, converged)``.

    The seeds are the softest directions of ``hdiag`` — where a negative curvature lives if
    there is one — taken **deterministically** and in both the real and the imaginary
    direction of each parameter, since the projector is real-linear and the two project
    differently. There is no random start: a stability verdict that moved between runs of the
    same script would be worse than none.

    ``tol`` is on the residual, relative to ``max(|lambda|, 1)``. Convergence is only ever
    needed to certify a *non-negative* answer: a negative Ritz value already bounds the true
    lowest eigenvalue from above (see :class:`TimeOddCurvature`).

    ⚠ **The budget is set by the awkward case, not the typical one.** Measured on N2
    CAS(8,12): from the target basis' own SCF orbitals the solve converges in 8-16 products,
    while at the *same* solution reached by projection from another basis it needs ~45 — the
    orbital frame differs by a rotation the exact-diagonal preconditioner is much worse in,
    and the lowest curvatures there sit in a cluster. A budget that fitted the first would
    report "not established" on the second, which is a weaker statement than the run
    deserves. Where it really does stall, that is what it says.
    """
    hdiag = np.ascontiguousarray(np.real(np.asarray(hdiag)), dtype=np.float64)
    n = hdiag.size
    if n == 0:
        return 0.0, np.zeros(0, dtype=np.complex128), 0, True
    basis: List[np.ndarray] = []
    applied: List[np.ndarray] = []
    n_matvec = 0

    def expand(vec: np.ndarray) -> bool:
        """Orthonormalize inside the subspace and apply the operator. False if redundant."""
        v = project(np.asarray(vec, dtype=np.complex128))
        for _ in range(2):                        # twice: the usual Gram-Schmidt insurance
            for b in basis:
                v = v - float(np.real(np.vdot(b, v))) * b
        nrm = float(np.sqrt(np.real(np.vdot(v, v))))
        if nrm < 1e-8:
            return False
        v = v / nrm
        basis.append(v)
        applied.append(project(hvp(v)))
        return True

    for k in np.argsort(hdiag)[:max(1, int(n_seed))]:
        for value in (1.0 + 0.0j, 0.0 + 1.0j):
            seed = np.zeros(n, dtype=np.complex128)
            seed[k] = value
            if expand(seed):
                n_matvec += 1
    if not basis:                                  # the subspace is empty: nothing to break
        return 0.0, np.zeros(n, dtype=np.complex128), 0, True

    lam = 0.0
    vec = np.zeros(n, dtype=np.complex128)
    converged = False
    for _ in range(int(max_iter)):
        k = len(basis)
        s = np.empty((k, k))
        for i in range(k):
            for j in range(k):
                s[i, j] = float(np.real(np.vdot(basis[i], applied[j])))
        s = 0.5 * (s + s.T)                        # exact projection of a symmetric operator
        vals, vecs = np.linalg.eigh(s)
        c = vecs[:, 0]
        lam = float(vals[0])
        vec = sum(c[i] * basis[i] for i in range(k))
        av = sum(c[i] * applied[i] for i in range(k))
        r = av - lam * vec
        resid = float(np.sqrt(np.real(np.vdot(r, r))))
        log.debug("time-odd curvature: subspace %d, lambda %.6e, residual %.3e", k, lam,
                  resid)
        converged = resid <= tol * max(abs(lam), 1.0)
        if converged or k >= max_subspace:
            break
        denom = hdiag - lam
        denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
        if not expand(r / denom):
            # The preconditioned residual lies in the subspace already. ⚠ Not convergence:
            # the residual is orthogonal to the subspace by construction, so an exhausted
            # space has a zero residual and the test above has already fired. This is the
            # preconditioner failing to point anywhere new, and it is reported as the
            # unconverged solve it is — nothing may call an unconverged solve stable.
            break
        n_matvec += 1
    # Deterministic overall sign. ``+v`` and ``-v`` are the same physics — E(kappa) =
    # E(Theta kappa) makes the two branches of a time-odd instability degenerate — but a run
    # that picked them by roundoff would take two different trajectories off the saddle.
    if vec.size:
        lead = int(np.argmax(np.abs(vec)))
        phase = vec[lead]
        if phase.real < 0.0 or (phase.real == 0.0 and phase.imag < 0.0):
            vec = -vec
    return lam, vec, n_matvec, converged


def measure_time_odd_curvature(opt: "OrbitalOptimizer", ints: CASIntegrals,
                               factors: ThreeIndexAO, h_ao: np.ndarray, c_spinor: np.ndarray,
                               gamma: np.ndarray, gamma2: np.ndarray, *,
                               tol: float = 1e-3, max_iter: int = 50) -> TimeOddCurvature:
    """Is the Kramers-constrained solution a minimum, or a saddle in the directions it forbids?

    The orbital analogue of the SCF's stability analysis: build the exact orbital Hessian at
    the converged point and find its lowest eigenvalue **restricted to the time-odd rotations**
    the constraint projected out. It answers the one question the constraint cannot —
    ``kramers_rotation`` decides whether a time-reversal-broken solution is *reachable*, and
    only this decides whether one *exists*.

    ⚠ **It is meaningful at a converged point and nowhere else.** Away from a stationary point
    a negative curvature says nothing about the solution; the caller checks convergence first.

    The Hessian is built at the caller's iterate, reusing the active Fock the optimizer's
    gradient memo already holds, and costs one Hessian-vector product per Davidson expansion
    — a few tens, once per run, against the many hundreds a second-order optimization spends.
    """
    if opt.kmap is None:
        raise ValueError("the time-odd curvature is defined against a Kramers-constrained "
                         "optimization; this one is unconstrained, so the whole rotation "
                         "space was already searched")
    with timer("time-reversal stability"):
        # Memoized on object identity: free at the iterate the driver just evaluated.
        opt.gradient(ints, gamma, gamma2, factors, c_spinor)
        hess = OrbitalHessian(ints, factors, h_ao, c_spinor, gamma, gamma2, opt.rows,
                              opt.cols, f_active_ao=getattr(opt, "_f_active_ao", None))
        kmap = opt.kmap
        value, direction, n_matvec, converged = lowest_projected_curvature(
            hess.matvec, hess.exact_diagonal(),
            lambda v: kramers_odd_project(v, kmap), tol=tol, max_iter=max_iter)
    # Charged to the optimizer's own counter, so the reported product count and
    # :attr:`CASSCFResult.work_units` include what the verdict cost. A measurement whose
    # price is invisible is a measurement nobody can decide to switch off.
    opt.n_hessian_matvec += int(n_matvec)
    return TimeOddCurvature(value=float(value), direction=direction,
                            converged=bool(converged), n_matvec=int(n_matvec))


def kramers_release_rotation(direction: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                             n_orb: int, max_rotation: float = KRAMERS_RELEASE_STEP
                             ) -> np.ndarray:
    """The unitary that steps off a time-odd saddle, scaled to ``max_rotation`` radians.

    A finite step is not a refinement of the optimizer's own: at the saddle the gradient
    vanishes in **every** direction (the exact gradient of a time-reversal-symmetric problem
    is time-even, and the even part is what convergence just drove to zero), so releasing the
    constraint alone would leave the run stationary and it would converge again on the spot.
    What breaks the symmetry is the displacement, and the instability then does the rest.
    """
    if direction.size == 0:
        return np.eye(int(n_orb), dtype=np.complex128)
    scale = float(np.max(np.abs(direction)))
    step = direction * (float(max_rotation) / scale) if scale > 0.0 else direction
    return unitary_from_antihermitian(_unpack(step, rows, cols, int(n_orb)))


def _pack(mat: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return mat[rows, cols]


def _unpack(vec: np.ndarray, rows: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    kappa = np.zeros((n, n), dtype=np.complex128)
    kappa[rows, cols] = vec
    kappa[cols, rows] = -np.conj(vec)
    return kappa


# --- The optimizer -------------------------------------------------------------------------

@dataclass
class OrbitalStep:
    """A *proposed* orbital rotation. Whether it is kept is the caller's decision — see
    :meth:`OrbitalOptimizer.step`, :meth:`~OrbitalOptimizer.accept` and
    :meth:`~OrbitalOptimizer.reject`."""
    kappa: np.ndarray
    unitary: np.ndarray
    energy: float
    grad_norm: float
    max_rotation: float
    trust: float
    second_order: bool = False


class OrbitalOptimizer:
    """Trust-region L-BFGS orbital optimizer. **RDMs in, rotation out**.

    Stateful only in the L-BFGS memory and the trust radius; the orbitals themselves live
    with the caller, which is what lets the same object drive a CI-CASSCF and a DMRG-CASSCF.
    """

    def __init__(self, spaces: OrbitalSpaces, *, max_step: float = DEFAULT_MAX_STEP,
                 memory: int = 10, active_active: bool = False,
                 labels: Optional[np.ndarray] = None,
                 trust: Optional[float] = None, mode: str = "auto",
                 second_order_start: float = SECOND_ORDER_START,
                 stall_patience: int = 3, stall_window: int = STALL_WINDOW,
                 stall_factor: float = STALL_FACTOR,
                 conv_grad: float = 1e-4, ah_tol: float = 1e-3, ah_max_iter: int = 60,
                 ah_recycle: int = 0, kramers: bool = False):
        if mode not in ("auto", "quasi-newton", "second-order"):
            raise ValueError("unknown optimizer mode {!r}; expected 'auto', 'quasi-newton' "
                             "or 'second-order'".format(mode))
        self.spaces = spaces
        self.max_step = float(max_step)
        self.memory = int(memory)
        self.labels = None if labels is None else np.atleast_2d(np.asarray(labels, dtype=int))
        self.rows, self.cols = spaces.rotation_pairs(active_active, labels=self.labels)
        #: Packed time-reversal mirror of the parameter list, or ``None`` when the rotation is
        #: unconstrained. See :meth:`_project` and :func:`kramers_parameter_map`.
        self.kmap = None
        if kramers:
            self.kmap = kramers_parameter_map(self.rows, self.cols, spaces.n_orb)
            if self.kmap is None:
                raise ValueError(
                    "kramers=True needs a rotation parameter list closed under the Kramers "
                    "index swap, and this one is not: an orbital space holds half a pair "
                    "(the spaces must be built on whole pairs, kuiva/spinor/expand.py) or an "
                    "irrep mask separates the partners of one")
        self.trust = float(trust if trust is not None else max_step)
        self.mode = mode
        self.second_order_start = float(second_order_start)
        self.stall_patience = int(stall_patience)
        self.stall_window = int(stall_window)
        self.stall_factor = float(stall_factor)
        self.conv_grad = float(conv_grad)
        self._gnorm_history: List[float] = []
        self.ah_tol = float(ah_tol)
        self.ah_max_iter = int(ah_max_iter)
        #: Ritz vectors carried from one augmented-Hessian solve to the next. ⚠ **Measured
        #: and OFF by default**: with the exact-diagonal preconditioner in place, seeding
        #: cost more products than it saved on the real heavy-element benchmark (+47%) and
        #: on every synthetic tried — the seeds are re-applied against a Hessian that has
        #: moved, and the preconditioned expansion finds the soft directions faster than
        #: the seeds repay. The machinery stays because the measurement was made *with* a
        #: good preconditioner; a future solver with a poor one may re-measure, not
        #: re-implement.
        self.ah_recycle = int(ah_recycle)
        self._ah_ritz: Optional[list] = None
        self._second_order = (mode == "second-order")
        self._stalls = 0
        self.n_hessian_matvec = 0
        self.n_second_order_steps = 0
        self._s: List[np.ndarray] = []
        self._y: List[np.ndarray] = []
        self._prev_grad: Optional[np.ndarray] = None
        self._prev_step: Optional[np.ndarray] = None
        self._prev_energy: Optional[float] = None
        self._pending: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
        self._ah_guess: Optional[np.ndarray] = None
        #: The Hessian and Krylov subspace :meth:`step` built at the current iterate,
        #: keyed by **object identity** of ``(ints, c_spinor, gamma)``. A trust-region
        #: rejection retries at the same iterate, where both are bitwise what they were —
        #: recomputing them is the single largest avoidable cost of a rejection. Cleared
        #: on :meth:`accept` (the iterate moves) and never checkpointed (like
        #: ``_pending``, it is half of an unfinished trial). The gradient has its own
        #: memo (:meth:`gradient`), which *survives* accept — the driver has already
        #: evaluated it at the accepted iterate for the L-BFGS update.
        self._iterate_cache: Optional[dict] = None
        self._grad_cache: Optional[dict] = None
        self.n_rejected = 0

    @property
    def n_parameters(self) -> int:
        """Complex rotation parameters (twice this many real degrees of freedom)."""
        return int(self.rows.size)

    # -- L-BFGS ---------------------------------------------------------------------------
    @staticmethod
    def _dot(a: np.ndarray, b: np.ndarray) -> float:
        """Real inner product on the complex parameter space."""
        return float(np.real(np.vdot(a, b)))

    def _project(self, vec: np.ndarray) -> np.ndarray:
        """The packed vector, constrained to rotations that keep the orbitals Kramers paired.

        The identity when the constraint is off (``kramers=False``), so every call site below
        reads the same with and without it and an unconstrained run is bitwise what it was.
        """
        return vec if self.kmap is None else kramers_project(vec, self.kmap)

    def _two_loop(self, g: np.ndarray, hdiag: np.ndarray) -> np.ndarray:
        """L-BFGS two-loop recursion with the diagonal Hessian as the initial inverse.

        The initial inverse is ``tau / hdiag`` rather than ``1 / hdiag``, with the standard
        self-scaling ``tau = (s.y) / (y. D^-1 .y)`` from the most recent pair. Without it the
        method inherits whatever absolute scale the approximate diagonal happens to have, and
        since that diagonal is a *preconditioner* and not a curvature estimate (see the module
        docstring), the resulting steps are systematically mis-scaled — measured here as
        convergence in ~180 macro-iterations instead of ~20.
        """
        q = g.copy()
        alphas = []
        for s, y in zip(reversed(self._s), reversed(self._y)):
            rho = 1.0 / self._dot(y, s)
            a = rho * self._dot(s, q)
            q = q - a * y
            alphas.append((rho, a, s, y))
        tau = 1.0
        if self._s:
            s, y = self._s[-1], self._y[-1]
            denom = self._dot(y, y / hdiag)
            if denom > 0.0:
                tau = self._dot(s, y) / denom
        r = tau * q / hdiag
        for rho, a, s, y in reversed(alphas):
            b = rho * self._dot(y, r)
            r = r + (a - b) * s
        return r

    def _update_memory(self, grad: np.ndarray) -> None:
        if self._prev_grad is None or self._prev_step is None:
            return
        s = self._prev_step
        y = grad - self._prev_grad
        curv = self._dot(y, s)
        if curv > 1e-12 * max(self._dot(s, s), 1e-30):
            self._s.append(s)
            self._y.append(y)
            if len(self._s) > self.memory:
                self._s.pop(0)
                self._y.pop(0)
        else:
            # Negative or vanishing curvature: keeping the pair would make the L-BFGS inverse
            # indefinite and the step a direction of ascent. Standard remedy is to skip it.
            log.debug("skipped an L-BFGS pair with curvature %.3e", curv)

    def reset_memory(self) -> None:
        """Forget the accumulated curvature (after a rejected step or an active-space change)."""
        self._s.clear()
        self._y.clear()
        self._prev_grad = None
        self._prev_step = None

    def reset_chart(self, *, trust_floor: Optional[float] = None,
                    keep_memory: bool = False) -> None:
        """Discard everything that belongs to the surface just left behind.

        Every piece of state this optimizer accumulates — L-BFGS pairs, the augmented-Hessian
        warm start, the stall counter, the gradient trajectory, the trust radius — is a
        statement about *one* energy surface. When an adaptive solver changes its space
        (:mod:`kuiva.mcscf.events`), that surface no longer exists, and carrying the state
        across is not conservatism but a wrong model: measured on the Ti(2+) benchmark, a run
        kept rejecting steps long after the determinant space had stabilized and the
        surface-to-surface jump was exactly zero, because the memory accumulated across the
        hops had poisoned both the curvature and the trust radius.

        The trust radius is **restored to a floor**, not inherited: the collapse that a run of
        rejections produced was a response to a surface that has been replaced, and starting
        the new chart at 1e-6 rad would spend a dozen iterations climbing back out.

        ``keep_memory`` retains the L-BFGS pairs, transporting curvature across the hop and
        hoping the surfaces are close. Measured, and reset wins: on the truncation benchmark
        with ``mode="quasi-newton"``, resetting gives a 1.5x lower final gradient and **2.6x
        fewer space changes** (5 adoptions in 14 proposals against 13 in 21). Transported
        curvature produces worse steps, which land the iterate where the selection keeps
        changing; the churn is the visible symptom.

        ⚠ **That comparison is meaningless under ``mode="second-order"``**, where the
        augmented-Hessian step never consults the pairs: reset and transport come out bitwise
        identical there, and reading it as "no difference" is a mistake already made once.
        Measure this knob with a step engine that uses the memory.
        """
        if not keep_memory:
            self.reset_memory()
        self._prev_grad = None
        self._prev_step = None
        self._ah_guess = None
        self._ah_ritz = None
        self._pending = None
        self._iterate_cache = None
        self._grad_cache = None
        self._prev_energy = None
        self._stalls = 0
        self._gnorm_history.clear()
        if trust_floor is not None:
            self.trust = min(self.max_step, max(self.trust, float(trust_floor)))

    # -- the step -------------------------------------------------------------------------
    def gradient(self, ints: CASIntegrals, gamma: np.ndarray, gamma2: np.ndarray,
                 factors: ThreeIndexAO, c_spinor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Packed orbital gradient and the diagonal Hessian preconditioner at ``c_spinor``.

        ``dE = 2 Re sum_{p>q} G_pq kappa_pq``, so the vector conjugate to ``kappa`` under the
        real inner product ``Re <g, kappa>`` is ``2 conj(G)``.

        Memoized on the **object identity** of ``(ints, c_spinor, gamma)``: the driver
        evaluates the gradient at an accepted iterate to feed the L-BFGS memory, and the
        next :meth:`step` call needs the gradient at exactly that iterate — without the
        memo it rebuilds the active Fock (a J/K pass) for a bitwise-identical result, once
        per accepted macro-iteration. A caller that rebuilt any of the three objects is at
        a new iterate even if the numbers agree, which is why identity, not value, is the
        key.
        """
        gc = self._grad_cache
        if (gc is not None and gc["ints"] is ints and gc["c"] is c_spinor
                and gc["gamma"] is gamma):
            self._f_active_ao = gc["f_active_ao"]
            return gc["grad"], gc["hdiag"]
        f_active_ao = _active_fock_ao(factors, c_spinor, self.spaces, gamma)
        self._f_active_ao = f_active_ao        # handed to the Hessian; one J/K build saved
        f_active = c_spinor.conj().T @ f_active_ao @ c_spinor
        g_mat = orbital_gradient(ints, gamma, gamma2, f_active)
        # ⚠ Projected *here*, so that everything built from a gradient — the L-BFGS pairs, the
        # Krylov subspace, the convergence test — lives in the constrained space rather than
        # being corrected at the end. For a time-reversal-symmetric Hamiltonian and a
        # time-even ensemble the exact gradient is already even, so what this removes is the
        # roundoff the Fock builds put into the odd directions; it is that roundoff, amplified
        # step over step, that the drift is made of.
        grad = self._project(2.0 * np.conj(_pack(g_mat, self.rows, self.cols)))
        hdiag = diagonal_hessian(ints, gamma, f_active, self.rows, self.cols)
        if self.kmap is not None:
            hdiag = kramers_mirror_mean(hdiag, self.kmap)
        self._grad_cache = {"ints": ints, "c": c_spinor, "gamma": gamma,
                            "grad": grad, "hdiag": hdiag, "f_active_ao": f_active_ao}
        return grad, hdiag

    def step(self, ints: CASIntegrals, gamma: np.ndarray, gamma2: np.ndarray,
             factors: ThreeIndexAO, c_spinor: np.ndarray, *,
             energy: Optional[float] = None, h_ao: Optional[np.ndarray] = None,
             hessian_vector: Optional[Callable[[np.ndarray], np.ndarray]] = None
             ) -> OrbitalStep:
        """Propose an orbital rotation for the current RDMs.

        The proposal is **not** committed: whether it is kept is decided by the caller after
        evaluating the energy at the rotated orbitals, via :meth:`accept` or :meth:`reject`.
        That separation is what makes the trust region real — a step that raises the energy
        can actually be undone, rather than merely making the *next* step smaller.

        ``h_ao`` is needed only when a second-order step is taken (to build
        :class:`OrbitalHessian`); ``hessian_vector`` overrides that with a caller-supplied
        product, which is how a future one-step method including the orbital-CI coupling would
        plug in.
        """
        if h_ao is None and hessian_vector is None and self.mode != "quasi-newton":
            raise ValueError("a second-order step needs h_ao (or an explicit hessian_vector); "
                             "pass it, or construct the optimizer with mode='quasi-newton'")
        # Rejection retry at the same iterate: the Hessian and its Krylov subspace are
        # bitwise what the previous call computed (the gradient is memoized inside
        # :meth:`gradient` itself). Identity, not value, is the right key — a caller that
        # rebuilt any of these objects is at a new iterate even if the numbers agree.
        grad, hdiag = self.gradient(ints, gamma, gamma2, factors, c_spinor)
        cache = self._iterate_cache
        if not (cache is not None and cache["ints"] is ints and cache["c"] is c_spinor
                and cache["gamma"] is gamma):
            cache = None
        e_now = cas_energy(ints, gamma, gamma2) if energy is None else float(energy)
        gnorm = float(np.linalg.norm(grad))

        use_second_order = self._decide_second_order(gnorm)
        second_order = False
        direction = None
        hess = None
        subspace = None
        if use_second_order:
            hv = hessian_vector
            if hv is None:
                if cache is not None and cache.get("hess") is not None:
                    hess = cache["hess"]
                else:
                    hess = OrbitalHessian(ints, factors, h_ao, c_spinor, gamma, gamma2,
                                          self.rows, self.cols,
                                          f_active_ao=getattr(self, "_f_active_ao", None))
                hv = hess.matvec
            if self.kmap is not None:
                # The Hessian commutes with time reversal, so the constrained problem is the
                # restriction of the unconstrained one to the even subspace: projecting the
                # product keeps the whole Krylov space there (the gradient that seeds it
                # already is). Without this the solve can pick up the near-null odd
                # directions the constraint exists to exclude and divide by their curvature.
                def hv(kappa_vec, _unconstrained=hv):
                    return self._project(_unconstrained(kappa_vec))
            # Inexact Newton (Eisenstat-Walker): solve loosely while the gradient is large —
            # the step is trust-region capped there anyway, so a tight solve burns
            # Hessian-vector products for a step that gets truncated — and tighten
            # automatically as the gradient falls, which is what buys superlinear convergence.
            eta = max(self.ah_tol, min(0.5, float(np.sqrt(gnorm))))
            reuse = cache.get("subspace") if cache is not None else None
            # The exact diagonal (see OrbitalHessian.exact_diagonal) preconditions the
            # solve; the Fock-difference hdiag stays what the quasi-Newton machinery uses.
            # A caller-supplied hessian_vector has no exact diagonal to offer.
            hdiag_ah = hess.exact_diagonal() if hess is not None else hdiag
            if self.kmap is not None:
                # Symmetrized for the same reason the quasi-Newton diagonal is: the solver's
                # linear combinations are all real, so the subspace stays in the constrained
                # space as long as every *elementwise* scaling inside it does too.
                hdiag_ah = kramers_mirror_mean(hdiag_ah, self.kmap)
            ah = augmented_hessian_step(grad, hv, hdiag_ah, tol=eta,
                                        max_iter=self.ah_max_iter, trust=self.trust,
                                        guess=self._ah_guess, subspace=reuse,
                                        recycle=self._ah_ritz,
                                        n_ritz_out=self.ah_recycle)
            subspace = ah.subspace
            if ah.ritz_vectors is not None:
                self._ah_ritz = ah.ritz_vectors
            self._ah_guess = ah.step.copy() if ah.step.size else None
            self.n_hessian_matvec += ah.n_matvec
            if self._dot(ah.step, grad) < 0.0:          # must be downhill to be usable
                direction = ah.step
                second_order = True
                self.n_second_order_steps += 1
            else:
                log.debug("augmented-Hessian step was not a descent direction (mu = %.3e); "
                          "falling back on the quasi-Newton step", ah.eigenvalue)

        if direction is None:
            direction = -self._two_loop(grad, hdiag)
            if self._dot(direction, grad) > 0.0:        # not a descent direction: fall back
                log.debug("L-BFGS direction was uphill; falling back on the preconditioned "
                          "gradient")
                self.reset_memory()
                direction = -grad / hdiag

        # Last line of defence: the preconditioner and the augmented-Hessian solve are both
        # only *approximately* time-reversal symmetric in floating point, so the direction is
        # projected before it is scaled — the trust radius then applies to the step actually
        # taken, and the pending step recorded for the L-BFGS memory is the one that was.
        direction = self._project(direction)
        max_rot = float(np.max(np.abs(direction))) if direction.size else 0.0
        if max_rot > self.trust:
            direction = direction * (self.trust / max_rot)
            max_rot = self.trust

        self._pending = (grad, direction, e_now)
        self._iterate_cache = {
            "ints": ints, "c": c_spinor, "gamma": gamma,
            # Only a Hessian this optimizer built itself is cached: a caller-supplied
            # hessian_vector may close over state this cache cannot see.
            "hess": hess if hessian_vector is None else None,
            "subspace": subspace if hessian_vector is None else None,
        }
        kappa = _unpack(direction, self.rows, self.cols, ints.n_orb)
        return OrbitalStep(kappa=kappa, unitary=unitary_from_antihermitian(kappa),
                           energy=e_now, grad_norm=gnorm, max_rotation=max_rot,
                           trust=self.trust, second_order=second_order)

    def _decide_second_order(self, gnorm: float) -> bool:
        """Escalation policy for ``mode="auto"``.

        Escalation is driven by **failure, not proximity**: where the quasi-Newton
        step converges at all it costs 3-4x less total work, because Hessian-vector products
        are not free. So the question is only "is the cheap step actually getting anywhere?",
        and the honest measure of that is the **gradient trajectory**:

        * *stalled* — the gradient has not fallen by ``stall_factor`` across the last
          ``stall_window`` iterations, or steps are being rejected repeatedly.
        * *not stalled* — the gradient keeps falling, however slowly. Leave it alone.

        Judging this on the **energy** instead does not work, and the trace that proved it is
        worth recording: on an easy problem the quasi-Newton run reached ``|g| = 1e-3`` with a
        per-iteration energy change of 1.5e-7, looking for all the world like a stall — and
        then converged on its own in 51 more cheap iterations. An energy-based detector
        escalated there and turned 118 work units into 332. The gradient was falling the whole
        time; the energy simply had nothing left to say.

        Once escalated the optimizer stays there: dropping back would discard the only model
        that was working.
        """
        if self.mode == "second-order":
            return True
        if self.mode == "quasi-newton":
            return False
        if self._second_order:
            return True
        self._gnorm_history.append(gnorm)
        if len(self._gnorm_history) > self.stall_window:
            self._gnorm_history.pop(0)
        no_progress = (len(self._gnorm_history) >= self.stall_window
                       and gnorm > self._gnorm_history[0] / self.stall_factor)
        stalled = (no_progress or self._stalls >= self.stall_patience) and gnorm > self.conv_grad
        if gnorm < self.second_order_start or stalled:
            self._second_order = True
            log.debug("escalating to the second-order step (|g| = %.3e, stalls = %d)",
                      gnorm, self._stalls)
            return True
        return False

    def accept(self, new_gradient: Optional[np.ndarray] = None) -> None:
        """Commit the pending step; ``new_gradient`` (at the rotated orbitals) feeds L-BFGS."""
        if self._pending is None:
            return
        grad, direction, energy = self._pending
        if new_gradient is not None:
            self._prev_grad = grad
            self._prev_step = direction
            self._update_memory(new_gradient)
        self._stalls = 0                    # progress was made; the rejection run is broken
        self._prev_energy = energy
        self.trust = min(1.5 * self.trust, self.max_step)
        self._pending = None
        self._iterate_cache = None          # the iterate moves; the Hessian is stale
        # _grad_cache is deliberately kept: the driver evaluated the gradient at the
        # accepted iterate for the L-BFGS update, and that is the iterate the next step
        # starts from. The identity key makes a stale entry a miss, never a wrong hit.

    # -- checkpointing ---------------------------------------------------------------
    def state_dict(self, *, space_key: Optional[str] = None) -> dict:
        """Everything a restart needs, as plain arrays and scalars.

        The trust radius, the L-BFGS pairs, the augmented-Hessian warm start, the stall
        counter, the gradient trajectory and the escalation flag — i.e. the whole of what
        makes iteration *n+1* differ from a cold start at the same orbitals. The orbitals
        themselves are the caller's, by the same design that lets one optimizer drive a
        CI-CASSCF and a DMRG-CASSCF.

        ``space_key`` is the identity of the *solver's* surface
        (:meth:`kuiva.mcscf.adaptive.AdaptiveCISolver.space_key`). It is recorded rather than
        used: :meth:`load_state_dict` is where it decides something.

        ⚠ The **pending step is deliberately not saved**. A checkpoint is written between
        macro-iterations, where the accept/reject has already happened and there is nothing
        pending; saving a half-evaluated trial step would restore a proposal whose trial
        energy was never computed.
        """
        empty = np.zeros((0, self.n_parameters), dtype=np.complex128)
        return {
            "n_parameters": self.n_parameters,
            "space_key": space_key,
            "trust": float(self.trust),
            "second_order": bool(self._second_order),
            "stalls": int(self._stalls),
            "grad_norm_history": np.asarray(self._gnorm_history, dtype=np.float64),
            "lbfgs_s": np.array(self._s, dtype=np.complex128) if self._s else empty,
            "lbfgs_y": np.array(self._y, dtype=np.complex128) if self._y else empty,
            "prev_grad": self._prev_grad,
            "prev_step": self._prev_step,
            "prev_energy": self._prev_energy,
            "ah_guess": self._ah_guess,
            "n_hessian_matvec": int(self.n_hessian_matvec),
            "n_second_order_steps": int(self.n_second_order_steps),
            "n_rejected": int(self.n_rejected),
        }

    def load_state_dict(self, state: dict, *, space_key: Optional[str] = None) -> None:
        """Restore :meth:`state_dict`. ⚠ **Curvature memory is chart-scoped**.

        If ``space_key`` differs from the one recorded in ``state``, the L-BFGS pairs, the
        augmented-Hessian warm start, the stall counter and the gradient trajectory are
        **cleared, not restored**, and the trust radius is taken to a floor rather than
        inherited. Every one of those is a statement about an energy surface, and restoring
        them against a *different* space is precisely the bug chart-scoping exists to prevent
        — the same reasoning as :meth:`reset_chart`, applied across a process boundary.

        The parameter count is checked: a state dict from a different orbital partition
        describes rotations that do not exist here, and restoring it would leave an L-BFGS
        memory indexed against the wrong pairs — plausible-looking and wrong.
        """
        stored = int(state.get("n_parameters", self.n_parameters))
        if stored != self.n_parameters:
            raise ValueError(
                "this optimizer has {} rotation parameters and the stored state has {}; the "
                "orbital partition, active_active, or the irrep mask (labels=) differs from "
                "the run that wrote it"
                .format(self.n_parameters, stored))
        recorded = state.get("space_key")
        same_chart = space_key is None or recorded is None or recorded == space_key

        self.trust = float(state.get("trust", self.trust))
        self._second_order = bool(state.get("second_order", self._second_order))
        self.n_hessian_matvec = int(state.get("n_hessian_matvec", 0))
        self.n_second_order_steps = int(state.get("n_second_order_steps", 0))
        self.n_rejected = int(state.get("n_rejected", 0))
        self._pending = None
        self._iterate_cache = None
        self._grad_cache = None
        # The Ritz seeds are a per-process warm start, deliberately not checkpointed: a
        # restart pays a few extra products on its first solve rather than carrying more
        # surface-dependent state across the boundary.
        self._ah_ritz = None

        if not same_chart:
            log.warning("the checkpoint was written on CI space %r and this run is on %r; "
                        "the L-BFGS curvature, the augmented-Hessian warm start and the "
                        "trust radius belong to the surface that no longer exists and are "
                        "discarded rather than restored ",
                        recorded, space_key)
            self.reset_chart(trust_floor=0.1 * self.max_step)
            return

        self._stalls = int(state.get("stalls", 0))
        self._gnorm_history = [float(g) for g in
                               np.asarray(state.get("grad_norm_history", []), dtype=float)]
        pairs_s = np.asarray(state.get("lbfgs_s", np.zeros((0, self.n_parameters))),
                             dtype=np.complex128)
        pairs_y = np.asarray(state.get("lbfgs_y", np.zeros((0, self.n_parameters))),
                             dtype=np.complex128)
        self._s = [np.ascontiguousarray(row) for row in pairs_s]
        self._y = [np.ascontiguousarray(row) for row in pairs_y]
        self._prev_grad = _as_optional_vector(state.get("prev_grad"))
        self._prev_step = _as_optional_vector(state.get("prev_step"))
        self._ah_guess = _as_optional_vector(state.get("ah_guess"))
        energy = state.get("prev_energy")
        self._prev_energy = None if energy is None else float(energy)

    def reject(self) -> None:
        """Discard the pending step, shrink the trust region, and count the stall.

        The L-BFGS memory is deliberately **kept**: in a trust-region method a rejection means
        the step was too long, not that the accumulated curvature is wrong, and discarding
        hard-won pairs on every rejection throws the method back to preconditioned steepest
        descent exactly when it is struggling.

        The iterate cache is kept too, deliberately: a rejection means the *next*
        :meth:`step` call happens at the same orbitals and RDMs, where the cached gradient,
        Hessian and Krylov subspace are still exact — the retry then costs no Hessian-vector
        products and no gradient build, only the small shifted re-solve at the new trust
        radius.
        """
        self.trust = max(0.25 * self.trust, 1e-6)
        self.n_rejected += 1
        self._stalls += 1
        self._pending = None


# --- Driver ---------------------------------------------------------------------------------

@dataclass
class CASSCFResult:
    """Outcome of an orbital optimization."""
    energy: float
    coeff: np.ndarray
    gamma: np.ndarray
    gamma2: np.ndarray
    converged: bool
    n_iterations: int
    grad_norm: float
    history: List[float] = field(default_factory=list)
    #: Cost bookkeeping. The honest measure of an MCSCF optimizer is not the macro-iteration
    #: count but the total work, because a Hessian-vector product is not free: it is two J/K
    #: builds plus a three-index transform. A method that halves the iterations while spending
    #: tens of products per iteration is more expensive, not less.
    n_hessian_matvec: int = 0
    n_second_order_steps: int = 0
    n_rejected: int = 0
    #: Trial points the CI solver refused (:class:`~kuiva.util.errors.SolverFailure`). Each
    #: one is a rejected step, not a failure of the run — but a converged result reached past
    #: several of them was optimized on a restricted set of directions, so the count is
    #: reported rather than swallowed.
    n_solver_failures: int = 0
    #: Lowest curvature of the orbital Hessian along the time-reversal-**odd** rotations a
    #: Kramers constraint forbids, measured at the converged point
    #: (:func:`measure_time_odd_curvature`); ``None`` where the test did not run — an
    #: unconstrained or unconverged optimization, or ``kramers_stability=False``. ⚠ ``None``
    #: is "not measured" and is never to be read as "stable".
    time_odd_curvature: Optional[float] = None
    #: True when that measurement found a saddle and the driver released the constraint and
    #: continued (see :func:`optimize_orbitals`). The energy below is then the broken
    #: solution's, and it is **lower** than the symmetric one the constrained leg reached.
    kramers_released: bool = False

    @property
    def work_units(self) -> float:
        """Approximate total cost in macro-iteration-equivalents.

        ⚠ The weight is **indicative, not exact**, and it is system dependent: it is the ratio
        of one Hessian-vector product to one (integral transform + CI solve). Measured on
        TiCl3 with the cheap CI — 12.9 s per product against 4.4 s per iteration — the ratio
        is ~2.9; the operation count alone suggests ~1.5. Where the CI is expensive (DMRG) the
        ratio falls below 1 and macro-iterations dominate instead. Use this to compare
        optimizer modes on *one* system, not to compare systems.
        """
        return self.n_iterations + HVP_WORK_WEIGHT * self.n_hessian_matvec


@threads.blas_stage
def optimize_orbitals(factors: ThreeIndexAO, h_ao: np.ndarray, c_spinor: np.ndarray,
                      spaces: OrbitalSpaces,
                      ci_solver: Callable[[CASIntegrals], Tuple[float, np.ndarray, np.ndarray]],
                      *, e_nuc: float = 0.0, max_iter: int = 50, conv_grad: float = 1e-4,
                      conv_energy: float = 1e-8, max_step: float = DEFAULT_MAX_STEP,
                      memory: int = 10, active_active: bool = False,
                      labels: Optional[np.ndarray] = None, mode: str = "auto",
                      kramers_rotation: Any = "auto", n_active_elec: Optional[int] = None,
                      kramers_stability: Any = "auto",
                      second_order_start: float = SECOND_ORDER_START,
                      callback: Optional[Callable[[dict], Optional[bool]]] = None,
                      report: bool = True,
                      optimizer_state: Optional[dict] = None, start_iteration: int = 0,
                      space_key: Optional[str] = None,
                      history: Optional[Sequence[float]] = None,
                      repair_orbitals: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                      extra_columns: Sequence[Tuple[Any, Callable[[], Any]]] = ()
                      ) -> CASSCFResult:
    """Alternate CI and orbital steps to convergence — the shared MCSCF driver.

    ``ci_solver(ints) -> (energy, gamma, gamma2)`` is the **only** thing that distinguishes a
    cheap-CI pre-optimization from a CASSCF or a DMRG-CASSCF. It receives the integrals at the
    current orbitals and returns the state-averaged RDMs over the active space.

    Convergence is on the orbital gradient norm *and* the energy change; both must be met,
    because a truncated CI can stall the energy while the orbitals are still moving.

    ⚠ **A solver that refuses a trial point is a rejected step, not the end of the run.**
    ``ci_solver`` may raise :class:`~kuiva.util.errors.SolverFailure` at a trial rotation —
    an eigensolver that did not converge, an iteration cap, a state count that cuts a
    degenerate block of *that point's* spectrum
    (:class:`~kuiva.util.errors.StateAverageSplit`). The trial is then rejected exactly as a
    step that raised the energy is: the trust radius shrinks and the next trial point is
    closer to one that worked, and the refusal is a ``WARNING`` naming the iteration.
    :attr:`CASSCFResult.n_solver_failures` counts them, because a result reached past several
    of them was optimized over a restricted set of directions.

    The refusal still **propagates** at the two points where there is nothing to reject: the
    first solve, at the starting orbitals, and the gradient build at an *accepted* point. And
    the catch is narrow by design — only ``SolverFailure``, the declared contract for "no
    usable answer at this point". Anything else raises, because a shape mismatch or a memory
    refusal turned into a rejected step is a run that shrinks its trust radius to nothing and
    prints no reason why.

    ``callback(info)`` is invoked after every macro-iteration with the current iteration,
    energy, gradient norm, energy change and step type — **plus** the current orbitals, RDMs
    and the optimizer itself, which is what a checkpoint writer needs
    (:mod:`kuiva.io.checkpoint`). The dict is extended additively and never renamed.
    **Returning ``False`` stops the run**, which is how an external wall-clock budget is
    enforced from the inside — a run killed by an external timeout leaves nothing behind,
    whereas one that stops itself has already written every iteration it completed.

    Restart
    ------------
    ``optimizer_state`` (from :meth:`OrbitalOptimizer.state_dict`), ``start_iteration`` and
    ``history`` resume a checkpointed run: pass the checkpointed orbitals as ``c_spinor`` and
    the three of these, and the trajectory continues rather than starting over.
    ``labels`` (an ``(n_orb, width)`` irrep-label array from :mod:`kuiva.symm`) restricts the
    rotation to **within** each irrep, so the orbitals stay symmetry-pure exactly rather than
    to a tolerance and a label still means something at convergence. ⚠ It is a constraint: the
    result is the lowest *symmetric* solution, which is not the global one wherever the
    symmetry is spontaneously broken.

    ``space_key`` is the CI solver's chart identity and is what makes the restored curvature
    chart-scoped — see :meth:`OrbitalOptimizer.load_state_dict`. ⚠ ``max_iter`` counts
    **total** macro-iterations, so a restart at iteration 12 with ``max_iter=50`` runs 38
    more; that is what makes an interrupted-and-restarted run cost the same as an
    uninterrupted one.

    Solver-specific columns
    -----------------------
    ``extra_columns`` is a sequence of ``(Column, getter)`` pairs appended to the iteration
    table; each ``getter()`` takes no arguments and returns that iteration's value. It exists
    so a solver can put its own quality number in the one table a reader is looking at — the
    tensor-network truncation weight is the case it was added for — **without this driver
    learning anything about that solver**. The getter closes over whatever it needs.

    ⚠ This is an *additive keyword*, deliberately, and not a restructuring of the loop below.
    That loop is the validated driver: it may grow arguments and its ``callback(info)`` dict
    may grow keys, but its control flow is not rearranged to accommodate a caller.

    Keeping the orbitals Kramers paired
    -----------------------------------
    ``kramers_rotation`` constrains the step so that a Kramers-paired orbital set stays
    paired — and therefore so that every orbital space stays closed under time reversal,
    which an unconstrained complex rotation does not (the module docstring has the mechanism
    and the measurement). ``"auto"`` (the default) decides from the **incoming orbitals**
    alone: constrained where they are paired to :data:`KRAMERS_PAIRING_TOL` and the rotation
    list is pair-closed, unconstrained where the constraint would be meaningless. ``True``
    refuses an unpaired set rather than pretending; ``False`` reproduces the unconstrained
    trajectory bitwise. The decision and what it rested on are reported on the ``rotation``
    line.

    Releasing it: is the symmetric solution a minimum?
    -------------------------------------------------
    A constraint converges to the lowest *time-reversal-symmetric* solution, which is not the
    unconstrained optimum wherever that symmetry is spontaneously broken — and at an even
    active electron count it genuinely can be (N2 CAS(6,8): the broken solution lies 0.64 Eh
    lower with its density 42% time-odd). So the constraint is not the end of the run.
    ``kramers_stability`` decides what happens at the converged point:

    * ``"auto"`` (the default) — where a release could be acted on (an ``"auto"``-imposed
      constraint, an **even** ``n_active_elec``, and macro-iterations left in the budget),
      measure the lowest curvature of the exact orbital Hessian **restricted to the time-odd
      rotations the constraint forbade** (:func:`measure_time_odd_curvature`). Non-negative:
      the symmetric solution is a minimum and the run is finished. Negative beyond
      :data:`KRAMERS_STABILITY_TOL`: the constraint is **released**, the orbitals are stepped
      off the saddle along the offending direction, and the optimization continues
      unconstrained — the orbital analogue of the SCF's ``stability="follow"``.
    * ``True`` — measure it wherever the rotation was constrained and the run converged, and
      report it. It **releases only where** ``"auto"`` would: an explicit
      ``kramers_rotation=True`` asked for a Kramers-restricted optimization and gets one, with
      the instability named rather than acted on, and an odd count is left alone because a
      broken solution there has no Kramers degeneracy and the state-averaging gate refuses it
      downstream — there is nothing to release *to*.
    * ``False`` — never measure. The cheapest setting, and the one that reproduces a
      constrained run exactly.

    ⚠ The measurement is meaningful **only at a converged point**, so a run that stopped on
    ``max_iter`` or on a callback is reported as not measured rather than as stable. ⚠ The
    released leg spends the **same** ``max_iter`` budget: the count is of total
    macro-iterations, exactly as it is across a restart, so a release near the end of a budget
    is reported as a run that did not converge rather than silently given a second one.

    ``repair_orbitals(c) -> c`` is applied to the starting orbitals and to **every trial
    rotation before its integrals are built**, so what is evaluated is what would be kept.

    ⚠ **Nothing passes it and nothing should: repairing the orbitals after a step is the
    remedy that does not work**, kept as a seam and as a record rather than as an option.
    Projecting each space onto its nearest time-reversal-closed span
    (:func:`kuiva.interface.api.kramers_span_repair`) does remove the drift, and is wrong
    anyway — "nearest closed span" stops meaning "almost the same span" once a block has
    drifted, so it re-selects the active space, measured as an N2 CAS(6,8) converging 0.6 Eh
    *above its own SCF energy*. The drift is fixed where it is caused, in the step.
    """
    c = np.ascontiguousarray(c_spinor, dtype=np.complex128)
    if repair_orbitals is not None:
        c = np.ascontiguousarray(repair_orbitals(c), dtype=np.complex128)
    kramers, pairing = resolve_kramers_rotation(kramers_rotation, c, spaces,
                                                active_active=active_active, labels=labels)
    if kramers_stability not in (True, False) and kramers_stability != "auto":
        raise ValueError("kramers_stability must be True, False or 'auto'; got {!r}"
                         .format(kramers_stability))
    opt = OrbitalOptimizer(spaces, max_step=max_step, memory=memory,
                           active_active=active_active, labels=labels, mode=mode,
                           second_order_start=second_order_start, conv_grad=conv_grad,
                           kramers=kramers)
    if optimizer_state is not None:
        opt.load_state_dict(optimizer_state, space_key=space_key)
    if report:
        out.subsection(log, "Orbital optimization")
        out.entries(log, [
            ("inactive / active / virtual spinors",
             "{} / {} / {}".format(spaces.n_inactive, spaces.n_active, spaces.n_virtual)),
            ("orbital rotation parameters", opt.n_parameters, "",
             "complex" if labels is None else
             "complex; irrep-blocked, {} of {} pairs kept".format(
                 opt.n_parameters, spaces.rotation_pairs(active_active)[0].size)),
            ("optimizer mode", mode, "",
             "second order below |g| = {:.1e}".format(second_order_start)
             if mode == "auto" else ""),
            ("rotation", "Kramers constrained" if kramers else "general complex", "",
             kramers_rotation_note(kramers, pairing)),
            ("gradient convergence", conv_grad, "", "", "{:.1e}"),
            ("maximum rotation per step", max_step, "rad", "", "{:.2f}"),
        ])
        table = out.Table(log, [out.col_iter(), out.col_energy("E [Eh]"), out.col_delta(),
                                out.col_resid("|g|"), out.Column("max rot", "{:.4f}", 8),
                                out.Column("step", "{}", 6, align="<"), out.col_time()]
                          + [column for column, _ in extra_columns])
        table.start()

    # Trust-region loop with genuine accept/reject: a trial rotation is evaluated before it
    # is kept, so a step that raises the energy is undone rather than merely regretted.
    ints = CASIntegrals.build(factors, h_ao, c, spaces, e_nuc=e_nuc)
    energy, gamma, gamma2 = ci_solver(ints)
    # ⚠ On a restart the supplied history already ends with the energy at these orbitals (it
    # is the checkpoint's own energy), so it is taken as it stands. Appending the recomputed
    # value instead would duplicate the last entry and make a restarted trajectory one
    # element longer than the uninterrupted one it is supposed to reproduce.
    history = [energy] if history is None else [float(e) for e in history]
    converged = False
    gnorm = np.inf
    n_solver_failures = 0
    it = int(start_iteration)
    for it in range(int(start_iteration) + 1, max_iter + 1):
        with timer("macro-iteration") as t_it:
            step = opt.step(ints, gamma, gamma2, factors, c, energy=energy, h_ao=h_ao)
            gnorm = step.grad_norm
            if gnorm < conv_grad:
                converged = True
                de = 0.0
            else:
                c_try = np.ascontiguousarray(c @ step.unitary)
                if repair_orbitals is not None:
                    # Before the integrals, so the evaluated point IS the kept point.
                    c_try = np.ascontiguousarray(repair_orbitals(c_try),
                                                 dtype=np.complex128)
                ints_try = CASIntegrals.build(factors, h_ao, c_try, spaces, e_nuc=e_nuc)
                try:
                    e_try, g_try, g2_try = ci_solver(ints_try)
                except SolverFailure as exc:
                    # ⚠ A point the solver cannot evaluate is a point the optimizer must not
                    # move to — it is not a failure of the calculation, and killing the run
                    # here throws away every macro-iteration already paid for. Rejecting
                    # shrinks the trust radius, so the next trial point is closer to one that
                    # did work; if none ever does, the run reports that it did not converge
                    # with these warnings in the log, which is a far more useful outcome than
                    # a traceback at macro-iteration three.
                    #
                    # ⚠ The catch is deliberately narrow. ``SolverFailure`` is the *declared*
                    # contract for "no usable answer at this point"
                    # (:mod:`kuiva.util.errors`); anything else — a shape mismatch, a memory
                    # refusal, a bug — propagates, because turning those into a rejected step
                    # would convert a diagnosable error into a run that shrinks its trust
                    # radius to nothing for a reason nothing prints.
                    n_solver_failures += 1
                    opt.reject()
                    de = 0.0
                    log.warning("the CI solver refused the trial point of macro-iteration "
                                "%d (%s); the step is rejected and the trust radius is now "
                                "%.2e", it, exc, opt.trust)
                else:
                    de = e_try - energy
                    if de <= conv_energy:                  # accept (a tiny rise is noise)
                        grad_new, _ = opt.gradient(ints_try, g_try, g2_try, factors, c_try)
                        opt.accept(grad_new)
                        c, ints, energy, gamma, gamma2 = (c_try, ints_try, e_try,
                                                          g_try, g2_try)
                        history.append(energy)
                        if abs(de) < conv_energy and gnorm < conv_grad:
                            converged = True
                    else:
                        opt.reject()
                        log.debug("rejected a step that raised the energy by %.3e Eh; trust "
                                  "radius now %.2e", de, opt.trust)
        if report:
            table.row(it, energy, de, gnorm, step.max_rotation,
                      "2nd" if step.second_order else "qn", t_it.wall,
                      *[getter() for _, getter in extra_columns])
        if callback is not None:
            # Extended additively for the checkpoint writer: the orbitals, the RDMs and
            # the optimizer are what a restart needs, and they are already in hand here. A
            # callback that ignores them is unaffected, which is why this is an extension
            # rather than a new hook -- optimize_orbitals' loop is the validated driver.
            info = {"iteration": it, "energy": energy, "grad_norm": gnorm, "de": de,
                    "second_order": step.second_order, "converged": converged,
                    "n_hessian_matvec": opt.n_hessian_matvec, "wall": t_it.wall,
                    "coeff": c, "gamma": gamma, "gamma2": gamma2, "spaces": spaces,
                    "optimizer": opt, "trust": opt.trust, "history": history,
                    "ci_solver": ci_solver, "e_nuc": e_nuc}
            if callback(info) is False:
                log.warning("orbital optimization stopped by callback at iteration %d "
                            "(|g| = %.3e); the result is the last iterate, not a converged "
                            "one", it, gnorm)
                break
        if converged:
            break

    # Is the constrained solution a minimum, or a saddle in the directions the constraint
    # forbade? Only a converged point can answer it, and only an even electron count has
    # anywhere to go if the answer is "saddle" (the docstring above says why).
    curvature = None
    releasable = (kramers_rotation == "auto" and n_active_elec is not None
                  and int(n_active_elec) % 2 == 0 and it < max_iter)
    if converged and opt.kmap is not None and kramers_stability is not False \
            and (releasable or kramers_stability is True):
        curvature = measure_time_odd_curvature(opt, ints, factors, h_ao, c, gamma, gamma2)

    if report:
        table.end("converged" if converged else
                  "NOT converged in {} macro-iterations".format(max_iter))
        entries = [("second-order steps taken", opt.n_second_order_steps),
                   ("Hessian-vector products", opt.n_hessian_matvec),
                   ("rejected steps", opt.n_rejected)]
        if n_solver_failures:
            # Only when it happened: a zero row here would put a line about a failure mode
            # in every output file in the project. A nonzero one is worth a reader's eye,
            # because those directions were never evaluated.
            entries.append(("trial points the CI refused", n_solver_failures, "",
                            "each one rejected the step; the trust radius shrank"))
        if curvature is not None:
            entries.append(("lowest time-odd curvature", curvature.value, "Eh/rad^2",
                            "{}; {} products".format(curvature.verdict, curvature.n_matvec),
                            out.SCI_FMT))
        out.entries(log, entries)
    if curvature is not None and curvature.unstable:
        if releasable:
            kept = ""
        elif kramers_rotation != "auto":
            kept = " (the constraint is kept: an explicit kramers_rotation was asked for)"
        elif n_active_elec is None or int(n_active_elec) % 2 == 1:
            kept = (" (the constraint is kept: at an odd active electron count a broken "
                    "solution has no Kramers degeneracy and is refused downstream, so there "
                    "is nothing to release to)")
        else:
            kept = " (the constraint is kept: the macro-iteration budget is spent)"
        log.warning("the Kramers-constrained solution is a SADDLE: the orbital Hessian has "
                    "curvature %.3e Eh/rad^2 along a time-reversal-odd rotation, so a "
                    "time-reversal-broken solution lies below it%s", curvature.value, kept)
    if curvature is not None and curvature.unstable and releasable:
        # Release and follow. A recursive call rather than a second loop: optimize_orbitals'
        # loop is the validated driver and is not restructured to accommodate this — what a
        # release needs is a *fresh unconstrained optimization from the displaced orbitals*,
        # which is exactly what one call is. The optimizer state is deliberately not carried
        # over: the parameter space is a different one, so the curvature memory is a memory
        # of another chart.
        log.warning("releasing the Kramers constraint and continuing from a %.2f rad step "
                    "along that direction; the run continues inside the same max_iter "
                    "budget (%d of %d macro-iterations spent)",
                    KRAMERS_RELEASE_STEP, it, max_iter)
        c_released = np.ascontiguousarray(
            c @ kramers_release_rotation(curvature.direction, opt.rows, opt.cols,
                                         spaces.n_orb))
        followed = optimize_orbitals(
            factors, h_ao, c_released, spaces, ci_solver, e_nuc=e_nuc, max_iter=max_iter,
            conv_grad=conv_grad, conv_energy=conv_energy, max_step=max_step, memory=memory,
            active_active=active_active, labels=labels, mode=mode, kramers_rotation=False,
            n_active_elec=n_active_elec, kramers_stability=False,
            second_order_start=second_order_start, callback=callback, report=report,
            start_iteration=it, space_key=space_key, history=None,
            repair_orbitals=repair_orbitals, extra_columns=extra_columns)
        return replace(followed,
                       history=history + followed.history,
                       n_hessian_matvec=opt.n_hessian_matvec + followed.n_hessian_matvec,
                       n_second_order_steps=(opt.n_second_order_steps
                                             + followed.n_second_order_steps),
                       n_rejected=opt.n_rejected + followed.n_rejected,
                       n_solver_failures=(n_solver_failures
                                          + followed.n_solver_failures),
                       time_odd_curvature=curvature.value, kramers_released=True)
    if not converged:
        log.warning("orbital optimization did not converge in %d macro-iterations "
                    "(|g| = %.3e, target %.1e); the orbitals are the last iterate and may "
                    "not be stationary", max_iter, gnorm, conv_grad)
    return CASSCFResult(energy=energy, coeff=c, gamma=gamma, gamma2=gamma2,
                        converged=converged, n_iterations=it, grad_norm=gnorm,
                        history=history, n_hessian_matvec=opt.n_hessian_matvec,
                        n_second_order_steps=opt.n_second_order_steps,
                        n_rejected=opt.n_rejected,
                        n_solver_failures=n_solver_failures,
                        time_odd_curvature=None if curvature is None else curvature.value)


__all__ = ["OrbitalSpaces", "CASIntegrals", "cas_integrals_memory_gb",
           "kramers_parameter_map", "kramers_project", "kramers_mirror_mean",
           "resolve_kramers_rotation", "kramers_rotation_note",
           "kramers_odd_project", "TimeOddCurvature", "lowest_projected_curvature",
           "measure_time_odd_curvature", "kramers_release_rotation",
           "KRAMERS_PAIRING_TOL", "KRAMERS_STABILITY_TOL", "KRAMERS_RELEASE_STEP",
           "hessian_response_memory_gb", "HESSIAN_RESPONSE_COPIES", "hessian_square_memory_gb",
           "OrbitalOptimizer", "OrbitalStep", "CASSCFResult",
           "OrbitalHessian", "augmented_hessian_step", "AHResult",
           "cas_energy", "generalized_fock", "averaged_fock", "fock_diagonal",
           "orbital_gradient",
           "diagonal_hessian",
           "optimize_orbitals", "unitary_from_antihermitian", "DEFAULT_MAX_STEP",
           "SECOND_ORDER_START", "STALL_WINDOW", "STALL_FACTOR"]
