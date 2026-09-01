"""Cheap CI pre-optimization.

What this is for, and what it is not
------------------------------------
The design asks for a cheap pre-optimization that rotates the raw spinor guess toward physical
active orbitals *before* the real CASSCF or DMRG. It serves two purposes and it is judged
only on those:

1. **Which spinors are correlated** — natural spinors and their occupation numbers, so the
   active space can be chosen from a generous candidate set (:meth:`PreoptResult.
   suggest_active_space`).
2. **How the correlated spinors are connected** — single-orbital entropies and mutual
   information, which is what the DMRG tensor network is built from: orbital ordering now
    and topology later.

**It is not a quantitative method and its total energy means nothing.** It is a truncated CI
in a truncated space; the number it prints is above the CASSCF energy by an amount nobody
should interpret. What must be right is the *qualitative* picture — which orbitals carry
fractional occupation, and which of them talk to each other. Everything below is chosen to
make that picture reliable and cheap, and nothing is chosen to make the energy good.

The method
----------
A **selected multireference CISD** in the candidate active space:

* The **reference space** is a small complete active space over a window of near-degenerate
  spinors around the Fermi level, not a single determinant. This matters more than anything
  else here: for antiferromagnetically coupled metal centres — the systems this code exists
  for — a single determinant is a qualitatively wrong starting point, and singles and doubles
  from it recover none of the near-degeneracy. The window is chosen automatically to keep the
  reference CAS under :data:`DEFAULT_MAX_REFERENCE` determinants.
* Candidates are all single and double excitations out of the reference space.
* If there are too many, they are ranked by their **first-order perturbative weight**
  ``|sum_I c_I <D|H|I>|^2 / (H_DD - E_ref)`` and the largest are kept — the standard CIPSI /
  ASCI selection, evaluated only against the (small) reference space, which is what makes it
  affordable. Optionally iterated, using the converged wavefunction's leading determinants as
  the new generators.
* The result is diagonalized for the lowest ``n_states`` roots and **state averaged**,
  which is imposed here explicitly because the optimizer downstream cannot recover it.

Cost, honestly stated
---------------------
The determinant space is an explicit list, so finding which determinants interact is
``O(N^2)`` (:func:`kuiva.ci.strings.connections`). With the default ``max_determinants`` this
is seconds, and it is exact within the space. That is the right trade for a qualitative
pre-optimizer over a few metal centres — 3 Dy(III) f9 centres is 27 electrons in 42 spinors,
where the *full* CI is ~10^11 determinants and this is 10^4 — but it does mean the knob that
matters is the determinant count, not the size of the active space. Push the count and the
``O(N^2)`` search dominates.

References
----------
* Perturbatively selected CI (CIPSI): B. Huron, J. P. Malrieu, P. Rancurel, J. Chem. Phys.
  58, 5745 (1973), doi:10.1063/1.1679199. Modern large-scale form: Y. Garniron et al.,
  "Quantum Package 2.0", J. Chem. Theory Comput. 15, 3591 (2019),
  doi:10.1021/acs.jctc.9b00176.
* ASCI (selection against a small set of generators, as used here): N. M. Tubman, J. Lee,
  T. Y. Takeshita, M. Head-Gordon, K. B. Whaley, J. Chem. Phys. 145, 044112 (2016),
  doi:10.1063/1.4955109.
* Heat-bath selected CI: A. A. Holmes, N. M. Tubman, C. J. Umrigar, J. Chem. Theory Comput.
  12, 3674 (2016), doi:10.1021/acs.jctc.6b00407.
* Natural orbitals and their occupation numbers as an active-space criterion: P.-O. Loewdin,
  Phys. Rev. 97, 1474 (1955), doi:10.1103/PhysRev.97.1474; P. Pulay, T. P. Hamilton,
  J. Chem. Phys. 88, 4926 (1988), doi:10.1063/1.454704; C. J. Stein, M. Reiher,
  J. Chem. Theory Comput. 12, 1760 (2016), doi:10.1021/acs.jctc.6b00156.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..ci.strings import (Determinants, connections, diagonal_energies, hamiltonian_matrix,
                          occupation_correlations, rdm12)
from ..integrals.transform import ThreeIndexAO
from ..rdm.entropy import entanglement_report, fiedler_order, mutual_information
from ..rdm.entropy import single_orbital_entropy
from ..util import output as out
from ..util import resources as res
from ..util.logging import get_logger
from ..util.timing import timer
from .adaptive import Proposal, SolverFailure, array_key
from .events import DEFAULT_EVENT_INTERVAL, DEFAULT_TAU, optimize_orbitals_events
from .orbopt import CASIntegrals, OrbitalSpaces, optimize_orbitals

log = get_logger(__name__)

#: Determinant-space size. The O(N^2) connection search makes this the cost knob.
DEFAULT_MAX_DETERMINANTS = 6000
#: Cap on the reference complete active space.
DEFAULT_MAX_REFERENCE = 500
#: Leading determinants used as generators when re-selecting.
DEFAULT_MAX_GENERATORS = 200
#: Natural-occupation window outside which a spinor is deemed uncorrelated.
DEFAULT_OCCUPATION_WINDOW = (0.02, 0.98)
#: Determinant count below which the CI Hamiltonian is diagonalized densely as a matter of
#: course. A dense complex ``eigh`` is exact, needs no tuning and cannot fail to converge.
DENSE_SOLVE_MAX_DET = 600
#: Determinant count below which a *failed* ARPACK solve falls back to the dense one. Larger
#: than :data:`DENSE_SOLVE_MAX_DET` because the fallback is rare and the alternative is no
#: answer at all: 4000 determinants is a 244 MB matrix, checked against the memory budget
#: before it is formed.
DENSE_FALLBACK_MAX_DET = 4000
#: Absolute accuracy [Eh] asked of the CI eigenvalues, from which ARPACK's *relative* ``tol``
#: is derived (:func:`_eigsh_tol`). ⚠ Asking for machine precision — ARPACK's default — makes
#: Lanczos grind for minutes or fail outright on the near-degenerate low pair that a
#: state-averaged Kramers-degenerate calculation always has. But the tolerance may not simply
#: be loosened to taste either: whatever it is set to becomes a **noise floor on E(kappa)**,
#: and an optimizer whose accept/reject test works at ``conv_energy = 1e-8`` Eh cannot be
#: handed a surface that wobbles at 1e-9. Measured on example 4 (TiCl3, 2000 determinants), a
#: flat relative ``tol = 1e-12`` moved the printed energies by up to 5e-9 Eh. 1e-11 Eh puts
#: the floor three orders below the acceptance threshold, and it is *absolute* because that
#: is the quantity that matters — a relative tolerance means something different for an
#: active-space energy of 10 Eh and one of 1000.
EIGSH_TARGET_EH = 1.0e-11
#: Floor on the derived relative tolerance: below this ARPACK is back to grinding for digits
#: nothing reads. There is no ceiling because the energy scale is floored at 1 Eh, so the
#: derived tolerance can never exceed :data:`EIGSH_TARGET_EH` itself.
EIGSH_MIN_TOL = 1.0e-15
#: Lanczos subspace size, as a multiple of the number of roots and as an absolute floor. The
#: SciPy default (``max(2k+1, 20)``) is what actually causes the stalling above: clustered
#: eigenvalues need room to separate. Costs ``ncv * ndet`` complex numbers.
EIGSH_NCV_FACTOR = 4
EIGSH_NCV_MIN = 48
#: Iteration cap, as a multiple of the space size. Deliberately *tighter* than SciPy's ``10n``
#: default: with the dense fallback below, failing early is cheaper than grinding on.
EIGSH_MAXITER_FACTOR = 5


# --- Reference space -----------------------------------------------------------------------

def _reference_window(h_diag: np.ndarray, n_spinor: int, n_elec: int,
                      max_reference: int) -> Tuple[np.ndarray, int]:
    """Choose the near-degenerate window around the Fermi level for the reference CAS.

    Grows a window symmetrically about the Fermi level until the complete active space it
    spans would exceed ``max_reference`` determinants. Ordering is by the diagonal of the
    effective one-electron Hamiltonian, which stands in for orbital energies — it is exactly
    that for canonical orbitals and a reasonable proxy otherwise.
    """
    order = np.argsort(np.real(h_diag))
    occupied = order[:n_elec]
    virtual = order[n_elec:]
    from math import comb

    best = (np.array([], dtype=int), 0)
    for k in range(1, min(occupied.size, virtual.size) + 1):
        window = np.concatenate([occupied[-k:], virtual[:k]])
        if comb(window.size, k) > max_reference:
            break
        best = (np.sort(window), k)
    return best


def reference_determinants(h_diag: np.ndarray, n_spinor: int, n_elec: int, *,
                           max_reference: int = DEFAULT_MAX_REFERENCE) -> Determinants:
    """A small complete active space around the Fermi level (see the module docstring)."""
    from itertools import combinations

    window, n_win_elec = _reference_window(h_diag, n_spinor, n_elec, max_reference)
    order = np.argsort(np.real(h_diag))
    frozen = [int(p) for p in order[:n_elec] if p not in set(window.tolist())]
    if window.size == 0:                       # degenerate case: a single determinant
        return Determinants.from_occupations([sorted(order[:n_elec].tolist())], n_spinor)
    occ_lists = [sorted(frozen + [int(window[i]) for i in combo])
                 for combo in combinations(range(window.size), n_win_elec)]
    log.debug("reference CAS: %d electrons in %d spinors -> %d determinants",
              n_win_elec, window.size, len(occ_lists))
    return Determinants.from_occupations(occ_lists, n_spinor)


# --- Candidate generation ---------------------------------------------------------------------

_U64 = np.uint64


def _excitations(mask: int, n_spinor: int, max_excitation: int) -> np.ndarray:
    """All determinant masks reachable from ``mask`` by 1 or 2 excitations."""
    occ = np.array([p for p in range(n_spinor) if (mask >> p) & 1], dtype=np.int64)
    vir = np.array([p for p in range(n_spinor) if not (mask >> p) & 1], dtype=np.int64)
    m = _U64(mask)
    bit_o = _U64(1) << occ.astype(_U64)
    bit_v = _U64(1) << vir.astype(_U64)
    out_masks = [((m ^ bit_o[:, None]) | bit_v[None, :]).ravel()]
    if max_excitation >= 2 and occ.size >= 2 and vir.size >= 2:
        oi, oj = np.triu_indices(occ.size, k=1)
        vi, vj = np.triu_indices(vir.size, k=1)
        rem = m ^ (bit_o[oi] | bit_o[oj])                       # (npair_o,)
        add = bit_v[vi] | bit_v[vj]                             # (npair_v,)
        out_masks.append((rem[:, None] | add[None, :]).ravel())
    return np.concatenate(out_masks)


def _select_space(space: Determinants, vec: np.ndarray, energies: np.ndarray,
                  h: np.ndarray, eri: np.ndarray, max_dets: int, max_excitation: int,
                  max_generators: int,
                  ensemble_weights: Optional[np.ndarray] = None) -> Determinants:
    """Grow the space by perturbatively ranked single and double excitations.

    Candidates are generated from at most ``max_generators`` **leading** determinants rather
    than from the whole space. That is the ASCI construction, and it is what keeps the cost
    bounded: generating from every determinant would produce ``|space| * n_o^2 n_v^2``
    candidates and make the selection more expensive than the CI it is preparing. The full
    space is still *kept* — only the generator set is truncated.

    ``ensemble_weights`` carries :func:`cheap_ci`'s ``ensemble_selection``: an array ranks
    against the state-averaged ensemble the *objective* uses, which is the default whenever
    more than one root is averaged; ``None`` ranks against the ground root alone.
    """
    n = space.n_spinor
    n_room = max_dets - space.ndet
    if n_room <= 0:
        return space
    vecs = vec if vec.ndim > 1 else vec[:, None]
    if ensemble_weights is None:
        amp = np.abs(vecs[:, 0])
    else:
        amp = np.sqrt(np.abs(vecs) ** 2 @ ensemble_weights)
    gen_idx = np.argsort(-amp)[:min(max_generators, space.ndet)]
    with timer("candidate generation"):
        cand = np.unique(np.concatenate(
            [_excitations(int(space.masks[g]), n, max_excitation) for g in gen_idx]))
        cand = cand[space.positions(cand) < 0]              # drop what we already have
    if cand.size == 0:
        return space
    if cand.size <= n_room:
        keep = cand
    else:
        with timer("perturbative selection"):
            keep = _rank_candidates(space, vec, energies, cand, h, eri, n_room, gen_idx,
                                    ensemble_weights)
    return Determinants(masks=np.concatenate([space.masks, keep]), n_spinor=n,
                        n_elec=space.n_elec)


def _rank_candidates(space: Determinants, vec: np.ndarray, energies: np.ndarray,
                     cand: np.ndarray, h: np.ndarray, eri: np.ndarray,
                     n_keep: int, gen_idx: np.ndarray,
                     ensemble_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Keep the ``n_keep`` candidates with the largest first-order perturbative weight.

    Weight ``|sum_I c_I <D|H|I>|^2 / |H_DD - E|`` over the generators ``I`` — the standard
    Epstein-Nesbet / CIPSI criterion. The generators are placed **first** in a joint list so
    ``connections(row_limit=n_gen)`` performs a rectangular ``n_gen x n_candidates`` search
    instead of a quadratic one; that single detail is the difference between seconds and
    hours once the candidate pool passes 10^5.

    With ``ensemble_weights`` — :func:`cheap_ci`'s ``ensemble_selection``, and the **default**
    whenever more than one root is averaged — the weight is summed over the averaged roots
    with their state weights, each root against its own denominator. ``None`` ranks against
    the ground root alone, which is what this did originally and is now the setting that has
    to be asked for: with several states averaged it keys the selection on a different
    quantity than the objective, and that is a measured inconsistency, not a stylistic one.
    On the Ti(2+) benchmark ranking against the ensemble **halves** the surface-to-surface
    gradient noise and lowers the state-averaged energy on every case tried.

    ⚠ **Halving the noise does not remove it, and no amount of selection hygiene will.** Any
    argmax criterion has decision boundaries, and the trajectory crossing one is the jump.
    That is what the event gating of :mod:`kuiva.mcscf.events` is for; this is complementary
    to it, not an alternative.
    """
    vecs = vec if vec.ndim > 1 else vec[:, None]
    e = np.atleast_1d(np.asarray(energies, dtype=float))
    if ensemble_weights is None:
        vecs, e, w = vecs[:, :1], e[:1], np.ones(1)
    else:
        w = np.asarray(ensemble_weights, dtype=float)
    gen_masks = space.masks[gen_idx]
    n_gen = gen_masks.size
    joint = Determinants(masks=np.concatenate([gen_masks, cand]),
                         n_spinor=space.n_spinor, n_elec=space.n_elec)
    conn = connections(joint, row_limit=n_gen)
    hmat = hamiltonian_matrix(joint, h, eri, conn)
    jvec = np.zeros((joint.ndet, e.size), dtype=np.complex128)
    jvec[:n_gen] = vecs[gen_idx]
    amp = hmat @ jvec                                       # (njoint, nstate)
    denom = diagonal_energies(joint, h, eri)[:, None] - e[None, :]
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    weight = (np.abs(amp) ** 2 / np.abs(denom)) @ w
    weight[:n_gen] = -np.inf                                # generators are already in
    chosen = np.argsort(-weight)[:n_keep]
    return joint.masks[chosen]


# --- The cheap CI ------------------------------------------------------------------------------

@dataclass
class CheapCIResult:
    """Outcome of one cheap-CI solve in a fixed active space."""

    energies: np.ndarray                  # (nstate,) [Eh] — qualitative only, see the module docstring
    civecs: np.ndarray                    # (ndet, nstate)
    dets: Determinants
    gamma: np.ndarray                     # state-averaged 1-RDM
    gamma2: Optional[np.ndarray]          # state-averaged 2-RDM
    occupation_correlation: np.ndarray    # <n_p n_q>
    weights: np.ndarray

    @property
    def n_determinants(self) -> int:
        return self.dets.ndet

    @property
    def leading_weight(self) -> float:
        """Weight of the largest determinant in the ground state — a multireference gauge."""
        return float(np.max(np.abs(self.civecs[:, 0]) ** 2))

    def natural_spinors(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(occupations, rotation)`` of the state-averaged 1-RDM, occupation descending.

        ``rotation`` is the unitary to apply to the **active columns of the orbital
        coefficients**, ``C_act -> C_act @ rotation``, after which the 1-RDM is
        ``diag(occupations)``. Occupations run from 1 (a fully occupied spinor — one electron)
        to 0.

        .. note::
           The rotation is ``conj(V)``, not ``V``, for ``gamma = V diag(n) V^dag``. Orbital
           coefficients and density matrices transform *oppositely*: under
           ``|p'> = sum_p |p> U_{pp'}`` the 1-RDM goes to ``U^T gamma U*``, so diagonalizing it
           requires ``U = conj(V)``. Using ``V`` leaves gamma non-diagonal in a way that looks
           plausible — the occupations are still bounded and sum correctly — and it is only
           visible if you check that the *diagonal* of the rotated gamma equals its spectrum.
           A test does exactly that. The same trap appears in
           :func:`kuiva.mcscf.orbopt._active_fock`.
        """
        w, v = np.linalg.eigh(self.gamma)
        order = np.argsort(-w)
        return np.clip(w[order], 0.0, 1.0), np.conj(v[:, order])

    def entanglement(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(s1, I)`` — single-orbital entropies and mutual information."""
        return (single_orbital_entropy(self.gamma),
                mutual_information(self.gamma, self.occupation_correlation))


def cheap_ci(h_eff: np.ndarray, eri: np.ndarray, n_elec: int, *, n_states: int = 1,
             max_determinants: int = DEFAULT_MAX_DETERMINANTS,
             max_reference: int = DEFAULT_MAX_REFERENCE,
             max_excitation: int = 2, selection_rounds: int = 2,
             max_generators: int = DEFAULT_MAX_GENERATORS,
             state_weights: Optional[Sequence[float]] = None,
             ensemble_selection: bool = True,
             with_2rdm: bool = True, report: bool = False) -> CheapCIResult:
    """Selected multireference CISD in the active space (see the module docstring).

    ``h_eff`` is the active-space effective one-electron Hamiltonian (the inactive Fock
    restricted to active, :meth:`~kuiva.mcscf.orbopt.CASIntegrals.h_active_effective`) and
    ``eri`` the active four-index integrals; energies are therefore relative to the inactive
    core and are *not* total energies.

    ``ensemble_selection`` (**on**) ranks the perturbative selection against all averaged
    roots rather than the ground one — see :func:`_rank_candidates`. It is on because the
    objective *is* the state average, and it lowers that average measurably. Turning it off
    recovers the original criterion. ⚠ With ``n_states == 1`` it is bitwise a no-op, by
    construction rather than by luck.
    """
    n_spinor = int(h_eff.shape[0])
    weights = (np.full(n_states, 1.0 / n_states) if state_weights is None
               else np.asarray(state_weights, dtype=float) / np.sum(state_weights))
    if weights.size != n_states:
        raise ValueError("state_weights has {} entries for {} states".format(
            weights.size, n_states))

    # The reference CAS can never exceed the total budget: it is the seed, not the space.
    dets = reference_determinants(np.diag(h_eff), n_spinor, n_elec,
                                  max_reference=min(max_reference, max_determinants))
    # Solve the reference CAS before selecting anything: the perturbative weight needs a
    # meaningful E_ref in its denominator and meaningful coefficients on the generators, and
    # the reference is small enough (a few hundred determinants) that this is free.
    energies, civecs = _solve(dets, h_eff, eri, n_states)

    # With one root the ensemble *is* the ground state, so the two criteria are the same
    # quantity — take the single-root path explicitly, so that a one-state calculation is
    # bitwise unaffected by this option rather than merely almost unaffected.
    sel_weights = weights if (ensemble_selection and n_states > 1) else None
    for rnd in range(max(1, selection_rounds)):
        grown = _select_space(dets, civecs, energies, h_eff, eri,
                              max_determinants, max_excitation, max_generators, sel_weights)
        if grown.ndet == dets.ndet:
            break                                        # nothing new survived selection
        dets = grown
        energies, civecs = _solve(dets, h_eff, eri, n_states)
        log.debug("selection round %d: %d determinants, E0 = %.8f Eh",
                  rnd + 1, dets.ndet, energies[0])

    conn = connections(dets)
    gamma, gamma2 = rdm12(dets, civecs, weights, conn, with_2rdm=with_2rdm)
    nn = occupation_correlations(dets, civecs, weights)
    result = CheapCIResult(energies=energies, civecs=civecs, dets=dets, gamma=gamma,
                           gamma2=gamma2, occupation_correlation=nn, weights=weights)
    if report:
        _report_ci(result, n_elec)
    return result


def solve_fixed_space(dets: Determinants, h_eff: np.ndarray, eri: np.ndarray, *,
                      n_states: int = 1, state_weights: Optional[Sequence[float]] = None,
                      conn=None, with_2rdm: bool = True) -> CheapCIResult:
    """Solve the CI in a **given** determinant space — no selection, no adaptation.

    This is what makes the cheap CI usable inside an orbital optimizer. :func:`cheap_ci`
    re-chooses its determinants for every set of integrals it is handed, so the energy it
    returns is not a smooth function of the orbitals: the space jumps, and with it the
    surface. Measured on Ti(2+), freezing the space improves the converged
    gradient by 3.8x for the quasi-Newton step and 8x for the second-order one, and lowers the
    energy in both cases — a Newton method suffers most, since it trusts a quadratic model of
    whatever surface it is given.

    Passing ``conn`` (from :func:`kuiva.ci.strings.connections`) reuses the pair search, which
    is the expensive part and depends only on the determinants, not on the integrals.
    """
    weights = (np.full(n_states, 1.0 / n_states) if state_weights is None
               else np.asarray(state_weights, dtype=float) / np.sum(state_weights))
    conn = connections(dets) if conn is None else conn
    energies, civecs = _solve(dets, h_eff, eri, n_states, conn=conn)
    gamma, gamma2 = rdm12(dets, civecs, weights, conn, with_2rdm=with_2rdm)
    nn = occupation_correlations(dets, civecs, weights)
    return CheapCIResult(energies=energies, civecs=civecs, dets=dets, gamma=gamma,
                         gamma2=gamma2, occupation_correlation=nn, weights=weights)


def dense_hamiltonian_gb(ndet: int) -> float:
    """Memory of **one** dense CI Hamiltonian, ``(ndet, ndet)`` complex."""
    return res.array_gb((ndet, ndet), np.complex128)


def _eigsh_tol(hmat) -> float:
    """ARPACK's relative ``tol`` for an absolute accuracy of :data:`EIGSH_TARGET_EH`.

    ARPACK measures convergence relative to ``|lambda|``, so the *useful* tolerance depends on
    how large the eigenvalue is. The scale is taken from the smallest diagonal element of the
    CI Hamiltonian, which is free (the matrix is already built) and bounds the lowest
    eigenvalue from above in magnitude closely enough for this purpose.
    """
    scale = max(abs(float(np.min(np.real(hmat.diagonal())))), 1.0)
    return float(max(EIGSH_MIN_TOL, EIGSH_TARGET_EH / scale))


def _dense_eigh(hmat, ndet: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Full diagonalization of the CI Hamiltonian, keeping the lowest ``k`` roots."""
    res.require("dense CI Hamiltonian", 2.0 * dense_hamiltonian_gb(ndet),
                note="{} determinants; the matrix plus LAPACK's copy".format(ndet),
                advice=["lower max_determinants",
                        "raise the memory limit if a dense solve is wanted at this size"])
    w, v = np.linalg.eigh(hmat.toarray())
    return w[:k], v[:, :k]


def _solve(dets: Determinants, h: np.ndarray, eri: np.ndarray, n_states: int, conn=None
           ) -> Tuple[np.ndarray, np.ndarray]:
    """Lowest ``n_states`` eigenpairs of the CI Hamiltonian.

    Dense for small spaces, ARPACK (``eigsh``) otherwise. The design sanctions leaning on an external
    eigensolver "initially if convenient"; the custom complex Davidson of ``ci/davidson.py``
    is for the full CI, where the matrix is never formed at all. Here it *is* formed,
    once, and a Lanczos solve on it is a few lines instead of a few hundred.

    ⚠ **A Lanczos solve can fail, and this one did.** With ARPACK's defaults — machine-
    precision tolerance and a 20-vector subspace — the near-degenerate lowest pair of a
    state-averaged Kramers-degenerate calculation either grinds for minutes or raises
    ``ArpackNoConvergence`` outright, at determinant counts as low as 1000. Three things
    change that, in order of importance: a subspace large enough for the cluster to separate
    (:data:`EIGSH_NCV_MIN` — this is the one that does most of the work), a tolerance matched
    to the accuracy anything downstream can use rather than to machine precision
    (:func:`_eigsh_tol`), and — when it still fails — a **dense fallback**, which is exact and
    cannot fail, up to :data:`DENSE_FALLBACK_MAX_DET`. Beyond that size the last resort is one
    loosened retry and then :class:`~kuiva.mcscf.adaptive.SolverFailure`, which the
    event-gated optimizer of :mod:`kuiva.mcscf.events` turns into a rejected step rather than
    a dead calculation.
    """
    from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigsh

    hmat = hamiltonian_matrix(dets, h, eri, conn)
    ndet = dets.ndet
    k = min(n_states, ndet)

    def lanczos(tol: float, ncv: int) -> Tuple[np.ndarray, np.ndarray]:
        w_, v_ = eigsh(hmat, k=k, which="SA", tol=tol, ncv=ncv,
                       maxiter=EIGSH_MAXITER_FACTOR * ndet)
        order = np.argsort(w_)
        return w_[order], v_[:, order]

    with timer("CI diagonalization"):
        if ndet <= DENSE_SOLVE_MAX_DET or k >= ndet - 1:
            w, v = _dense_eigh(hmat, ndet, k)
        else:
            ncv = min(ndet, max(EIGSH_NCV_FACTOR * k + 1, EIGSH_NCV_MIN))
            tol = _eigsh_tol(hmat)
            try:
                w, v = lanczos(tol, ncv)
            except (ArpackNoConvergence, ArpackError) as exc:
                log.warning("ARPACK did not converge the lowest %d roots of %d determinants "
                            "(%s); falling back", k, ndet, type(exc).__name__)
                if ndet <= DENSE_FALLBACK_MAX_DET:
                    w, v = _dense_eigh(hmat, ndet, k)
                else:
                    try:                  # a wider subspace and a tolerance we can live with
                        w, v = lanczos(1e-8, min(ndet, 4 * ncv))
                    except (ArpackNoConvergence, ArpackError) as exc2:
                        raise SolverFailure(
                            "the CI in {} determinants did not converge ({}), and the space "
                            "is past the {}-determinant dense fallback".format(
                                ndet, exc2, DENSE_FALLBACK_MAX_DET))
                    log.warning("the loosened retry converged; these roots carry a 1e-8 "
                                "relative tolerance, not %.0e", tol)
    if k < n_states:
        log.warning("only %d determinants for %d requested states; the space is too small",
                    dets.ndet, n_states)
        w = np.concatenate([w, np.full(n_states - k, w[-1])])
        v = np.concatenate([v, np.zeros((v.shape[0], n_states - k))], axis=1)
    return np.real(w), np.asarray(v, dtype=np.complex128)


def _report_ci(ci: CheapCIResult, n_elec: int) -> None:
    occ, _ = ci.natural_spinors()
    out.entries(log, [
        ("determinants", ci.n_determinants),
        ("states averaged", int(ci.energies.size)),
        ("leading determinant weight", ci.leading_weight, "", "", "{:.4f}"),
        ("lowest root (active space only)", float(ci.energies[0]), "Eh",
         "not a total energy", out.E_FMT),
        ("trace of the 1-RDM", float(np.real(np.trace(ci.gamma))), "electrons", "",
         "{:.6f}"),
        ("partially occupied spinors", int(np.count_nonzero(
            (occ > DEFAULT_OCCUPATION_WINDOW[0]) & (occ < DEFAULT_OCCUPATION_WINDOW[1]))),
         "", "0.02 < n < 0.98"),
    ])


# --- The cheap CI as an adaptive solver ---------------------------------------------------------

class CheapCISolver:
    """The selected cheap CI behind the :class:`~kuiva.mcscf.adaptive.AdaptiveCISolver`
    contract, for the event-gated optimizer of :mod:`kuiva.mcscf.events`.

    The split is exactly the one :func:`solve_fixed_space` already documents, made explicit
    and put under the optimizer's control:

    * :meth:`solve` never selects. The first call chooses the space (there has to be one) and
      every call after it solves in that space, so ``E(kappa)`` is the smooth, deterministic
      surface the trust region and the quadratic model assume.
    * :meth:`propose` runs a full :func:`cheap_ci` at the *current* integrals and hands the
      result back **without adopting it**. The optimizer compares it against the incumbent at
      those same integrals and adopts only a genuine variational improvement.

    Measured on the Ti(2+) CAS(10,18) benchmark, the fresh
    selection beat the incumbent **once in five proposals**, and gating on that one adoption
    took the converged gradient from 8.7e-2 — a hard stall, per-iteration re-selection — to
    1.1e-4, at 16% *less* work than simply freezing the space. Per-iteration re-selection is
    almost pure noise; it is the rare real improvement that is worth having, and only if the
    optimizer is told when it happens.

    ⚠ **The proposal is only comparable at the integrals it was made at.** Both energies
    include ``ints.e_core``, and they must come from the same :class:`CASIntegrals`; comparing
    a candidate energy from one point against an incumbent from another compares two orbital
    sets, not two spaces.
    """

    def __init__(self, n_elec: int, *, n_states: int = 1, **ci_kwargs):
        self.n_elec = int(n_elec)
        self.n_states = int(n_states)
        self.ci_kwargs = dict(ci_kwargs)
        self._dets: Optional[Determinants] = None
        self._conn = None
        self._key: Optional[str] = None
        self._candidate: Optional[Tuple[str, CheapCIResult]] = None
        #: The most recent solve, for callers that want the wavefunction (occupations,
        #: entanglement) rather than just the RDMs the optimizer asked for.
        self.last: Optional[CheapCIResult] = None
        self.n_solves = 0                 # fixed-space solves
        self.n_selections = 0             # full selections: the expensive call

    # -- the contract -------------------------------------------------------------------
    def solve(self, ints: CASIntegrals):
        h_eff, eri = ints.h_active_effective(), ints.active_eri()
        if self._dets is None:
            ci = cheap_ci(h_eff, eri, self.n_elec, n_states=self.n_states, **self.ci_kwargs)
            self.n_selections += 1
            self._install(ci.dets)
        else:
            ci = solve_fixed_space(self._dets, h_eff, eri, n_states=self.n_states,
                                   conn=self._conn,
                                   state_weights=self.ci_kwargs.get("state_weights"),
                                   with_2rdm=self.ci_kwargs.get("with_2rdm", True))
        self.n_solves += 1
        self.last = ci
        return self._energy(ci, ints), ci.gamma, ci.gamma2

    def propose(self, ints: CASIntegrals) -> Optional[Proposal]:
        h_eff, eri = ints.h_active_effective(), ints.active_eri()
        ci = cheap_ci(h_eff, eri, self.n_elec, n_states=self.n_states, **self.ci_kwargs)
        self.n_selections += 1
        key = array_key(ci.dets.masks)
        if key == self._key:
            return None                    # the selection reproduced the incumbent space
        self._candidate = (key, ci)
        return Proposal(energy=self._energy(ci, ints), gamma=ci.gamma, gamma2=ci.gamma2,
                        key=key, label=self._overlap_label(ci.dets))

    def adopt(self, key) -> None:
        if self._candidate is None or self._candidate[0] != key:
            raise ValueError("no proposal with key {!r} is pending; adopt() takes the key of "
                             "the most recent propose()".format(key))
        _, ci = self._candidate
        self._install(ci.dets)
        self.last = ci
        self._candidate = None

    def space_key(self) -> Optional[str]:
        return self._key

    # -- internals ----------------------------------------------------------------------
    def _install(self, dets: Determinants) -> None:
        """Make ``dets`` the incumbent space. The connection search is the expensive part and
        depends only on the determinants, so it is done once here rather than per solve."""
        self._dets = dets
        self._conn = connections(dets)
        self._key = array_key(dets.masks)
        self._candidate = None

    def _energy(self, ci: CheapCIResult, ints: CASIntegrals) -> float:
        return float(np.dot(ci.weights, ci.energies)) + ints.e_core

    def _overlap_label(self, dets: Determinants) -> str:
        if self._dets is None:
            return "{} determinants".format(dets.ndet)
        shared = int(np.intersect1d(dets.masks, self._dets.masks).size)
        return "{}/{} shared".format(shared, dets.ndet)

    @property
    def n_determinants(self) -> int:
        return 0 if self._dets is None else self._dets.ndet

    def __repr__(self) -> str:
        return "CheapCISolver(n_elec={}, n_states={}, ndet={}, key={})".format(
            self.n_elec, self.n_states, self.n_determinants,
            None if self._key is None else self._key[:8])


# --- Driving the orbital optimizer -------------------------------------------------------------

@dataclass
class PreoptResult:
    """Pre-optimized orbitals, plus what the active space and the tensor network need."""

    coeff: np.ndarray                     # (2*nao, n) spinor coefficients, natural in the active space
    spaces: OrbitalSpaces
    ci: CheapCIResult
    natural_occupation: np.ndarray        # (n_active,) eigenvalues of gamma, descending
    orbital_occupation: np.ndarray        # (n_active,) occupation OF each returned orbital
    entropy: np.ndarray                   # (n_active,) single-orbital entropies
    mutual_information: np.ndarray        # (n_active, n_active)
    energy: float
    converged: bool
    history: List[float] = field(default_factory=list)

    def suggest_active_space(self, window: Tuple[float, float] = DEFAULT_OCCUPATION_WINDOW,
                             entropy_threshold: float = 0.0) -> np.ndarray:
        """Active spinors worth keeping: fractionally occupied, optionally also entangled.

        Returns indices **into the current orbital set**, so the result is
        directly usable as ``OrbitalSpaces.active`` for the real CASSCF or DMRG. A spinor at
        occupation 1 or 0 carries no correlation and belongs in the inactive or virtual space;
        the entropy threshold is the sharper criterion of Stein & Reiher and is offered as an
        additional filter, off by default because it needs the same judgement call about where
        to cut.

        .. warning::
           **This finds orbitals that are partially *occupied*; it cannot find an empty
           orbital that a later, better treatment would populate.** For the strongly
           correlated open shells this code targets — antiferromagnetically coupled centres,
           multi-electron d and f shells — that is exactly the right criterion, and it works:
           on TiCl3 (d1) the suggestion is the two spinors carrying occupation 0.5005/0.5006,
           i.e. precisely the Kramers doublet of the single d electron. But the *empty*
           members of that d manifold come back at ~1e-4 and are not suggested, because at
           this level of correlation nothing populates them. A single electron in a
           near-degenerate manifold is the clear case: occupation-based selection returns the
           doublet, not the manifold.

           So treat the result as a **lower bound** on the active space, to be combined with
           orbital character and with the near-degeneracy structure — which is what a reproducible definition
           requires of an active-space definition anyway ("stated in physical terms, not as
           an orbital-index window"). The entanglement data is the complementary half: it
           says which of the selected spinors are connected, and to what.

        Selection uses :attr:`orbital_occupation` — the occupation *of each returned orbital*
        — and not the eigenvalue spectrum :attr:`natural_occupation`. The two coincide only
        when the returned basis is exactly natural, and for a *truncated* CI it is not: the
        selected determinant space depends on the basis, so re-solving after the
        natural-spinor rotation gives a slightly different state. Using the spectrum would
        silently pair occupation numbers with the wrong orbital indices.
        """
        occ = self.orbital_occupation
        keep = (occ > window[0]) & (occ < window[1])
        if entropy_threshold > 0.0:
            keep &= self.entropy > entropy_threshold
        return self.spaces.active[np.nonzero(keep)[0]]

    def dmrg_ordering(self) -> np.ndarray:
        """Fiedler ordering of the active spinors for the initial MPS."""
        return fiedler_order(self.mutual_information)


#: How the determinant space is allowed to move during the orbital optimization. One axis,
#: three values, because "who owns the space" is a single question:
#:
#: ``"event"``
#:     The optimizer owns it (:func:`kuiva.mcscf.events.optimize_orbitals_events`): the space
#:     is held fixed between *events*, and at an event a fresh selection is adopted only if it
#:     genuinely lowers the energy at the same integrals. The default — see
#:     :func:`preoptimize` for the measurement.
#: ``"frozen"``
#:     Select once from the starting orbitals and never again. This is event gating with the
#:     event cadence set to "never", so the old ``freeze_determinants=True`` behaviour stays
#:     exactly expressible.
#: ``"adaptive"``
#:     Re-select at every point, the old ``freeze_determinants=False``. Kept because an
#:     optimizer that makes no smoothness assumption would be entitled to it; ⚠ it is what
#:     stalls the second-order step at ``|g| = 1.9e-1``.
SPACE_POLICIES = ("event", "frozen", "adaptive")


def preoptimize(factors: ThreeIndexAO, h_ao: np.ndarray, c_spinor: np.ndarray,
                spaces: OrbitalSpaces, n_active_elec: int, *, e_nuc: float = 0.0,
                n_states: int = 1, max_iter: int = 20, report: bool = True,
                natural_spinors: bool = True, mode: str = "quasi-newton",
                conv_grad: float = 1e-3, space_policy: str = "event",
                freeze_determinants: Optional[bool] = None,
                tau: float = DEFAULT_TAU, event_interval: int = DEFAULT_EVENT_INTERVAL,
                **ci_kwargs) -> PreoptResult:
    """Run the cheap CI inside the shared orbital optimizer, then analyse the result.

    This is the pre-optimization entry point, and it is deliberately a thin composition: the cheap CI is
    handed to :func:`kuiva.mcscf.orbopt.optimize_orbitals` as its ``ci_solver`` callback, so
    the orbital optimization here is *the same code* that a CASSCF or a DMRG-CASSCF will use
. Swapping the callback is the only difference between them.

    The active-space orbitals are finally rotated to **natural spinors**, which fixes the
    otherwise arbitrary basis inside the active space, orders it by occupation, and is the
    form both consumers (CASSCF start, network seed) want.

    **The optimizer defaults differ from the general ones on purpose:** ``mode="quasi-newton"``
    and a loose ``conv_grad`` of 1e-3. A pre-optimizer is not trying to find a stationary
    point — it is trying to produce sensible orbitals cheaply, and its answers (natural
    occupations, entanglement) are qualitative by construction. Letting it escalate to
    exact Hessian-vector products would spend production-CASSCF effort on a starting guess.
    Pass ``mode="auto"`` if a pre-optimization genuinely refuses to make progress.

    ``space_policy`` decides **who owns the determinant space** (:data:`SPACE_POLICIES`), and
    the default is ``"event"``. Re-selecting at every point makes ``E(kappa)`` discontinuous
    and is what the whole mechanism exists to remove; simply *freezing* the space removes the
    discontinuity but also locks in whatever was chosen from the starting orbitals, which may
    not be the right space once the orbitals have moved. Event gating is both: the space is
    fixed between events, and at an event a fresh selection is adopted only when it lowers the
    energy at the same integrals by more than ``tau``. Measured on the Ti(2+) CAS(10,18) proxy
    at 30 macro-iterations (measured):

    ====================  ==========  ==============  ========
    space policy          ``|g|``     ``E`` [Eh]      work
    ====================  ==========  ==============  ========
    ``"adaptive"``        8.7e-2      -852.19925       130
    ``"frozen"``          2.4e-4      -852.22548      1032
    ``"event"``           1.1e-4      **-852.23368**   872
    ====================  ==========  ==============  ========

    Event gating wins on all three columns at once — a lower gradient, 8.2 mEh lower energy
    and *less* work than freezing — and it does so with **one adoption in five proposals**.
    Per-iteration re-selection is almost pure noise; it is the rare genuine improvement that
    is worth having, and only if the optimizer is told when it happens. ⚠ The re-selecting
    row's digits are knife-edge sensitive and mean nothing; that it stops two to three orders
    of magnitude short is the content.

    ``freeze_determinants`` is the previous spelling and still works: ``True`` means
    ``space_policy="frozen"``, ``False`` means ``"adaptive"``. It overrides ``space_policy``
    when given, so existing callers keep their exact behaviour.

    ``tau`` and ``event_interval`` are the event knobs; see
    :func:`kuiva.mcscf.events.optimize_orbitals_events`. ⚠ The pre-optimizer's own
    ``conv_grad`` of 1e-3 is looser than the gradient at which events start to matter, so on
    an easy system the default policy costs one extra selection and changes nothing else.

    The final analysis solve, after the natural-spinor rotation, always re-selects: it is one
    solve rather than an optimization surface, and the rotated basis deserves a space chosen
    for it.

    ⚠ **The returned orbitals are not exactly Kramers paired, and a consumer that needs them
    to be must repair them.** A truncated CI in a truncated space legitimately drifts off
    pairing, and the state-averaging gate downstream assumes it holds; repair with
    :func:`kuiva.spinor.expand.nearest_kramers_paired`, applied per orbital space so no pair
    crosses a space boundary. The :class:`kuiva.CheapCI` stage does exactly that and is the
    reason a script built on the stage API never meets this.
    """
    if freeze_determinants is not None:
        space_policy = "frozen" if freeze_determinants else "adaptive"
    if space_policy not in SPACE_POLICIES:
        raise ValueError("space_policy must be one of {}; got {!r}"
                         .format(SPACE_POLICIES, space_policy))
    if report:
        out.section(log, "Cheap CI pre-optimization")
        out.entries(log, [
            ("active electrons", n_active_elec),
            ("active spinors", spaces.n_active),
            ("states", n_states),
            ("determinant budget", ci_kwargs.get("max_determinants",
                                                 DEFAULT_MAX_DETERMINANTS)),
        ])
        out.note(log, "energies below are qualitative: a truncated CI in a truncated space")

    # One implementation of the CI behind all three policies: `CheapCISolver.solve` selects
    # once and then holds the space, which *is* "frozen"; "event" hands the whole solver to
    # the controller so it can also propose; "adaptive" re-selects by construction. Sharing
    # the implementation is what makes the policies comparable — a separate frozen code path
    # would differ from the event one in ways nothing measures.
    solver = CheapCISolver(n_active_elec, n_states=n_states, **ci_kwargs)

    if space_policy == "event":
        opt = optimize_orbitals_events(factors, h_ao, c_spinor, spaces, solver, e_nuc=e_nuc,
                                       max_iter=max_iter, mode=mode, conv_grad=conv_grad,
                                       n_active_elec=n_active_elec,
                                       tau=tau, event_interval=event_interval, report=report)
    else:
        def callable_solver(ints: CASIntegrals):
            if space_policy == "adaptive":
                res = cheap_ci(ints.h_active_effective(), ints.active_eri(), n_active_elec,
                               n_states=n_states, **ci_kwargs)
                solver.last = res
                return (float(np.dot(res.weights, res.energies)) + ints.e_core,
                        res.gamma, res.gamma2)
            return solver.solve(ints)

        opt = optimize_orbitals(factors, h_ao, c_spinor, spaces, callable_solver, e_nuc=e_nuc,
                                max_iter=max_iter, mode=mode, conv_grad=conv_grad,
                                n_active_elec=n_active_elec, report=report)
    coeff = opt.coeff
    if natural_spinors:
        coeff = coeff.copy()
        coeff[:, spaces.active] = coeff[:, spaces.active] @ solver.last.natural_spinors()[1]

    # ⚠ The analysis solve is done at the **returned** orbitals, always, and not read off the
    # optimizer's last call. Two reasons, and each is enough on its own. Orbital entropies are
    # properties of a *mode partition*, so they are only meaningful in the basis they are
    # measured in, and the basis that matters is the one handed downstream — reporting them
    # from the pre-rotation basis would give the DMRG an ordering for a basis it never sees.
    # And the optimizer's last solve is at whatever point it evaluated last, which after a
    # *rejected* step is a trial point that was thrown away; analysing that is analysing
    # orbitals nobody receives. One extra solve, seconds, and it makes the returned
    # wavefunction correspond to the returned coefficients by construction.
    ints_final = CASIntegrals.build(factors, h_ao, coeff, spaces, e_nuc=e_nuc)
    ci = cheap_ci(ints_final.h_active_effective(), ints_final.active_eri(), n_active_elec,
                  n_states=n_states, **ci_kwargs)
    occ, _ = ci.natural_spinors()

    if report:
        out.subsection(log, "Active-space analysis")
        s1, info = entanglement_report(ci.gamma, ci.occupation_correlation,
                                       labels=spaces.active)
    else:
        s1 = single_orbital_entropy(ci.gamma)
        info = mutual_information(ci.gamma, ci.occupation_correlation)

    orbital_occ = np.clip(np.real(np.diag(ci.gamma)), 0.0, 1.0)
    result = PreoptResult(coeff=coeff, spaces=spaces, ci=ci, natural_occupation=occ,
                          orbital_occupation=orbital_occ, entropy=s1,
                          mutual_information=info, energy=opt.energy,
                          converged=opt.converged, history=opt.history)
    if report:
        keep = result.suggest_active_space()
        lo, hi = DEFAULT_OCCUPATION_WINDOW
        in_window = orbital_occ[(orbital_occ > lo) & (orbital_occ < hi)]
        out.entry(log, "suggested active spinors", int(keep.size), "",
                  "of {} candidates".format(spaces.n_active))
        out.entry(log, "suggested active electrons", float(in_window.sum()), "", "",
                  "{:.3f}")
    return result


__all__ = ["CheapCIResult", "CheapCISolver", "PreoptResult", "cheap_ci", "preoptimize",
           "reference_determinants", "solve_fixed_space", "dense_hamiltonian_gb",
           "SPACE_POLICIES",
           "DEFAULT_MAX_DETERMINANTS", "DEFAULT_MAX_REFERENCE", "DEFAULT_OCCUPATION_WINDOW",
           "DENSE_SOLVE_MAX_DET", "DENSE_FALLBACK_MAX_DET", "EIGSH_TARGET_EH"]
