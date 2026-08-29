"""``<S^2>`` over converged CI states: the multiplicity reading, and the spin-purity number.

What this is for
----------------
Reading a two-component spectrum is otherwise degeneracy counting by hand. ``<S^2>`` supplies
the piece counting cannot: with spin-orbit coupling **off** it is ``S(S+1)`` exactly, so the
term multiplicity ``2S+1`` is read straight off a block; with spin-orbit coupling **on** ``S``
is not a good quantum number at all, and the same number becomes a *diagnostic of spin purity*
— how much of a nominal ``^{2S+1}L_J`` level is really that spin. Both are computed here and
the second is always labelled as what it is. :mod:`kuiva.props.assign` is the consumer that
turns them into an assignment offer.

⚠ Per **degenerate block**, never per state
-------------------------------------------
Inside a degenerate block the eigensolver may return any unitary mixture of the members, so an
individual ``<I|S^2|I>`` is a property of that arbitrary choice and can differ run to run. What
is invariant is the block trace, and what is reported is therefore ``Tr_block(S^2) / size``.
Same discipline as the g values of :mod:`kuiva.props.multiplet` and the reduced populations of
:mod:`kuiva.props.population`, and for the same reason. The per-state values are kept on the
result for a caller that knows what it is doing, and the report does not print them.

How it is computed, and what could go wrong
-------------------------------------------
``S^2 = sum_k S_k S_k`` with ``S_k = sum_pq (s_k)_pq a^dag_p a_q`` is a **two**-body operator,
so the obvious route is one 2-RDM per state. It is not needed: each ``S_k`` is one-body, so

    <I|S_k S_k|I> = || S_k |I> ||^2                        (S_k Hermitian)

and ``S_k|I>`` is one contraction of the excitation map the CI already builds
(:meth:`kuiva.mcscf.casci.FullCISolver.one_body_moments`). One gather per state, three GEMVs.

⚠ **The norm is over the whole orbital space, and the CAS space is not all of it.** Splitting
``S_k`` by orbital space, acting on a CAS state whose inactive spinors are all occupied and
whose virtuals are all empty, exactly four pieces survive::

    S_k |I> = (S_k^{act} + t_k) |I>        in the CAS space; t_k = Tr_inactive(s_k)
            + sum_{t,i} B_ti  a^dag_t a_i |I>          B = s_k[active, inactive]
            + sum_{v,t} C_vt  a^dag_v a_t |I>          C = s_k[virtual, active]
            + sum_{v,i} D_vi  a^dag_v a_i |I>          D = s_k[virtual, inactive]

(the reversed blocks all annihilate: ``a^dag_i`` on a filled inactive spinor and ``a_v`` on an
empty virtual one both give zero). The four families have different occupation patterns, so
they are mutually orthogonal and the squares simply add::

    B term:  sum_i sum_tt' B_ti B*_t'i (delta_tt' - gamma_tt')
    C term:  sum_v sum_tt' C_vt C*_vt' gamma_t't
    D term:  ||D||_F^2                                 (state-independent)

with ``gamma_tu = <I|a^dag_t a_u|I>`` the state's active 1-RDM, which comes back from the same
call that gathers the map. ⚠ **They are computed and included, never assumed away** — the same
rule :func:`kuiva.props.dump.inactive_moment` follows for the inactive trace, and for the same
reason: ``B``, ``C`` and ``D`` vanish identically when each orbital space is spanned by whole
Kramers pairs of a spin-separable orbital set, which is the normal case and is exactly the
assumption worth *measuring* rather than trusting. Dropping them instead would give a
plausible number that is quietly too small — on a constructed non-separable case, too small by
93 % of itself.

⚠ **What is warned about is their CONTRIBUTION, not the matrix element.** A converged
general-complex CASSCF mixes spin — that is what spin-orbit coupling *is* — so ``B``, ``C`` and
``D`` are routinely ~1e-4 there and warning on that would fire on every spin-orbit run and say
nothing. :data:`SPIN_LEAKAGE_WARN` is therefore a threshold on how far the three terms move
``<S^2>`` itself. Both numbers reach the report either way.

References
----------
* ``S^2`` as a one- plus two-body operator and its use as a spin-contamination diagnostic:
  standard, e.g. A. Szabo, N. S. Ostlund, "Modern Quantum Chemistry", McGraw-Hill (1989),
  ch. 2.5 and 3.8.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util.logging import get_logger
from .dump import DEFAULT_INACTIVE_TOL, spinor_operator
from .multiplet import HARTREE_TO_CM, degenerate_blocks

log = get_logger(__name__)

#: ⚠ **Advisory, and set on the CORRECTION rather than on the matrix element.** A converged
#: general-complex CASSCF mixes spin — that is what spin-orbit coupling *is* — so its orbital
#: spaces are never exactly spin-separable and the ``|S_k|`` element connecting them is
#: routinely 1e-4. Warning on that would fire on every spin-orbit run and say nothing. What is
#: worth a warning is the out-of-space term moving ``<S^2>`` itself, so the threshold is on
#: that contribution, absolutely, on the ``S(S+1)`` scale.
#:
#: 0.01 is fixed here rather than measured: it shifts ``2S+1`` by about 0.005, well inside
#: :data:`kuiva.props.assign.SPIN_ROUND_TOL`, so anything below it cannot change a label. The
#: correction is *always* included whatever this is set to; this only decides who is told.
SPIN_LEAKAGE_WARN = 0.01


def spin_from_s_squared(s_squared: float) -> float:
    """``S`` from ``S(S+1)``, i.e. ``(sqrt(1 + 4 <S^2>) - 1) / 2``.

    ⚠ Negative input (which only roundoff around a singlet can produce) is clamped so that
    ``S`` comes back as exactly zero, rather than as ``nan`` or as a tiny negative number.
    A ``nan`` propagating into a multiplicity would read as "could not be measured"; a
    negative ``S`` would give ``2S+1`` just below one and a purity deviation out of nothing.
    """
    return max(0.0, 0.5 * (np.sqrt(max(0.0, 1.0 + 4.0 * float(s_squared))) - 1.0))


@dataclass(frozen=True)
class SpinAnalysis:
    """``<S^2>`` of a converged spectrum, blocked by degeneracy.

    Attributes
    ----------
    blocks : tuple of (start, size)
        The degenerate blocks, in energy order — the same grouping
        :func:`kuiva.props.multiplet.analyse_spectrum` makes, at the same tolerance.
    block_s_squared : ndarray (n_blocks,)
        ``Tr_block(S^2) / size`` — the invariant. **This is the number to read.**
    state_s_squared : ndarray (n_states,)
        Per state, in energy order. ⚠ Basis-dependent inside a degenerate block; kept for a
        caller that has a reason, and deliberately absent from :meth:`report`.
    energies_cm : ndarray (n_blocks,)
        Block energies relative to the ground state [cm^-1].
    leakage : float
        Largest ``|S_k|`` matrix element connecting two different orbital spaces [hbar]. Zero
        for a spin-separable orbital set (module docstring); its contribution is included in
        :attr:`block_s_squared` whatever it is. ⚠ On a converged spin-orbit CASSCF this is
        routinely ~1e-4 and means nothing on its own — spin *is* mixed there. What says
        whether it matters is :attr:`block_leak`.
    block_leak : ndarray (n_blocks,)
        How much of :attr:`block_s_squared` comes from the out-of-CAS-space terms. The part
        that would silently be missing if they had been assumed away.
    inactive_s : ndarray (3,)
        ``Tr_inactive(s_k)``, which is zero for a Kramers-paired inactive space. Included,
        not assumed.
    has_soc : bool
        Whether the Hamiltonian these states came from carries spin-orbit coupling. ⚠ It
        changes what the number *means*, not how it is computed: ``False`` makes ``2S+1`` a
        quantum number, ``True`` makes it a purity diagnostic.
    """

    blocks: Tuple[Tuple[int, int], ...]
    block_s_squared: np.ndarray
    state_s_squared: np.ndarray
    energies_cm: np.ndarray
    leakage: float = 0.0
    block_leak: np.ndarray = field(default_factory=lambda: np.zeros(0))
    inactive_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    has_soc: bool = True

    @property
    def block_spin(self) -> np.ndarray:
        """``S`` implied by each block's ``<S^2>``."""
        return np.array([spin_from_s_squared(v) for v in self.block_s_squared])

    @property
    def block_multiplicity(self) -> np.ndarray:
        """``2S+1`` per block. An integer only where spin is a good quantum number."""
        return 2.0 * self.block_spin + 1.0

    def purity(self) -> np.ndarray:
        """Distance of each block's ``2S+1`` from the nearest integer — 0 for a pure spin.

        With spin-orbit coupling **off** this is roundoff and nothing else, and a value that
        is *not* says something is wrong with the calculation rather than with the spin.

        ⚠ With it **on** the deviation has **two** sources and this number does not separate
        them: the states may genuinely be spin-mixed, and the *orbitals* may not be spin-pure
        (a converged general-complex CASSCF's are not — mixing spin is what spin-orbit
        coupling does). :attr:`block_leak` is what tells them apart, since the second source
        is exactly the out-of-active-space contribution. A TiCl₃ ``3d¹`` doublet — one
        electron, so no spin it *can* mix with — still reports 1.1e-03 here, all of it from
        the orbitals. Read the two lines together or neither.
        """
        m = self.block_multiplicity
        return np.abs(m - np.round(m))

    def report(self, logger=None) -> None:
        """The INFO table: one row per degenerate block, never per state."""
        logger = logger or log
        out.subsection(logger, "Spin expectation values")
        out.entries(logger, [
            ("blocks", len(self.blocks)),
            ("spin-orbit coupling", "on" if self.has_soc else "off"),
            ("inactive contribution to S", float(np.max(np.abs(self.inactive_s))), "hbar",
             "exactly zero for a Kramers-paired inactive set", "{:.2e}"),
            ("S leakage out of the active space", self.leakage, "hbar",
             "zero for a spin-separable orbital set; included either way", "{:.2e}"),
            ("its largest contribution to <S^2>",
             float(np.max(np.abs(self.block_leak))) if np.size(self.block_leak) else 0.0,
             "", "the part that would be missing if it were assumed away", "{:.2e}"),
        ])
        out.note(logger, "S is not conserved with spin-orbit coupling on: read 2S+1 as a "
                         "spin-purity measure, not a multiplicity"
                 if self.has_soc else
                 "spin-orbit coupling is off, so 2S+1 is the term multiplicity")
        columns = [out.col_count("block", 7), out.Column("states", "{:d}", 8),
                   out.Column("E [cm^-1]", out.CM_FMT, 14),
                   out.Column("<S^2>", "{:.4f}", 10),
                   out.Column("S", "{:.3f}", 8),
                   out.Column("2S+1", "{:.3f}", 9),
                   out.Column("|dev|", "{:.1e}", 9)]
        table = out.Table(logger, columns).start()
        spins, mult, dev = self.block_spin, self.block_multiplicity, self.purity()
        for b, (_, size) in enumerate(self.blocks):
            table.row(b + 1, size, float(self.energies_cm[b]),
                      float(self.block_s_squared[b]), float(spins[b]), float(mult[b]),
                      float(dev[b]))
        table.end()

    def __repr__(self) -> str:
        return "SpinAnalysis(blocks={}, <S^2>={})".format(
            len(self.blocks), np.round(self.block_s_squared, 4).tolist())


def spin_squared_states(solver, s_active: np.ndarray, *,
                        inactive_trace: Sequence[float] = (0.0, 0.0, 0.0),
                        leak_blocks: Optional[Sequence[np.ndarray]] = None,
                        vectors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """``<I|S^2|I>`` for every state — the arithmetic of the module docstring.

    Returns ``(s_squared, leak)``, both ``(n_states,)`` real: the total, and how much of it
    came from the out-of-CAS-space families. ⚠ The second is returned rather than folded
    away because it is the only thing that says whether assuming the spaces spin-separable
    would have mattered — a diagnostic whose value you cannot see is not one.

    Parameters
    ----------
    solver : an object with ``one_body_moments`` — :class:`kuiva.mcscf.casci.FullCISolver`
        or :class:`kuiva.dmrg.DMRGSolver`. ⚠ Duck-typed on purpose: ``props`` imports
        neither ``mcscf`` nor ``dmrg``. The CI route evaluates the square through the
        excitation map, the network route through each root's own 1- and 2-RDM; a solver
        providing neither refuses rather than approximating.
    s_active : ``(3, n_act, n_act)`` — the spin operator over the **active** spinors.
    inactive_trace : ``(3,)`` — ``Tr_inactive(s_k)``, from
        :func:`kuiva.props.dump.inactive_moment`.
    leak_blocks : optional sequence of three ``(3, ...)`` arrays ``(B, C, D)``, the
        active-inactive, virtual-active and virtual-inactive blocks of ``s_k``. Omitted means
        the spaces are exactly spin-separable and the correction is identically zero.
    vectors : the CI vectors; the solver's stored states by default.
    """
    if not hasattr(solver, "one_body_moments"):
        raise NotImplementedError(
            "<S^2> is evaluated through the solver's one_body_moments(), which {} does "
            "not provide. The conventional-CI solver applies S to the CI vectors through "
            "the determinant excitation map; the tensor-network solver contracts the "
            "same quantities through each root's own densities"
            .format(type(solver).__name__))
    s_act = np.ascontiguousarray(s_active, dtype=np.complex128)
    if s_act.ndim != 3 or s_act.shape[0] != 3:
        raise ValueError("s_active must be (3, n_act, n_act); got {}".format(s_act.shape))
    expect, square, rdm1 = solver.one_body_moments(s_act, vectors)
    t = np.asarray(inactive_trace, dtype=float).ravel()
    if t.size != 3:
        raise ValueError("inactive_trace must hold three components; got {}".format(t.size))

    # || (S_k^act + t_k) |I> ||^2, the CAS-space part. The cross term is real because S_k is
    # Hermitian and t_k is its inactive trace, hence real.
    s2 = np.sum(square + 2.0 * t[:, None] * expect + (t ** 2)[:, None], axis=0)

    leak = np.zeros_like(s2)
    if leak_blocks is not None:
        b_all, c_all, d_all = (np.asarray(x, dtype=np.complex128) for x in leak_blocks)
        # Constant in the state: the virtual-inactive family moves an electron between two
        # spaces the CAS state occupies identically in every determinant.
        leak += float(np.sum(np.abs(d_all) ** 2))
        for k in range(3):
            bb = b_all[k] @ b_all[k].conj().T                       # (n_act, n_act)
            cc = c_all[k].conj().T @ c_all[k]                       # (n_act, n_act)
            trace_bb = float(np.real(np.trace(bb)))
            for i in range(rdm1.shape[0]):
                gamma = rdm1[i]
                leak[i] += trace_bb - float(np.real(np.sum(gamma * bb)))
                leak[i] += float(np.real(np.sum(cc * gamma)))
    return np.real(s2 + leak), np.real(leak)


def spin_analysis(solver, coeff_ao: np.ndarray, spaces, s_ao: np.ndarray,
                  energies: Sequence[float], *, vectors: Optional[np.ndarray] = None,
                  tol_cm: float = 1.0, has_soc: bool = True,
                  inactive_tol: float = DEFAULT_INACTIVE_TOL) -> SpinAnalysis:
    """``<S^2>`` of a converged CI spectrum, blocked by degeneracy.

    Parameters
    ----------
    solver : the CI solver the states came from (see :func:`spin_squared_states`).
    coeff_ao : ``(2*nao, n_orb)`` complex — the **converged** spinors in the AO basis. ⚠ The
        same pairing rule the property dump states: these must be the orbitals the states
        were solved at.
    spaces : :class:`kuiva.mcscf.orbopt.OrbitalSpaces` — the active/inactive/virtual
        partition.
    s_ao : ``(nao, nao)`` — the scalar AO overlap, which is the metric the spin operator
        needs (it acts on the spin index only).
    energies : ``(n_states,)`` total state energies [Eh], for the blocking.
    tol_cm : degeneracy tolerance, as :func:`kuiva.props.multiplet.degenerate_blocks`.
    has_soc : whether the Hamiltonian carried spin-orbit coupling. Reporting only.
    """
    from ..spinor.expand import spin_operator
    from .dump import inactive_moment

    s_mo = spinor_operator(coeff_ao, spin_operator(s_ao))
    act = np.asarray(spaces.active, dtype=int)
    inact = np.asarray(spaces.inactive, dtype=int)
    virt = np.asarray(spaces.virtual, dtype=int)

    inact_s = inactive_moment(s_mo, inact, name="S", tol=inactive_tol)
    s_act = np.stack([sk[np.ix_(act, act)] for sk in s_mo])
    b = np.stack([sk[np.ix_(act, inact)] for sk in s_mo])
    c = np.stack([sk[np.ix_(virt, act)] for sk in s_mo])
    d = np.stack([sk[np.ix_(virt, inact)] for sk in s_mo])
    leakage = max(float(np.max(np.abs(x))) if x.size else 0.0 for x in (b, c, d))

    s2_states, leak_states = spin_squared_states(
        solver, s_act, inactive_trace=inact_s, leak_blocks=(b, c, d), vectors=vectors)
    worst_leak = float(np.max(np.abs(leak_states))) if leak_states.size else 0.0
    if worst_leak > SPIN_LEAKAGE_WARN:
        log.warning(
            "the spin operator connects the active space to the inactive/virtual one by up "
            "to %.3e hbar, and that moves <S^2> by as much as %.3e: S|I> leaves the CAS "
            "space and does so by an amount large enough to matter. The contribution is "
            "computed and INCLUDED, so the numbers below are right; what it says is that "
            "these orbital spaces are far from spin-separable, and a spin label attached to "
            "such a state means correspondingly less", leakage, worst_leak)
    e = np.asarray(energies, dtype=float).ravel()
    if e.size != s2_states.size:
        raise ValueError("{} energies for {} states".format(e.size, s2_states.size))
    order = np.argsort(e)
    e_cm = (e[order] - e[order][0]) * HARTREE_TO_CM
    s2_sorted, leak_sorted = s2_states[order], leak_states[order]

    blocks = degenerate_blocks(e_cm, tol_cm=tol_cm)
    block_s2 = np.array([float(np.mean(s2_sorted[s:s + n])) for s, n in blocks])
    block_leak = np.array([float(np.mean(leak_sorted[s:s + n])) for s, n in blocks])
    block_e = np.array([float(np.mean(e_cm[s:s + n])) for s, n in blocks])
    log.debug("<S^2>: %d states -> %d blocks, block values %s",
              e.size, len(blocks), np.round(block_s2, 4).tolist())
    return SpinAnalysis(blocks=tuple(blocks), block_s_squared=block_s2,
                        state_s_squared=s2_sorted, energies_cm=block_e,
                        leakage=leakage, block_leak=block_leak,
                        inactive_s=inact_s, has_soc=bool(has_soc))


__all__ = ["SPIN_LEAKAGE_WARN", "SpinAnalysis", "spin_analysis", "spin_from_s_squared",
           "spin_squared_states"]
