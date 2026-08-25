"""The property-matrix dump: Kuiva's actual product.

Everything upstream of this file exists to produce matrices in the basis of the spin-orbit
eigenstates — the effective Hamiltonian ``H``, the three magnetic-moment components ``mu_x``,
``mu_y``, ``mu_z``, and the three electric-dipole components ``d_x``, ``d_y``, ``d_z`` — which
an **external** ITO / Stevens / crystal-field code turns into the quantities an experiment
measures. The project scope puts that analysis explicitly out of
scope, so this file is the boundary: a plain-text, versioned, self-describing file, entirely
separate from the log stream (logging never contaminates machine-readable output).

⚠ **That boundary is where the electric dipole stops too.** This module writes the operator
and its phase-invariant reductions (``Tr_block(d_i d_j)`` and the block-to-block line strength
``sum |d_IJ|^2``); it computes no oscillator strength, no Einstein coefficient and no radiative
rate. Those need a transition energy, a refractive index and a convention for what is being
measured, and they belong to the same external code the crystal-field analysis does.

The physics in one line
-----------------------
``mu = -(L + g_e S) mu_B``, and in the many-electron state basis

    mu^{IJ}_k = -( sum_{tu} (L_k + g_e S_k)_{tu} gamma^{IJ}_{tu}
                   + delta_IJ sum_{i in inactive} (L_k + g_e S_k)_{ii} )   [mu_B]

with ``gamma^{IJ}_{tu} = <I|E_tu|J>`` the transition density matrices — the third
consumer of the *same* excitation map and the *same* intermediate the sigma vector and the
RDMs are built from. ``L`` and ``S`` are one-electron operators, so nothing beyond the
one-particle transition densities is ever needed, however large the CI.

The electric dipole rides on exactly the same contraction, with two differences that are
easy to get wrong and impossible to notice afterwards:

    d^{IJ}_k = -( sum_{tu} (r - R_G)_{k,tu} gamma^{IJ}_{tu}
                  + delta_IJ sum_{i in inactive} (r - R_G)_{k,ii} )
               + delta_IJ sum_A Z_A (R_A - R_G)_k                          [e a_0]

⚠ **The nuclear term is on the diagonal only** — off the diagonal it would invent a transition
dipole proportional to the nuclear charge — and ⚠ **the inactive term is not zero here**: ``r``
is time *even*, so a Kramers pair contributes twice its expectation value rather than
cancelling as it does for ``L`` and ``S``. A diagonal element of ``d`` is therefore the state's
dipole moment and an off-diagonal one is a transition dipole. ⚠ For a **charged** molecule the
diagonal obeys ``d(R_G) = d(0) - q R_G`` and moves with the gauge origin; transition elements
between distinct states do not, whatever the charge. :func:`write_dump` warns, and the header
carries the charge, the origin and the nuclear vector.

⚠ **The inactive term is computed for every operator, not assumed away for any of them** —
and what the *theorem* says about it depends on the operator's behaviour under time reversal.
A Kramers pair ``(psi, T psi)`` contributes ``<psi|A|psi> + <T psi|A|T psi> = 0`` for any
time-**odd** ``A``, so a Kramers-paired inactive set contributes exactly nothing to ``L`` or
``S``. That is a theorem about the *orbitals*, and a CASSCF that has broken Kramers symmetry
in the inactive space violates it: :func:`inactive_moment` measures it and warns above a
tolerance rather than skipping the term, because the failure it guards against is a moment
matrix silently missing a core contribution, which looks entirely plausible. For the
time-**even** ``r`` the same sum is a real and generally large number, so the warning is
switched off there (``expect_zero=False``) while the term itself is computed and used exactly
as the others are.

Four things about this file that are decisions, not details
-----------------------------------------------------------
1. ⚠ **``H`` is diagonal**, unlike OpenMolcas RASSI's. Kuiva's CI is already two-component,
   so its roots *are* the spin-orbit eigenstates: there is no separate spin-orbit mixing step
   to leave off-diagonal elements behind. A reader coming from a two-step (scalar CASSCF +
   RASSI) workflow will expect otherwise, so the header says it.
2. ⚠ **No picture change is applied to the property operators by default** (an explicit
   standing decision), and ⚠ **when it is applied, one flag applies it to all of them**: a
   file whose ``mu`` carried the correction and whose ``d`` did not would be a hybrid with
   nothing in it saying which half was which. The electric operator is *even*, so its
   four-component matrix has a small-component block where the magnetic one has none, but it
   goes through the same ``X`` and ``R``
   (:func:`kuiva.interface.pyscf_bridge.picture_changed_dipole`) and it is recorded in its own
   header field, ``picture_change_on_dipole``.
   ``L`` and ``S`` are the bare non-relativistic AO operators used
   unchanged in the two-component basis. This matches RASSI, which is what makes the
   cross-code comparison like-for-like. :func:`write_dump` emits a ``WARNING`` at the point of
   writing and records the treatment in the header, in the same way the mean field records its
   own standing obligations.

   **The size of the approximation has been measured** and the correction is available as a
   non-default option (``property_picture_change=True`` on the front end, which transforms the
   four-component moment operator with the same ``X`` and ``R`` that decouple the Hamiltonian;
   Peng & Reiher 2012). On free-ion multiplets it moves a g factor by **1e-04 relative at
   Z = 5, rising smoothly to 1e-03 at Z = 81**, reaching 2.6e-03 on Yb(3+) ``2F5/2``; on a 3d
   complex's ground doublet by 1.9e-04; and it splits **no** degeneracy anywhere (exactly
   zero). ⚠ **Two things that bound reads of those numbers.** They *grow with Z*, so they are a
   bound for the elements measured and not for a heavier one. And on a level whose ``g``
   approaches zero the *relative* shift is inflated by its own denominator — the largest seen,
   8e-03, is such a case, whose absolute shift is only ~5x the median — so quote an absolute
   shift alongside a relative one whenever ``g`` is small.

   ⚠ **Turning the correction on changes what a stored ``mu`` means**, while ``format_version``
   stays the same: the header field ``picture_change_on_properties`` is what distinguishes the
   two, and a reader that ignores the header will misread the moment matrices. That field is
   the reason the format version does not move — it exists precisely to carry this — but it
   makes reading the header obligatory rather than optional. ⚠ **The picture-changed moment
   does not separate into an ``L`` part and an ``S`` part**, so in such a file the ``L`` and
   ``S`` blocks are the bare operators that ``mu`` was *not* built from, and are written for
   reference only.
3. ⚠ **Phases are arbitrary and are not canonicalized**. Within a degenerate block the
   eigenvectors are defined only up to a unitary mixing, so an element-by-element comparison
   of these matrices — against another program, or against another run of this one — is
   meaningless. Compare through :mod:`kuiva.props.multiplet`'s invariants: degeneracy
   patterns, relative energies, and ``M_ij = Tr_block(mu_i mu_j)`` with its principal g
   values. :meth:`PropertyMatrices.analyse` is that reduction, one call away.
4. **The header carries** :meth:`kuiva.interface.pyscf_bridge.SpinOrbitX2C.provenance` — both
   the ``ScreeningRecord`` and the ``DecouplingRecord`` — plus the gauge origin
   and the active space. A stored property matrix that does not say which Hamiltonian
   produced it is not interpretable, and the difference between a screened and an unscreened
   one is 5-30% on every splitting in it.

The format
----------
Line oriented, ``#`` comments, ``[SECTION]`` markers, one ``i j Re Im`` record per matrix
element. It is deliberately dull: the format is a contract with an external code and
is easy to change later, so it optimizes for being trivially parseable in any language rather
than for compactness. :func:`read_dump` is a working parser and the round-trip test — a
format nobody has ever read back is a format with an undetected ambiguity in it.

**Portability:** this module is **orchestration** — formatting and one contraction over
matrices of state dimension. Nothing here is a kernel and nothing here should ever be ported.

References
----------
* Magnetic moments and pseudospin g tensors from ab initio spin-orbit states: L. F. Chibotaru,
  L. Ungur, J. Chem. Phys. 137, 064112 (2012), doi:10.1063/1.4739763.
* Transition density matrices between CI states: J. Olsen, B. O. Roos, P. Jorgensen,
  H. J. Aa. Jensen, J. Chem. Phys. 89, 2185 (1988), doi:10.1063/1.455063.
* Picture change of property operators under X2C (what item 2 above omits): D. Peng,
  M. Reiher, J. Chem. Phys. 136, 244108 (2012), doi:10.1063/1.4729788.
* The free-electron g factor: CODATA 2018, doi:10.1103/RevModPhys.93.025010.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util.logging import get_logger
from .multiplet import (G_ELECTRON, HARTREE_TO_CM, Multiplet, analyse_spectrum,
                        block_line_strengths, spectrum_line_strengths)

log = get_logger(__name__)

#: Bumped whenever the *meaning* of anything already in the file changes. Adding a section or
#: a header key does not require a bump; renaming one, or changing a unit or a sign
#: convention, does. A consumer that does not recognise the version must refuse the file.
FORMAT_VERSION = 1

#: Tolerance [hbar] on the inactive contribution to ``L`` and ``S``, which is exactly zero
#: for a Kramers-paired inactive set. Sized well above the 1e-13-ish rounding of a congruence
#: transformation on a few hundred functions and far below any physically meaningful moment.
DEFAULT_INACTIVE_TOL = 1e-8


def _kuiva_version() -> str:
    """The running code's version, for the file header."""
    from .. import __version__
    return str(__version__)


# --- operators in the spinor MO basis ------------------------------------------------------

def spinor_operators(coeff_ao: np.ndarray, l_ao_2c: np.ndarray,
                     s_ao_2c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(L, S)`` in the spinor MO basis: two congruences, ``C^dag A C``.

    Parameters
    ----------
    coeff_ao : ``(2*nao, n_orb)`` complex — the spinors in the AO basis.
    l_ao_2c, s_ao_2c : ``(3, 2*nao, 2*nao)`` — the two-component AO operators, from
        :meth:`kuiva.interface.pyscf_bridge.PropertyIntegrals.two_component` and
        :func:`kuiva.spinor.expand.spin_operator`.

    Returns ``(3, n_orb, n_orb)`` pairs in units of hbar.
    """
    c = np.ascontiguousarray(coeff_ao, dtype=np.complex128)
    l = np.asarray(l_ao_2c)
    s = np.asarray(s_ao_2c)
    if l.shape != s.shape or l.ndim != 3 or l.shape[0] != 3:
        raise ValueError("L and S must both be (3, 2*nao, 2*nao); got {} and {}"
                         .format(l.shape, s.shape))
    if l.shape[1] != c.shape[0]:
        raise ValueError("the operators span {} two-component AO rows and the spinors {}"
                         .format(l.shape[1], c.shape[0]))
    return spinor_operator(c, l), spinor_operator(c, s)


def spinor_operator(coeff_ao: np.ndarray, op_ao_2c: np.ndarray) -> np.ndarray:
    """One two-component AO operator in the spinor MO basis: ``C^dag A C`` per component.

    ``(3, 2*nao, 2*nao)`` in, ``(3, n_orb, n_orb)`` out. :func:`spinor_operators` is the
    two-operator case of this; the picture-changed moment operator
    (:meth:`kuiva.interface.pyscf_bridge.PropertyIntegrals.moment_operator`) is a third
    operator that goes through the same congruence and must not grow its own.
    """
    c = np.ascontiguousarray(coeff_ao, dtype=np.complex128)
    a = np.asarray(op_ao_2c)
    if a.ndim != 3 or a.shape[0] != 3:
        raise ValueError("the operator must be (3, 2*nao, 2*nao), got {}".format(a.shape))
    if a.shape[1] != c.shape[0]:
        raise ValueError("the operator spans {} two-component AO rows and the spinors {}"
                         .format(a.shape[1], c.shape[0]))
    ct = c.conj().T
    return np.stack([ct @ ak @ c for ak in a])


def inactive_moment(op_mo: np.ndarray, inactive: Sequence[int], *,
                    name: str = "operator", tol: float = DEFAULT_INACTIVE_TOL,
                    expect_zero: bool = True) -> np.ndarray:
    """``sum_{i in inactive} A_ii`` for each component — computed, checked, never assumed.

    Returns the ``(3,)`` real trace. For a **time-odd** operator it must vanish for a
    Kramers-paired inactive set, because a Kramers pair contributes equal and opposite
    expectation values; a nonzero result above ``tol`` means the inactive spinors are no
    longer Kramers paired, which is a statement about the orbitals and is worth a warning.
    Nonzero or not, the value is *used*, so a broken inactive space degrades the moments
    rather than silently dropping a term.

    ⚠ ``expect_zero=False`` for a **time-even** operator, and the electric dipole is the case
    it exists for. ``r`` is time even, so a Kramers pair contributes ``2 <psi|r|psi>`` and the
    inactive electrons carry a perfectly real share of the molecule's dipole: warning about it
    would be warning that the core exists. The term is still computed, still used and still
    reported — what changes is only whether a nonzero value is an anomaly.
    """
    idx = np.asarray(inactive, dtype=int).ravel()
    op = np.asarray(op_mo)
    if idx.size == 0:
        return np.zeros(3)
    trace = np.array([np.real(np.trace(opk[np.ix_(idx, idx)])) for opk in op])
    worst = float(np.max(np.abs(trace)))
    if expect_zero and worst > tol:
        log.warning("the inactive space contributes %.3e hbar to <%s>, which must be exactly "
                    "zero for a Kramers-paired inactive set (L and S are both time odd). The "
                    "inactive spinors are evidently no longer Kramers paired; the "
                    "contribution is included in the moment matrices as computed, but the "
                    "orbitals are worth inspecting", worst, name)
    return trace


def state_operator_matrices(op_active: np.ndarray, tdm: np.ndarray,
                            inactive_trace: Optional[np.ndarray] = None) -> np.ndarray:
    """Lift a one-electron operator into the state basis.

    ``A^{IJ}_k = sum_{tu} A_{k,tu} gamma^{IJ}_{tu} + delta_IJ * inactive_trace_k``.

    Parameters
    ----------
    op_active : ``(3, n_act, n_act)`` — the operator over the **active** spinors.
    tdm : ``(n_states, n_states, n_act, n_act)`` — ``gamma^{IJ}_{tu} = <I|E_tu|J>``, from
        :meth:`kuiva.mcscf.casci.FullCISolver.transition_densities`.
    inactive_trace : ``(3,)``, optional — from :func:`inactive_moment`.

    Returns ``(3, n_states, n_states)``, complex and Hermitian.
    """
    op = np.asarray(op_active)
    g = np.asarray(tdm)
    if op.ndim != 3 or op.shape[0] != 3:
        raise ValueError("the operator must be (3, n_act, n_act), got {}".format(op.shape))
    if g.ndim != 4 or g.shape[2:] != op.shape[1:]:
        raise ValueError("the transition densities must be (ns, ns, {0}, {0}), got {1}"
                         .format(op.shape[1], g.shape))
    # (3, na, na) x (ns, ns, na, na) -> (3, ns, ns). Written as one GEMM on the flattened
    # orbital pair index rather than an einsum (einsum does not dispatch to BLAS on a hot path) -- though at state dimensions this
    # is never hot, so the reason here is only that it is also the clearer expression.
    na = op.shape[1]
    ns = g.shape[0]
    mat = (op.reshape(3, na * na) @ g.reshape(ns * ns, na * na).T).reshape(3, ns, ns)
    if inactive_trace is not None:
        mat = mat + np.asarray(inactive_trace, dtype=float)[:, None, None] * np.eye(ns)
    return mat


# --- the dump ------------------------------------------------------------------------------

@dataclass(frozen=True)
class PropertyMatrices:
    """``H`` and ``mu`` in the spin-orbit eigenstate basis — what the dump exists to produce.

    Attributes
    ----------
    energies : ``(n_states,)`` — total state energies [Eh], as :attr:`H`'s diagonal.
    mu : ``(3, n_states, n_states)`` complex — magnetic moments [mu_B], ``-(L + g_e S)``.
    l, s : ``(3, n_states, n_states)`` complex — the two halves separately [hbar]. Not part
        of the external contract, but written to the file because they cost nothing and they
        are what a disagreement is bisected with.
    inactive_l, inactive_s : ``(3,)`` — the measured inactive contributions, which are zero
        for a Kramers-paired inactive space (see :func:`inactive_moment`). Reported so that
        "it was checked" is visible in the file rather than only in the code.
    picture_change : str — what was done to the property operators. Empty or a string starting
        ``"none"`` means the bare non-relativistic ``L`` and ``S`` were used unchanged in the
        two-component basis, which is the default. ⚠ Anything else means ``mu`` was **not**
        built from :attr:`l` and :attr:`s`, and the two files are not comparable element for
        element.
    d : ``(3, n_states, n_states)`` complex or ``None`` — the **total** electric dipole
        [e a_0]: electronic plus, on the diagonal, the nuclear term. So ``d[k].diagonal()`` is
        each state's dipole moment and ``d[k][I,J]`` is a transition dipole. ``None`` where the
        reference carried no dipole integrals.
    nuclear_dipole : ``(3,)`` — the nuclear half, ``sum_A Z_A (R_A - R_G)``, kept separately so
        the two parts of the diagonal can still be told apart after the fact.
    inactive_d : ``(3,)`` — the inactive electrons' share. ⚠ Unlike :attr:`inactive_l` this is
        **not** zero and must not be: ``r`` is time *even*, so a Kramers pair contributes twice
        its expectation value rather than cancelling.
    molecular_charge : int — ⚠ nonzero means every diagonal element of :attr:`d`, and every
        block-internal invariant built from it, **moves with the gauge origin** as
        ``d(R_G) = d(0) - q R_G``. Transition elements between distinct states do not.
    dipole_picture_change : str — the electric operator's counterpart of
        :attr:`picture_change`, and it moves with it: one flag governs both, so these two are
        either both ``"none"`` or both a Peng-Reiher record. It is a separate field because a
        consumer reading only the header must be able to see what ``d`` means without parsing
        the provenance JSON.
    """

    energies: np.ndarray
    mu: np.ndarray
    l: np.ndarray
    s: np.ndarray
    gauge_origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_label: str = "unspecified"
    g_electron: float = G_ELECTRON
    active_space: str = ""
    provenance: Dict[str, object] = field(default_factory=dict)
    inactive_l: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inactive_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    comments: Tuple[str, ...] = ()
    picture_change: str = ""
    #: ⚠ Defaulted to ``None`` rather than to zeros: "no dipole was computed" and "the dipole
    #: is zero" are different statements, and a symmetric molecule makes the second one true.
    d: Optional[np.ndarray] = None
    nuclear_dipole: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inactive_d: np.ndarray = field(default_factory=lambda: np.zeros(3))
    molecular_charge: int = 0
    dipole_picture_change: str = ""

    @property
    def picture_changed(self) -> bool:
        """True when ``mu`` carries the X2C picture change on the property operators."""
        return bool(self.picture_change) and not self.picture_change.startswith("none")

    @property
    def has_dipole(self) -> bool:
        """Whether electric dipole matrices are present."""
        return self.d is not None

    @property
    def dipole_is_origin_dependent(self) -> bool:
        """True for a charged molecule — see :attr:`molecular_charge`."""
        return int(self.molecular_charge) != 0

    @property
    def n_states(self) -> int:
        return int(np.size(self.energies))

    @property
    def hamiltonian(self) -> np.ndarray:
        """``H`` as a full ``(n, n)`` matrix. ⚠ **Diagonal** — see the module docstring."""
        return np.diag(np.asarray(self.energies, dtype=np.complex128))

    def relative_energies_cm(self) -> np.ndarray:
        e = np.asarray(self.energies, dtype=float)
        return (e - e.min()) * HARTREE_TO_CM

    def analyse(self, tol_cm: float = 1.0,
                pseudo_doublet_tol_cm: Optional[float] = None) -> List[Multiplet]:
        """The phase-invariant reduction — the **only** sound way to compare these.

        Degeneracy pattern, relative energies, and the invariant ``M_ij = Tr_block(mu_i mu_j)``
        with its principal g values. Any validation of this file's contents must go through
        here; element-by-element comparison of :attr:`mu` compares arbitrary phases.

        ``pseudo_doublet_tol_cm`` groups tunnelling-split singlets into **non-Kramers
        pseudo-doublets** — the integer-spin (Tb/Ho-type) case, where the moment belongs to
        the pair and each singlet alone has none. Opt-in and never inferred; see
        :func:`kuiva.props.multiplet.analyse_spectrum`.
        """
        return analyse_spectrum(self.energies, self.mu, tol_cm=tol_cm,
                                pseudo_doublet_tol_cm=pseudo_doublet_tol_cm, d=self.d)

    def line_strengths(self, tol_cm: float = 1.0,
                       multiplets: Optional[List[Multiplet]] = None) -> np.ndarray:
        """``S_AB = sum_k sum_{I in A, J in B} |d_k[I,J]|^2`` [(e a_0)^2] over the multiplets.

        The phase-invariant statement about a **transition**, and the only sound way to compare
        this file's ``d`` between two runs or two codes: a double sum of squared moduli over two
        whole degenerate blocks survives the arbitrary mixing inside each of them, which an
        element-by-element comparison does not.

        ⚠ **A line strength is not an oscillator strength and not a rate.** Kuiva writes
        operators and their invariants; turning one into an intensity needs the transition
        energy, a refractive index and a convention for what is being measured, and that
        analysis lives in the external property code, exactly as the ITO and crystal-field
        analysis does.
        """
        if self.d is None:
            raise ValueError(
                "these property matrices carry no electric dipole, so there are no line "
                "strengths to compute. The reference was built without dipole integrals")
        blocks = self.analyse(tol_cm=tol_cm) if multiplets is None else multiplets
        return spectrum_line_strengths(self.energies, self.d, blocks)

    def hermiticity_error(self) -> float:
        """``max |A - A^dag|`` over the three moment components — a structural self-check."""
        mu = np.asarray(self.mu)
        return float(np.max(np.abs(mu - mu.conj().transpose(0, 2, 1)))) if mu.size else 0.0

    def report(self, logger=None, pseudo_doublet_tol_cm: Optional[float] = None) -> None:
        """The INFO summary: the multiplet table, never the matrices themselves.

        ⚠ A block with no g values prints ``nan``, not ``0``: a ``size = 1`` block carries no
        magnetic moment, and "not defined here" is a different statement from "measured zero".
        With ``pseudo_doublet_tol_cm`` the table gains a ``Delta`` column and non-Kramers
        pairs are marked — read ``g_3`` as ``g_z`` there, and ``g_1``/``g_2`` as the residual
        that should be zero by symmetry.
        """
        logger = logger or log
        out.subsection(logger, "Spin-orbit multiplets and magnetic moments")
        out.entries(logger, [
            ("states", self.n_states),
            ("gauge origin", "({:.4f}, {:.4f}, {:.4f}) bohr".format(
                *np.asarray(self.gauge_origin).ravel()), "", self.origin_label),
            ("free-electron g factor", self.g_electron, "", "", "{:.8f}"),
            ("inactive contribution to L", float(np.max(np.abs(self.inactive_l))), "hbar",
             "exactly zero for a Kramers-paired inactive set", "{:.2e}"),
            ("moment matrix hermiticity", self.hermiticity_error(), "mu_B", "", "{:.2e}"),
        ])
        multiplets = self.analyse(pseudo_doublet_tol_cm=pseudo_doublet_tol_cm)
        paired = pseudo_doublet_tol_cm is not None
        columns = [out.col_count("block", 7), out.Column("states", "{:d}", 8),
                   out.Column("E [cm^-1]", out.CM_FMT, 14),
                   out.Column("spread", "{:.3e}", 11),
                   out.Column("g_1", "{:.4f}", 9), out.Column("g_2", "{:.4f}", 9),
                   out.Column("g_3", "{:.4f}", 9)]
        if paired:
            columns.append(out.Column("Delta", "{:.3e}", 11))
        table = out.Table(logger, columns)
        table.start()
        for i, m in enumerate(multiplets):
            g = m.g_values if m.g_values else (float("nan"),) * 3
            row = [i, m.size, m.energy_cm, m.spread_cm, g[0], g[1], g[2]]
            if paired:
                row.append(float("nan") if m.tunnelling_gap_cm is None
                           else m.tunnelling_gap_cm)
            table.row(*row)
        # ⚠ Each clause appears only when the table it describes contains the thing it
        # describes. A note about `nan` under a table with no `nan` in it is noise, and it
        # would also move every committed reference that has none.
        note = "g values are principal values of M_ij = Tr_block(mu_i mu_j); phases are arbitrary"
        if any(not m.g_values for m in multiplets):
            note += "; nan = the block carries no moment (a single state has none)"
        if paired:
            note += ("; Delta marks a non-Kramers pseudo-doublet, where g_3 is g_z and "
                     "g_1/g_2 are a residual that is zero by symmetry")
        table.end(note + " ")
        self._report_axes(logger, multiplets)
        self._report_dipole(logger, multiplets)

    @staticmethod
    def _report_axes(logger, multiplets: List[Multiplet]) -> None:
        """The principal magnetic axes, beside the g values they belong to.

        ⚠ A separate table rather than three more columns on the one above, for two reasons:
        nine numbers per block would not fit the standard width, and the g-value table is what
        a cross-code comparison is read from — it stays exactly as wide as it was.

        Each row states whether its easy axis is a **direction** at all. An easy-plane or
        isotropic block has none, and the vector printed for one is whatever the eigensolver
        chose inside the degenerate plane; saying so on the line is the difference between a
        result and an artefact.
        """
        rows = [m for m in multiplets if m.g_axes is not None and len(m.g_values) == 3]
        if not rows:
            return
        table = out.Table(logger, [
            out.col_count("block", 7), out.Column("g_max", "{:.4f}", 9),
            # ⚠ `{:.3g}`, not a fixed number of decimals: a strongly axial doublet reaches
            # an axiality of 1e6 and up (measured on TiCl3), where two decimal places are
            # both unreadable and meaningless.
            out.Column("axiality", "{:.3g}", 10),
            out.Column("e_x", "{:+.4f}", 9), out.Column("e_y", "{:+.4f}", 9),
            out.Column("e_z", "{:+.4f}", 9),
            out.Column("axis", "{}", 12, align="<"),
            out.Column("det g", "{}", 6)])
        table.start()
        for i, m in enumerate(multiplets):
            if m.g_axes is None or len(m.g_values) != 3:
                continue
            axis = np.asarray(m.g_axes)[:, -1]
            defined = m.easy_axis_is_defined()
            table.row(i, max(m.g_values), m.axiality, axis[0], axis[1], axis[2],
                      "easy" if defined else "in a plane",
                      "?" if m.g_sign == 0.0 else ("+" if m.g_sign > 0 else "-"))
        table.end("the easy axis is the principal direction of the largest g; 'in a plane' "
                  "means the top two g values coincide and the direction printed is an "
                  "arbitrary choice within that plane. det g is the sign M cannot carry, "
                  "'?' where it is not defined -- a block that is not a Kramers doublet, or "
                  "one so axial that det g is numerically zero and its sign would be "
                  "rounding ")

    def _report_dipole(self, logger, multiplets: List[Multiplet]) -> None:
        """The electric dipole, through invariants only — never element by element.

        Two invariant quantities per block and nothing else: the block-averaged permanent
        moment ``Tr_b(d_k) / size``, which survives any mixing inside the block, and the line
        strength to the **ground** block, which survives mixing inside either. A single
        ``d[I,J]`` printed here would be a phase, and the phases in this object are arbitrary.

        ⚠ The charge line is the one that has to be read: for a charged molecule the permanent
        column moves with the gauge origin and the strength column does not.
        """
        if self.d is None:
            return
        order = np.argsort(np.asarray(self.energies, dtype=float))
        d_sorted = np.asarray(self.d)[:, order, :][:, :, order]
        strengths = block_line_strengths(d_sorted, [(m.start, m.size) for m in multiplets])
        nuc = np.asarray(self.nuclear_dipole, dtype=float).ravel()
        inact = np.asarray(self.inactive_d, dtype=float).ravel()
        charged = self.dipole_is_origin_dependent
        out.subsection(logger, "Electric dipole")
        out.entries(logger, [
            ("molecular charge", int(self.molecular_charge), "e",
             "the permanent moments below MOVE with the gauge origin" if charged
             else "neutral, so every dipole below is origin independent"),
            ("nuclear contribution", "({:+.6f}, {:+.6f}, {:+.6f})".format(*nuc), "e*a0",
             "diagonal only; a transition dipole gets nothing from it"),
            ("inactive contribution", "({:+.6f}, {:+.6f}, {:+.6f})".format(*inact), "e*a0",
             "generally nonzero: r is time EVEN, unlike L and S"),
        ])
        table = out.Table(logger, [
            out.col_count("block", 7),
            out.Column("d_x", "{:+.5f}", 11), out.Column("d_y", "{:+.5f}", 11),
            out.Column("d_z", "{:+.5f}", 11),
            out.Column("|d|", "{:.5f}", 11),
            out.Column("S to block 0", out.SCI_FMT, 15)])
        table.start()
        for i, m in enumerate(multiplets):
            sl = slice(m.start, m.start + m.size)
            mean = np.array([float(np.real(np.trace(dk[sl, sl]))) / m.size
                             for dk in d_sorted])
            table.row(i, mean[0], mean[1], mean[2], float(np.linalg.norm(mean)),
                      float(strengths[i, 0]))
        table.end("d is the block-averaged permanent moment Tr_b(d_k)/size [e*a0], electronic "
                  "plus nuclear; S is the line strength sum |d_IJ|^2 to the ground block "
                  "[(e*a0)^2]. Both are phase invariant; individual matrix elements are not. "
                  "S is a line strength and NOT an oscillator strength or a rate ")

    def write(self, path, **kwargs) -> Path:
        """Write the property dump file. See :func:`write_dump`."""
        return write_dump(path, self, **kwargs)

    @classmethod
    def from_dump(cls, path) -> "PropertyMatrices":
        """Rebuild these matrices from a file :func:`write_dump` wrote.

        The counterpart of :meth:`write`, and the reason it exists is comparison: the phases
        in this file are arbitrary, so the *only* sound way to compare two stored runs — or a
        stored run against a fresh one — is through :meth:`analyse`, and until this existed
        that meant re-implementing the reduction against :func:`read_dump`'s dictionary at
        every call site that wanted it. Now::

            a = PropertyMatrices.from_dump("before.props").analyse()
            b = PropertyMatrices.from_dump("after.props").analyse()

        ⚠ **What comes back is the file, not the calculation.** The header's provenance,
        gauge origin, active space and picture-change record are restored because they are
        what makes the numbers interpretable; nothing else about the run is here, and this
        object cannot be handed back to a stage. :func:`read_dump` remains the raw form for
        anyone who wants the header verbatim.

        ⚠ ``L`` and ``S`` are written by ``write_dump(include_l_s=True)``, the default, but
        are **not** part of the external contract. A file written without them comes back with
        zero-filled ``l`` and ``s``: ``mu`` is the quantity, and it is always present.

        ⚠ The electric dipole comes back as ``None`` when the file has none — **not** as
        zeros. A molecule whose symmetry forbids a dipole has a genuinely zero ``d``, so
        zero-filling would make "this file does not carry a dipole" and "this molecule has no
        dipole" the same object, and :meth:`line_strengths` would then answer a question the
        file never asked.
        """
        raw = read_dump(path)
        header, matrices = raw["header"], raw["matrices"]
        energies = np.asarray(raw["energies"], dtype=float)
        n = int(energies.size)

        def stack(prefix: str) -> np.ndarray:
            found = [matrices.get("{}_{}".format(prefix, a)) for a in "xyz"]
            if any(m is None for m in found):
                return np.zeros((3, n, n), dtype=np.complex128)
            return np.ascontiguousarray(np.stack(found))

        mu = stack("mu")
        if not mu.any() and n:
            raise ValueError(
                "{}: no mu_x/mu_y/mu_z matrices in this file, so it is not a property dump "
                "this class can be rebuilt from".format(path))
        origin = np.asarray([float(x) for x in
                             header.get("gauge_origin_bohr", "0 0 0").split()], dtype=float)
        inactive = raw.get("inactive", {})
        picture = header.get("picture_change_on_properties", "")
        # ⚠ `None`, not zeros, when the file has no dipole: a symmetric molecule's dipole IS
        # zero, so zeros would make "not written" indistinguishable from "measured zero".
        d_found = [matrices.get("d_" + a) for a in "xyz"]
        d = (None if any(m is None for m in d_found)
             else np.ascontiguousarray(np.stack(d_found)))
        nuclear = np.asarray([float(x) for x in
                              header.get("nuclear_dipole_ea0", "0 0 0").split()], dtype=float)
        return cls(
            energies=energies, mu=mu, l=stack("L"), s=stack("S"),
            d=d, nuclear_dipole=nuclear if nuclear.size == 3 else np.zeros(3),
            inactive_d=np.asarray(inactive.get("d", np.zeros(3)), dtype=float),
            molecular_charge=int(header.get("molecular_charge", 0)),
            dipole_picture_change=(
                "" if header.get("picture_change_on_dipole", "none") == "none"
                else header["picture_change_on_dipole"]),
            gauge_origin=origin if origin.size == 3 else np.zeros(3),
            origin_label=header.get("gauge_origin_choice", "unspecified"),
            g_electron=float(header.get("g_electron", G_ELECTRON)),
            active_space=header.get("active_space", ""),
            provenance=dict(raw.get("provenance") or {}),
            inactive_l=np.asarray(inactive.get("L", np.zeros(3)), dtype=float),
            inactive_s=np.asarray(inactive.get("S", np.zeros(3)), dtype=float),
            picture_change="" if picture == "none" else picture)

    def __repr__(self) -> str:
        return "PropertyMatrices({} states, gauge origin {}, |dE| = {:.1f} cm^-1)".format(
            self.n_states, self.origin_label, float(self.relative_energies_cm().max()))


def property_matrices(coeff_ao: np.ndarray, spaces, tdm: np.ndarray, energies,
                      properties, s_ao: np.ndarray, *, g_electron: float = G_ELECTRON,
                      provenance: Optional[Dict[str, object]] = None,
                      active_space: str = "", comments: Sequence[str] = (),
                      inactive_tol: float = DEFAULT_INACTIVE_TOL) -> PropertyMatrices:
    """Assemble ``H`` and ``mu`` in the SOC eigenstate basis.

    Parameters
    ----------
    coeff_ao : ``(2*nao, n_orb)`` complex — the **converged** spinors in the AO basis. ⚠ These
        must be the orbitals the CI states were solved at; a dump built from one orbital set
        and one state set that do not match is Hermitian, plausible and wrong.
    spaces : :class:`kuiva.mcscf.orbopt.OrbitalSpaces` — the active/inactive partition.
    tdm : ``(ns, ns, n_act, n_act)`` — ``<I|E_tu|J>``.
    energies : ``(ns,)`` — total state energies [Eh] (``CASCIResult.total_energies``).
    properties : :class:`kuiva.interface.pyscf_bridge.PropertyIntegrals`.
    s_ao : ``(nao, nao)`` — the scalar AO overlap, the metric the spin operator needs.
    """
    from ..spinor.expand import spin_operator

    l_mo, s_mo = spinor_operators(coeff_ao, properties.two_component(), spin_operator(s_ao))
    act = np.asarray(spaces.active, dtype=int)
    inactive = np.asarray(spaces.inactive, dtype=int)

    inact_l = inactive_moment(l_mo, inactive, name="L", tol=inactive_tol)
    inact_s = inactive_moment(s_mo, inactive, name="S", tol=inactive_tol)

    ix = np.ix_(act, act)
    l_states = state_operator_matrices(np.stack([lk[ix] for lk in l_mo]), tdm, inact_l)
    s_states = state_operator_matrices(np.stack([sk[ix] for sk in s_mo]), tdm, inact_s)

    moment_ao = properties.moment_operator()
    if moment_ao is None:
        from .multiplet import magnetic_moment_matrices
        mu = magnetic_moment_matrices(l_states, s_states, g_e=g_electron)
    else:
        # ⚠ The picture-changed moment does NOT separate into an L part and an S part: the
        # four-component magnetic interaction is the odd operator c alpha.A, so what is
        # transformed is the whole of (L + 2S). The 2 is Dirac's g factor; the QED anomaly is
        # not part of that operator and is added here as (g_e - 2) S, on the picture-changed
        # spin operator when one was built and on the bare S otherwise.
        m_mo = spinor_operator(coeff_ao, moment_ao)
        inact_m = inactive_moment(m_mo, inactive, name="L+2S", tol=inactive_tol)
        m_states = state_operator_matrices(np.stack([mk[ix] for mk in m_mo]), tdm, inact_m)
        anomaly_ao = properties.anomaly_spin()
        if anomaly_ao is None:
            anomaly_states = s_states
        else:
            a_mo = spinor_operator(coeff_ao, anomaly_ao)
            inact_a = inactive_moment(a_mo, inactive, name="S (picture-changed)",
                                      tol=inactive_tol)
            anomaly_states = state_operator_matrices(
                np.stack([ak[ix] for ak in a_mo]), tdm, inact_a)
        mu = -(m_states + (float(g_electron) - 2.0) * anomaly_states)
        # L and S below stay the BARE operators. They remain perfectly good operators and the
        # file still reports them; what changed is that mu is no longer built from them, which
        # is what the header has to say.

    d_states, inact_d = _dipole_states(coeff_ao, properties, act, inactive, tdm,
                                       inactive_tol=inactive_tol)

    return PropertyMatrices(
        energies=np.asarray(energies, dtype=float).ravel(), mu=mu, l=l_states, s=s_states,
        gauge_origin=np.asarray(properties.gauge_origin, dtype=float).ravel(),
        origin_label=properties.origin_label, g_electron=float(g_electron),
        active_space=active_space,
        provenance=dict(provenance or {}, properties=properties.provenance()),
        inactive_l=inact_l, inactive_s=inact_s, comments=tuple(comments),
        # ⚠ Left EMPTY when there is no picture change, so the header keeps writing the bare
        # "none" it always has: this field feeds a stored file that committed references and
        # external consumers already parse, and widening its default value would move every
        # one of them for no gain.
        picture_change=("" if moment_ao is None
                        else str(properties.provenance().get("picture_change", ""))),
        d=d_states, nuclear_dipole=properties.nuclear_dipole_vector(), inactive_d=inact_d,
        molecular_charge=int(getattr(properties, "molecular_charge", 0) or 0),
        dipole_picture_change=("" if getattr(properties, "dipole_picture_change", None) is None
                               else str(properties.provenance()
                                        .get("dipole_picture_change", ""))))


def _dipole_states(coeff_ao: np.ndarray, properties, act: np.ndarray, inactive: np.ndarray,
                   tdm: np.ndarray, *, inactive_tol: float) -> Tuple[Optional[np.ndarray],
                                                                     np.ndarray]:
    """``(d^{IJ}, inactive_trace)`` — the total electric dipole in the state basis [e a_0].

    ``d = -(r - R_G)`` over the electrons plus, **on the diagonal only**, the nuclear
    ``sum_A Z_A (R_A - R_G)``. Three terms, and each of the three is a way to get a plausible
    wrong answer on its own:

    * dropping the nuclear term leaves a diagonal that is not a dipole moment, while every
      transition element stays exactly right — so nothing looks wrong;
    * dropping the inactive term leaves the *valence* dipole wearing the name of the total,
      and unlike ``L`` and ``S`` that term is **not** zero here (``r`` is time even);
    * adding the nuclear term off the diagonal would give every pair of states a spurious
      transition dipole proportional to the nuclear charge.

    The neutral-molecule symmetry check is what tests all three at once: for a molecule whose
    point group forbids a dipole, the three terms must cancel to zero, and they only do if all
    three are present and correctly signed.
    """
    if not getattr(properties, "has_dipole", False):
        return None, np.zeros(3)
    d_mo = spinor_operator(coeff_ao, properties.dipole_two_component())
    # ⚠ expect_zero=False: r is time EVEN, so a Kramers-paired inactive set contributes twice
    # its expectation value rather than cancelling. Warning here would be warning that the
    # core electrons exist.
    inact_d = inactive_moment(d_mo, inactive, name="d", tol=inactive_tol, expect_zero=False)
    ix = np.ix_(act, act)
    nuclear = properties.nuclear_dipole_vector()
    return (state_operator_matrices(np.stack([dk[ix] for dk in d_mo]), tdm,
                                    inact_d + nuclear),
            inact_d)


# --- the file ------------------------------------------------------------------------------

_ELEMENT_FMT = "{:6d} {:6d}  {:+.16e} {:+.16e}\n"


def write_dump(path, matrices: PropertyMatrices, *, title: str = "",
               include_l_s: bool = True, include_dipole: bool = True,
               threshold: float = 0.0) -> Path:
    """Write the property-matrix file and return its path.

    Parameters
    ----------
    include_l_s : bool
        Also write ``L`` and ``S`` separately. They are not part of the external contract —
        ``H`` and ``mu`` are — but they cost little and they are what an argument about a
        g factor gets settled with. Turn them off for a large state count.
    include_dipole : bool
        Also write the three electric dipole matrices ``d_x, d_y, d_z``. **On by default**, and
        adding them does **not** move ``FORMAT_VERSION``: the version tracks a change in the
        *meaning* of a stored field, not the arrival of a new one, so an existing consumer that
        reads ``H`` and ``mu`` is unaffected. Turn it off for a large state count, or when the
        reference carried no dipole integrals at all (in which case nothing is written either
        way).
    threshold : float
        Skip matrix elements smaller than this in modulus. ``0.0`` (the default) writes every
        element, which keeps the file's row count predictable from ``n_states`` alone.

    ⚠ Emits a standing ``WARNING`` about the treatment of the property operators every time it
    is called — whichever treatment was used. That is deliberate and it is not configurable:
    the file it produces will outlive this conversation, and the one thing worse than an
    approximation is a file that does not say it was made. A **non-default** treatment warns
    the more loudly of the two, because it is the one a reader will not be expecting.
    """
    path = Path(path)
    n = matrices.n_states
    if matrices.picture_changed:
        log.warning("the property operators in %s carry the X2C PICTURE CHANGE (%s). This is "
                    "NOT the default and it is not what OpenMolcas RASSI does, so mu in this "
                    "file is not comparable element for element with a file written without "
                    "it, and L and S are written as the bare operators mu was NOT built from. "
                    "The treatment is recorded in the header", path.name,
                    matrices.picture_change)
    else:
        log.warning("the property operators L and S carry NO picture-change transformation "
                    ": they are the bare non-relativistic AO operators used "
                    "unchanged in the two-component basis. This matches OpenMolcas RASSI, so a "
                    "cross-code comparison is like-for-like. The approximation has been "
                    "measured and is small: below 0.3%% on every free-ion multiplet g factor "
                    "from Z=5 to Z=81, 0.02%% on a 3d complex's ground doublet, and it splits "
                    "no degeneracy at all. It is recorded in the header of %s", path.name)

    write_dipole = bool(include_dipole) and matrices.has_dipole
    if write_dipole and matrices.dipole_is_origin_dependent:
        log.warning("this molecule carries a charge of %+d, so the ELECTRIC DIPOLE is "
                    "origin dependent: every diagonal element of d in %s -- and every "
                    "invariant built from within one degenerate block -- shifts by -q R_G "
                    "with the gauge origin, which is %s at (%.6f, %.6f, %.6f) bohr. Transition "
                    "elements between distinct states are unaffected and may be compared "
                    "freely. The charge and the origin are both in the header",
                    int(matrices.molecular_charge), path.name, matrices.origin_label,
                    *np.asarray(matrices.gauge_origin).ravel())

    blocks: List[Tuple[str, np.ndarray, str, str]] = [
        ("H", matrices.hamiltonian, "Eh",
         "effective Hamiltonian; DIAGONAL, see the header")]
    for k, axis in enumerate("xyz"):
        blocks.append(("mu_" + axis, matrices.mu[k], "mu_B",
                       "magnetic moment, {}".format(axis)))
    if write_dipole:
        for k, axis in enumerate("xyz"):
            blocks.append(("d_" + axis, matrices.d[k], "e*a0",
                           "electric dipole, {}; electronic + nuclear (diagonal only)"
                           .format(axis)))
    if include_l_s:
        for k, axis in enumerate("xyz"):
            blocks.append(("L_" + axis, matrices.l[k], "hbar",
                           "orbital angular momentum, {}".format(axis)))
        for k, axis in enumerate("xyz"):
            blocks.append(("S_" + axis, matrices.s[k], "hbar", "spin, {}".format(axis)))

    header = [
        ("format", "KUIVA_PROPERTY_MATRICES"),
        ("format_version", str(FORMAT_VERSION)),
        # The code version beside the format version, and they answer different questions:
        # `format_version` says whether this parser may read the file at all, `code_version`
        # says which Kuiva computed the numbers in it. A stored product
        # outlives the session that wrote it.
        ("code_version", _kuiva_version()),
        ("n_states", str(n)),
        ("energy_unit", "Eh"),
        ("moment_unit", "mu_B"),
        ("g_electron", "{:.11f}".format(matrices.g_electron)),
        ("gauge_origin_bohr", " ".join("{:.12f}".format(x) for x in
                                       np.asarray(matrices.gauge_origin).ravel())),
        ("gauge_origin_choice", matrices.origin_label),
        ("active_space", matrices.active_space or "unspecified"),
        ("hamiltonian_is_diagonal", "yes"),
        ("picture_change_on_properties", matrices.picture_change or "none"),
        ("phase_convention", "arbitrary (not canonicalized)"),
    ]
    if write_dipole:
        header.extend([
            ("dipole_unit", "e*a0"),
            ("dipole_includes_nuclear", "yes (diagonal only)"),
            ("nuclear_dipole_ea0", " ".join(
                "{:.12f}".format(x) for x in np.asarray(matrices.nuclear_dipole).ravel())),
            ("molecular_charge", str(int(matrices.molecular_charge))),
            ("picture_change_on_dipole", matrices.dipole_picture_change or "none"),
            ("dipole_origin_dependence",
             "diagonal elements and block-internal invariants shift by -q R_G; transition "
             "elements between distinct states do not" if matrices.dipole_is_origin_dependent
             else "none (neutral molecule)"),
        ])

    lines: List[str] = []
    w = lines.append
    w("# Kuiva property matrices in the basis of the spin-orbit eigenstates.\n")
    if title:
        w("# {}\n".format(title))
    w("#\n")
    w("# H is DIAGONAL: this CI is already two-component, so its roots ARE the spin-orbit\n"
      "# eigenstates and there is no separate spin-orbit mixing step. A reader coming from a\n"
      "# two-step (scalar CASSCF + RASSI) workflow should expect otherwise.\n")
    w("#\n")
    if matrices.picture_changed:
        w("# WARNING: the X2C picture change IS applied to the moment operator, which is NOT\n"
          "# the default and NOT what OpenMolcas RASSI does. mu below is therefore not\n"
          "# comparable with a file written without it. mu was built from the transformed\n"
          "# (L + 2S) plus (g_e - 2) S, NOT from the L and S blocks of this file, which are\n"
          "# the bare operators and are written for reference only. See the header.\n")
    else:
        w("# WARNING: no picture-change transformation is applied to L and S. They are the bare\n"
          "# non-relativistic AO operators used unchanged in the two-component basis, which is\n"
          "# what OpenMolcas RASSI does. The approximation has been measured: it is below 0.3%\n"
          "# on every free-ion multiplet g factor from Z=5 to Z=81, 0.02% on the ground doublet\n"
          "# of a 3d complex, and it splits no degeneracy. It grows with Z, so treat it as a\n"
          "# bound for the elements measured and not as one for a heavier system.\n")
    w("#\n")
    if write_dipole:
        w("# d_x/d_y/d_z are the TOTAL electric dipole in e*a0: the electronic operator\n"
          "# -(r - R_G) over all electrons, plus the nuclear sum_A Z_A (R_A - R_G) added to\n"
          "# the DIAGONAL only. So a diagonal element is that state's dipole moment and an\n"
          "# off-diagonal one is a transition dipole. The nuclear vector is in the header, so\n"
          "# the two parts can be separated again.\n")
        if matrices.dipole_is_origin_dependent:
            w("#\n")
            w("# WARNING: this molecule is CHARGED, so the electric dipole depends on the\n"
              "# gauge origin. Diagonal elements and any invariant formed inside one\n"
              "# degenerate block shift by -q R_G if the origin moves; transition elements\n"
              "# between distinct states do not. Compare charged systems only at one origin.\n")
        w("#\n")
    w("# WARNING: state phases are arbitrary and degenerate states mix arbitrarily. Compare\n"
      "# these matrices only through invariants: degeneracy patterns, relative energies, and\n"
      "# M_ij = Tr_block(mu_i mu_j) with its principal g values. For the dipole the invariants\n"
      "# are Tr_block(d_i d_j) and the block-to-block line strength sum |d_IJ|^2.\n")
    w("#\n")
    for line in matrices.comments:
        w("# {}\n".format(line))
    if matrices.comments:
        w("#\n")

    w("[HEADER]\n")
    for key, value in header:
        w("{:32s} {}\n".format(key, value))
    w("[END]\n\n")

    w("# Which Hamiltonian produced this: screening and decoupling\n"
      "# records in full. A property matrix that does not say whether the two-electron\n"
      "# spin-orbit picture change was included is not interpretable -- the difference is\n"
      "# 5-30% on every splitting in it.\n")
    w("[PROVENANCE]\n")
    w(json.dumps(matrices.provenance, sort_keys=True, indent=2))
    w("\n[END]\n\n")

    rel = matrices.relative_energies_cm()
    w("[ENERGIES]\n")
    w("# index    energy [Eh]                relative [cm^-1]\n")
    for i, e in enumerate(np.asarray(matrices.energies, dtype=float).ravel()):
        w("{:6d}  {:+.16e}  {:+.8e}\n".format(i, float(e), float(rel[i])))
    w("[END]\n\n")

    w("[INACTIVE]\n")
    w("# sum over inactive spinors of <i|A|i>; exactly zero for a Kramers-paired inactive\n"
      "# set, since L and S are both time odd. Computed, not assumed.\n")
    w("L  " + " ".join("{:+.6e}".format(x) for x in np.asarray(matrices.inactive_l)) + "\n")
    w("S  " + " ".join("{:+.6e}".format(x) for x in np.asarray(matrices.inactive_s)) + "\n")
    if write_dipole:
        w("# d is the ELECTRONIC inactive share of the dipole and is NOT zero: r is time\n"
          "# EVEN, so a Kramers pair contributes twice its expectation value instead of\n"
          "# cancelling. It is already inside the d matrices below, and is written here only\n"
          "# so the total can be taken apart again.\n")
        w("d  " + " ".join("{:+.6e}".format(x)
                           for x in np.asarray(matrices.inactive_d)) + "\n")
    w("[END]\n\n")

    for name, mat, unit, note in blocks:
        a = np.asarray(mat, dtype=np.complex128)
        w("[MATRIX {}]\n".format(name))
        w("shape      {} {}\n".format(n, n))
        w("unit       {}\n".format(unit))
        w("hermitian  yes\n")
        w("# {}\n".format(note))
        w("# row    col     Re                        Im\n")
        for i in range(n):
            for j in range(n):
                if threshold and abs(a[i, j]) < threshold:
                    continue
                w(_ELEMENT_FMT.format(i, j, float(a[i, j].real), float(a[i, j].imag)))
        w("[END]\n\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    # ⚠ Written whole, then moved into place: a dump truncated by an interrupt is worse than
    # no dump, because it parses. Same discipline as the checkpoint writer.
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text("".join(lines))
    tmp.replace(path)
    out.blank(log)
    out.entry(log, "property matrices written to", str(path), "",
              "{} states, {} matrices".format(n, len(blocks)))
    return path


def read_dump(path) -> Dict[str, object]:
    """Parse a file written by :func:`write_dump`. The round-trip test, and a worked example.

    Returns ``{"header": {...}, "provenance": {...}, "energies": ndarray,
    "inactive": {"L": ndarray, "S": ndarray}, "matrices": {name: complex ndarray}}``.

    Refuses a file whose ``format_version`` it does not know, rather than guessing — the
    version exists precisely so that a consumer can refuse.
    """
    text = Path(path).read_text().splitlines()
    header: Dict[str, str] = {}
    provenance: Dict[str, object] = {}
    energies: List[float] = []
    inactive: Dict[str, np.ndarray] = {}
    matrices: Dict[str, np.ndarray] = {}

    section: Optional[str] = None
    buffer: List[str] = []
    current: Optional[np.ndarray] = None
    for raw in text:
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("["):
            tag = line.strip()[1:-1]
            if tag == "END":
                if section == "PROVENANCE":
                    provenance = json.loads("\n".join(buffer))
                section, buffer, current = None, [], None
                continue
            section = tag.split()[0]
            if section == "MATRIX":
                name = tag.split()[1]
                current = None
                matrices[name] = None            # placeholder, sized by its `shape` line
                buffer = [name]
            continue
        if section == "HEADER":
            key, _, value = line.strip().partition(" ")
            header[key] = value.strip()
        elif section == "PROVENANCE":
            buffer.append(line)
        elif section == "ENERGIES":
            energies.append(float(line.split()[1]))
        elif section == "INACTIVE":
            parts = line.split()
            inactive[parts[0]] = np.array([float(x) for x in parts[1:]])
        elif section == "MATRIX":
            parts = line.split()
            if parts[0] == "shape":
                current = np.zeros((int(parts[1]), int(parts[2])), dtype=np.complex128)
                matrices[buffer[0]] = current
            elif parts[0] in ("unit", "hermitian"):
                continue
            elif current is None:
                # A clear refusal beats a TypeError from indexing None: the shape line is what
                # sizes the matrix, so an element before it means the file is malformed.
                raise ValueError(
                    "{}: matrix {!r} has an element line before its `shape` line, so there is "
                    "nothing to read it into".format(path, buffer[0] if buffer else "?"))
            else:
                i, j = int(parts[0]), int(parts[1])
                current[i, j] = complex(float(parts[2]), float(parts[3]))

    version = int(header.get("format_version", -1))
    if version != FORMAT_VERSION:
        raise ValueError(
            "{} declares format_version {} and this parser knows version {}; refusing to "
            "guess. The version exists so that a consumer can refuse rather than "
            "misinterpret.".format(path, version, FORMAT_VERSION))
    return {"header": header, "provenance": provenance,
            "energies": np.array(energies, dtype=float), "inactive": inactive,
            "matrices": matrices}


__all__ = ["FORMAT_VERSION", "DEFAULT_INACTIVE_TOL", "PropertyMatrices",
           "property_matrices", "spinor_operators", "spinor_operator", "inactive_moment",
           "state_operator_matrices", "write_dump", "read_dump"]
