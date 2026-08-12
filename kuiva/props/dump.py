"""The property-matrix dump: Kuiva's actual product.

Everything upstream of this file exists to produce four matrices in the basis of the
spin-orbit eigenstates — the effective Hamiltonian ``H`` and the three magnetic-moment
components ``mu_x``, ``mu_y``, ``mu_z`` — which an **external** ITO / Stevens / crystal-field
code turns into the quantities an experiment measures. The project scope puts that analysis explicitly out of
scope, so this file is the boundary: a plain-text, versioned, self-describing file, entirely
separate from the log stream (logging never contaminates machine-readable output).

The physics in one line
-----------------------
``mu = -(L + g_e S) mu_B``, and in the many-electron state basis

    mu^{IJ}_k = -( sum_{tu} (L_k + g_e S_k)_{tu} gamma^{IJ}_{tu}
                   + delta_IJ sum_{i in inactive} (L_k + g_e S_k)_{ii} )   [mu_B]

with ``gamma^{IJ}_{tu} = <I|E_tu|J>`` the transition density matrices — the third
consumer of the *same* excitation map and the *same* intermediate the sigma vector and the
RDMs are built from. ``L`` and ``S`` are one-electron operators, so nothing beyond the
one-particle transition densities is ever needed, however large the CI.

⚠ **The inactive term is computed, not assumed away.** A Kramers pair ``(psi, T psi)``
contributes ``<psi|A|psi> + <T psi|A|T psi> = 0`` for any time-odd ``A``, and both ``L`` and
``S`` are time odd, so a Kramers-paired inactive set contributes exactly nothing. That is a
theorem about the *orbitals*, and a CASSCF that has broken Kramers symmetry in the inactive
space violates it. :func:`inactive_moment` measures it and warns above a tolerance rather
than skipping the term — the failure it guards against is a moment matrix that is silently
missing a core contribution, which looks entirely plausible.

Four things about this file that are decisions, not details
-----------------------------------------------------------
1. ⚠ **``H`` is diagonal**, unlike OpenMolcas RASSI's. Kuiva's CI is already two-component,
   so its roots *are* the spin-orbit eigenstates: there is no separate spin-orbit mixing step
   to leave off-diagonal elements behind. A reader coming from a two-step (scalar CASSCF +
   RASSI) workflow will expect otherwise, so the header says it.
2. ⚠ **No picture change is applied to the property operators** (an explicit standing decision). ``L`` and
   ``S`` are the bare non-relativistic AO operators used unchanged in the two-component
   basis. This matches RASSI, which is what makes the Tier-2 comparison like-for-like, and it
   is an approximation of **unmeasured size**. :func:`write_dump` emits a ``WARNING`` at the
   point of writing and records the omission in the header, in the same way the mean field records its
   own standing obligations. Removing it means transforming ``L`` and ``S`` with the ``R``
   matrix of :mod:`kuiva.x2c.decouple` (Peng & Reiher 2012).
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
from .multiplet import G_ELECTRON, HARTREE_TO_CM, Multiplet, analyse_spectrum

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
    ct = c.conj().T
    return (np.stack([ct @ lk @ c for lk in l]),
            np.stack([ct @ sk @ c for sk in s]))


def inactive_moment(op_mo: np.ndarray, inactive: Sequence[int], *,
                    name: str = "operator", tol: float = DEFAULT_INACTIVE_TOL) -> np.ndarray:
    """``sum_{i in inactive} A_ii`` for each component — computed, checked, never assumed.

    Returns the ``(3,)`` real trace. It **must** vanish for a Kramers-paired inactive set,
    because ``L`` and ``S`` are time-odd and a Kramers pair contributes equal and opposite
    expectation values; a nonzero result above ``tol`` means the inactive spinors are no
    longer Kramers paired, which is a statement about the orbitals and is worth a warning.
    Nonzero or not, the value is *used*, so a broken inactive space degrades the moments
    rather than silently dropping a term.
    """
    idx = np.asarray(inactive, dtype=int).ravel()
    op = np.asarray(op_mo)
    if idx.size == 0:
        return np.zeros(3)
    trace = np.array([np.real(np.trace(opk[np.ix_(idx, idx)])) for opk in op])
    worst = float(np.max(np.abs(trace)))
    if worst > tol:
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

    def analyse(self, tol_cm: float = 1.0) -> List[Multiplet]:
        """The phase-invariant reduction — the **only** sound way to compare these.

        Degeneracy pattern, relative energies, and the invariant ``M_ij = Tr_block(mu_i mu_j)``
        with its principal g values. Any validation of this file's contents must go through
        here; element-by-element comparison of :attr:`mu` compares arbitrary phases.
        """
        return analyse_spectrum(self.energies, self.mu, tol_cm=tol_cm)

    def hermiticity_error(self) -> float:
        """``max |A - A^dag|`` over the three moment components — a structural self-check."""
        mu = np.asarray(self.mu)
        return float(np.max(np.abs(mu - mu.conj().transpose(0, 2, 1)))) if mu.size else 0.0

    def report(self, logger=None) -> None:
        """The INFO summary: the multiplet table, never the matrices themselves."""
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
        table = out.Table(logger, [
            out.col_count("block", 7), out.Column("states", "{:d}", 8),
            out.Column("E [cm^-1]", out.CM_FMT, 14),
            out.Column("spread", "{:.3e}", 11),
            out.Column("g_1", "{:.4f}", 9), out.Column("g_2", "{:.4f}", 9),
            out.Column("g_3", "{:.4f}", 9)])
        table.start()
        for i, m in enumerate(self.analyse()):
            g = m.g_values if m.g_values else (float("nan"),) * 3
            table.row(i, m.size, m.energy_cm, m.spread_cm, g[0], g[1], g[2])
        table.end("g values are principal values of M_ij = Tr_block(mu_i mu_j); "
                  "phases are arbitrary ")

    def write(self, path, **kwargs) -> Path:
        """Write the property dump file. See :func:`write_dump`."""
        return write_dump(path, self, **kwargs)

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
    from .multiplet import magnetic_moment_matrices
    mu = magnetic_moment_matrices(l_states, s_states, g_e=g_electron)

    return PropertyMatrices(
        energies=np.asarray(energies, dtype=float).ravel(), mu=mu, l=l_states, s=s_states,
        gauge_origin=np.asarray(properties.gauge_origin, dtype=float).ravel(),
        origin_label=properties.origin_label, g_electron=float(g_electron),
        active_space=active_space,
        provenance=dict(provenance or {}, properties=properties.provenance()),
        inactive_l=inact_l, inactive_s=inact_s, comments=tuple(comments))


# --- the file ------------------------------------------------------------------------------

_ELEMENT_FMT = "{:6d} {:6d}  {:+.16e} {:+.16e}\n"


def write_dump(path, matrices: PropertyMatrices, *, title: str = "",
               include_l_s: bool = True, threshold: float = 0.0) -> Path:
    """Write the property-matrix file and return its path.

    Parameters
    ----------
    include_l_s : bool
        Also write ``L`` and ``S`` separately. They are not part of the external contract —
        ``H`` and ``mu`` are — but they cost little and they are what an argument about a
        g factor gets settled with. Turn them off for a large state count.
    threshold : float
        Skip matrix elements smaller than this in modulus. ``0.0`` (the default) writes every
        element, which keeps the file's row count predictable from ``n_states`` alone.

    ⚠ Emits the standing ``WARNING`` about the missing picture change on the property
    operators every time it is called. That is deliberate and it is not configurable: the file
    it produces will outlive this conversation, and the one thing worse than the approximation
    is a file that does not say it was made.
    """
    path = Path(path)
    n = matrices.n_states
    log.warning("the property operators L and S carry NO picture-change transformation "
                ": they are the bare non-relativistic AO operators used "
                "unchanged in the two-component basis. This matches OpenMolcas RASSI, so a "
                "cross-code comparison is like-for-like, but the size of the approximation "
                "has not been measured. It is recorded in the header of %s", path.name)

    blocks: List[Tuple[str, np.ndarray, str, str]] = [
        ("H", matrices.hamiltonian, "Eh",
         "effective Hamiltonian; DIAGONAL, see the header")]
    for k, axis in enumerate("xyz"):
        blocks.append(("mu_" + axis, matrices.mu[k], "mu_B",
                       "magnetic moment, {}".format(axis)))
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
        ("picture_change_on_properties", "none"),
        ("phase_convention", "arbitrary (not canonicalized)"),
    ]

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
    w("# WARNING: no picture-change transformation is applied to L and S. They are the bare\n"
      "# non-relativistic AO operators used unchanged in the two-component basis, which is\n"
      "# what OpenMolcas RASSI does; the size of the approximation is unmeasured.\n")
    w("#\n")
    w("# WARNING: state phases are arbitrary and degenerate states mix arbitrarily. Compare\n"
      "# these matrices only through invariants: degeneracy patterns, relative energies, and\n"
      "# M_ij = Tr_block(mu_i mu_j) with its principal g values.\n")
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
           "property_matrices", "spinor_operators", "inactive_moment",
           "state_operator_matrices", "write_dump", "read_dump"]
