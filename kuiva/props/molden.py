"""Molden output for spinor densities.

What this writes, and what it is not
------------------------------------
The molden format holds **real orbitals**: a visualizer expands ``psi(r) = sum_mu c_mu
chi_mu(r)`` and draws its isosurfaces. A two-component spinor is not a real orbital and its
density's square root is not in the AO span, so a spinor cannot be written to this format
directly (:mod:`kuiva.spinor.density` gives the argument and the counter-example).

What is written instead is the **exact decomposition of the spinor density into real
components**::

    rho(r) = sum_k w_k * (u_k . chi(r))^2

with at most four ``u_k`` per Kramers pair. Each ``u_k`` is written as one molden orbital with
occupation ``w_k``. Consequences the reader of a file must know, and which the file's own
header states:

* **A single entry is not "the spinor".** It is one component of its density. Where the
  leading weight is ~1 the component *is* the orbital picture; where the weights are
  ~0.5/0.5 — the normal case for a spin-orbit-coupled ``m_j`` function — the density is the
  sum of two isosurfaces and looking at one alone is misleading.
* **Phase and sign are meaningless.** The components come from a density; only their squares
  entered it.
* **The total electron density of the file is exact.** Occupations are ``n_group * w_k``, so
  summing every component's square with its occupation reproduces the true density, and the
  occupations sum to the number of electrons. That is the invariant to test against. (Up to
  components dropped below ``tolerance``, which is a deliberate 1e-3 of a group by default:
  writing 1e-6-weight components would fill a viewer with noise.)
* **Empty orbitals are written too**, with ``Occup= 0``. An active space is chosen by looking
  at the orbitals a calculation *might* populate, so a dump that silently omitted them would
  be useless for the job it exists to do.
* **One entry per Kramers pair by default.** Partners have identical densities, so a
  per-spinor listing is redundant, and inside a larger degenerate manifold a *single* spinor's
  density is not even well defined (block sums are invariant; single spinors are not).

Format details that are easy to get wrong
-----------------------------------------
* **AO ordering.** Molden orders ``l >= 2`` shells ``m = 0, +1, -1, ...`` where the integral
  library orders them ``-l..+l``, and p shells agree between the two.
  :func:`kuiva.basis.layout.molden_ao_order` is the single definition, asserted equal to
  PySCF's own ``molden.order_ao_index``.
* **Spherical only** (``[5d] [7f] [9g]``). Cartesian bases are refused at ingestion, so
  no Cartesian normalization fix-up is needed — that path is where most molden writers go
  wrong.
* **Contraction coefficients** are over normalized primitives, which is what the integral
  library stores and what the format expects; they are written through unchanged.

.. warning::
   ⚠ **h functions (``l = 5``) are not part of the molden standard.** The format defines up to
   ``[9g]``. By default they are **dropped**, with a WARNING and with the discarded Loewdin
   weight measured and written into the file header, because a silently truncated orbital is a
   picture of something else. ``include_high_l=True`` writes them anyway, in the obvious
   continuation of molden's own m ordering, with an ``[11h]`` marker: **not standard**, not
   readable by every program, and supported by some visualizers under some interpretation.
   It exists so a code that does read them (e.g. Kaijo,
   https://github.com/mansikkamaki/kaijo) gets the whole function. The file header records
   which of the two happened.

References
----------
* G. Schaftenaar, J. H. Noordik, "Molden: a pre- and post-processing program for molecular and
  electronic structures", J. Comput.-Aided Mol. Design 14, 123 (2000),
  doi:10.1023/A:1008193805436.
* G. Schaftenaar, E. Vlieg, G. Vriend, "Molden 2.0: quantum chemistry meets proteins",
  J. Comput.-Aided Mol. Design 31, 789 (2017), doi:10.1007/s10822-017-0042-5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..basis.layout import ANGULAR, MOLDEN_MAX_L, AOLayout, molden_ao_order
from ..spinor.density import DEFAULT_WEIGHT_TOLERANCE, decompose_density
from ..util import output as out
from ..util.logging import get_logger
from .population import (lowdin_coefficients, resolve_groups,
                         warn_if_groups_not_degenerate)

log = get_logger(__name__)

#: Components below this fraction of a group's density are not written: they would be plotted
#: as numerical noise, and every extra entry is a file a human has to click through.
DEFAULT_COMPONENT_TOLERANCE = 1e-3


@dataclass(frozen=True)
class MoldenOrbital:
    """One entry of the ``[MO]`` section: a real orbital over the AO basis."""

    coefficients: np.ndarray            # (nao,) real, in library AO order
    occupation: float
    energy: float
    label: str = "A"


def _format_number(value: float) -> str:
    return "{:18.14g}".format(value)


def _write_header(fh, layout: AOLayout, *, max_l: Optional[int],
                  provenance: Sequence[str]) -> None:
    fh.write("[Molden Format]\n")
    # The version with the writer's name: a molden file is opened months later by a viewer that
    # says nothing about where it came from.
    from .. import __version__
    fh.write("written by kuiva {}\n".format(__version__))
    for line in provenance:
        fh.write("{}\n".format(line))

    fh.write("[Atoms] (AU)\n")
    for ia in range(layout.natm):
        x, y, z = layout.coords_bohr[ia]
        fh.write("{:s}   {:d}   {:d}   {:18.14f}   {:18.14f}   {:18.14f}\n".format(
            layout.atom_symbols[ia], ia + 1, int(round(layout.atom_charges[ia])), x, y, z))

    fh.write("[GTO]\n")
    for ia in range(layout.natm):
        fh.write("{:d} 0\n".format(ia + 1))
        for sh in layout.shells:
            if sh.atom != ia or (max_l is not None and sh.l > max_l):
                continue
            fh.write(" {:s}   {:2d} 1.00\n".format(ANGULAR[sh.l], sh.nprim))
            for ip in range(sh.nprim):
                fh.write("    {:s}  {:s}\n".format(_format_number(sh.exponents[ip]),
                                                   _format_number(sh.coefficients[ip])))
        fh.write("\n")

    written_max_l = max((sh.l for sh in layout.shells
                         if max_l is None or sh.l <= max_l), default=0)
    markers = ["[5d]", "[7f]", "[9g]"]
    if written_max_l >= 5:
        # ⚠ Non-standard; see the module docstring. Written only when h functions are in.
        markers.append("[11h]")
    fh.write("\n".join(markers) + "\n")
    fh.write("\n")


def write_molden(path, layout: AOLayout, orbitals: Sequence[MoldenOrbital], *,
                 include_high_l: bool = False,
                 provenance: Sequence[str] = ()) -> Dict[str, object]:
    """Write real orbitals to a molden file. The low-level writer.

    Parameters
    ----------
    path : str or Path
    layout : AOLayout
        Geometry and basis, from :func:`kuiva.interface.pyscf_bridge.ao_layout`.
    orbitals : sequence of MoldenOrbital
        Coefficients in **library** AO order; the molden permutation is applied here.
    include_high_l : bool
        Write ``l > 4`` shells rather than dropping them. ⚠ Non-standard — see the module
        docstring. A WARNING is emitted either way when such shells are present.
    provenance : sequence of str
        Free-text lines written into the header, after ``[Molden Format]``. This is where the
        Hamiltonian provenance record and the "these are density components" statement go.

    Returns
    -------
    dict with ``n_orbitals``, ``n_ao_written`` and ``dropped_ao`` — what the caller reports.
    """
    max_l = None if include_high_l else MOLDEN_MAX_L
    high_l = layout.high_l_mask(MOLDEN_MAX_L)
    n_high = int(high_l.sum())
    if n_high:
        if include_high_l:
            log.warning("writing %d basis functions with l > %d into a molden file. The "
                        "format defines up to [9g]; l > 4 uses the obvious continuation of "
                        "molden's m ordering with an [11h] marker and is NOT standard - many "
                        "visualizers will refuse the file or misread it.", n_high,
                        MOLDEN_MAX_L)
        else:
            log.warning("dropping %d basis functions with l > %d from the molden file: the "
                        "format defines only up to [9g]. The written orbitals are truncated, "
                        "not the ones computed. Pass include_high_l=True to write them "
                        "anyway (non-standard).", n_high, MOLDEN_MAX_L)

    order = molden_ao_order(layout.shells, max_l=max_l)
    with open(path, "w") as fh:
        _write_header(fh, layout, max_l=max_l, provenance=provenance)
        fh.write("[MO]\n")
        for orb in orbitals:
            coeff = np.asarray(orb.coefficients, dtype=float).ravel()
            if coeff.size != layout.nao:
                raise ValueError("orbital has {} coefficients for {} basis functions"
                                 .format(coeff.size, layout.nao))
            fh.write(" Sym= {:s}\n".format(orb.label))
            fh.write(" Ene= {:15.10g}\n".format(orb.energy))
            fh.write(" Spin= Alpha\n")
            fh.write(" Occup= {:10.5f}\n".format(orb.occupation))
            for i, mu in enumerate(order):
                fh.write(" {:3d}    {:s}\n".format(i + 1, _format_number(coeff[mu])))

    return {"n_orbitals": len(orbitals), "n_ao_written": int(order.size),
            "dropped_ao": 0 if include_high_l else n_high}


# --- Spinor densities ----------------------------------------------------------------------

@dataclass(frozen=True)
class SpinorMoldenReport:
    """What a spinor molden dump produced — the numbers the caller logs and tests assert."""

    path: str
    n_groups: int
    n_orbitals: int
    dropped_ao: int
    #: (n_groups,) leading component weight: 1 means one picture is the whole density.
    leading_weight: np.ndarray
    #: (n_groups,) number of components written.
    n_components: np.ndarray
    #: (n_groups,) Loewdin weight lost to dropped high-l functions; zero if none were dropped.
    truncated_weight: np.ndarray
    #: Electrons represented by the written occupations.
    written_electrons: float

    def report(self, logger=None) -> None:
        """Log the INFO summary block."""
        logger = logger or log
        out.entries(logger, [
            ("molden file", self.path),
            ("density groups written", self.n_groups),
            ("real components written", self.n_orbitals),
            ("electrons represented", self.written_electrons, "", "", "{:.6f}"),
            ("min leading component weight",
             float(self.leading_weight.min()) if self.n_groups else 1.0, "",
             "1.0 = a real orbital; 0.5 = two components needed", "{:.4f}"),
            ("max components for one group",
             int(self.n_components.max()) if self.n_groups else 0),
        ])
        if self.dropped_ao:
            worst = float(self.truncated_weight.max()) if self.n_groups else 0.0
            out.entry(logger, "weight lost to dropped l > 4", worst, "",
                      "of one spinor, worst group", "{:.2e}")


def write_spinor_molden(path, layout: AOLayout, c_ao: np.ndarray, s_ao: np.ndarray, *,
                        occupation: Optional[np.ndarray] = None,
                        energy: Optional[np.ndarray] = None,
                        columns: Optional[Sequence[int]] = None,
                        group: Union[str, Sequence[Sequence[int]]] = "kramers",
                        include_high_l: bool = False,
                        tolerance: float = DEFAULT_COMPONENT_TOLERANCE,
                        provenance: Sequence[str] = (),
                        report: bool = True) -> SpinorMoldenReport:
    """Write spinor densities to a molden file as exact real components.

    Parameters
    ----------
    path : str or Path
    layout : AOLayout
    c_ao : ndarray (2*nao, n) complex
        Spinor coefficients in the **AO** basis, spin-blocked row layout.
    s_ao : ndarray (nao, nao)
        AO overlap. Needed because the components are made S-orthonormal and because the
        weight lost to dropped high-l functions is measured in the Loewdin metric.
    occupation, energy : ndarray (n,), optional
        Occupation numbers and orbital energies over ``c_ao``'s columns. The energies are the
        diagonal Fock elements in this basis where that is what the caller has; they are
        written to ``Ene=`` and are for ordering and identification only. Occupations default
        to 1 per spinor, which makes the file a picture of the orbitals rather than of the
        density.
    columns : sequence of int, optional
        Which spinors to write; all of them by default. **Selecting the active space is the
        normal use** — a file with every virtual orbital in it is large and nobody looks at it.
    group : {"kramers", "none", "degenerate"} or list of index groups
        How columns are grouped before decomposition; see
        :func:`kuiva.props.population.orbital_populations`. The default sums Kramers pairs,
        which is what makes each entry an invariant object.
    include_high_l : bool
        See the module docstring. Default ``False``: drop, warn, and measure what was lost.
    tolerance : float
        Components below this fraction of a group's density are not written.
    """
    c_ao = np.asarray(c_ao)
    nao = s_ao.shape[0]
    columns = (np.arange(c_ao.shape[1]) if columns is None
               else np.asarray(columns, dtype=int).ravel())
    occ = (np.ones(columns.size) if occupation is None
           else np.asarray(occupation, dtype=float)[columns])
    ene = (np.zeros(columns.size) if energy is None
           else np.asarray(energy, dtype=float)[columns])
    sel = np.ascontiguousarray(c_ao[:, columns])

    groups = resolve_groups(group, columns.size, occ)
    # A group's density is scaled by ONE occupation below, so the group must be degenerate.
    warn_if_groups_not_degenerate(groups, occ, group)

    # Loewdin weight sitting on functions a standard molden file cannot hold. Measured, not
    # bounded: it is the honest statement of what a truncated picture is missing.
    high_l = layout.high_l_mask(MOLDEN_MAX_L)
    lost = np.zeros(len(groups))
    if high_l.any() and not include_high_l:
        ct = lowdin_coefficients(sel, s_ao)
        per_orbital = np.abs(ct[:nao]) ** 2 + np.abs(ct[nao:]) ** 2
        for ig, g in enumerate(groups):
            weight = per_orbital[:, g].sum(axis=1)
            lost[ig] = float(weight[high_l].sum() / max(weight.sum(), 1e-300))

    orbitals: List[MoldenOrbital] = []
    leading = np.zeros(len(groups))
    n_comp = np.zeros(len(groups), dtype=int)
    for ig, g in enumerate(groups):
        # ⚠ Decomposed with **unit** weights, then scaled by the group's mean occupation.
        # Weighting the decomposition by the occupations directly would make an *empty*
        # orbital produce no components at all — and the empty members of an active shell are
        # exactly what one wants to look at when choosing an active space. The two are
        # identical whenever the occupations within a group are equal, which is what a
        # degenerate block means and what the grouping rules deliver — and a group whose
        # occupations differ is warned about above.
        dec = decompose_density(sel[:, g], None, s_ao,
                                tolerance=max(tolerance, DEFAULT_WEIGHT_TOLERANCE))
        total = float(dec.weights.sum())
        leading[ig] = dec.leading_weight / total if total > 0.0 else 0.0
        n_comp[ig] = dec.n_components
        scale = float(occ[g].mean())
        label = _group_label(columns, g)
        for k in range(dec.n_components):
            orbitals.append(MoldenOrbital(
                coefficients=dec.components[:, k],
                occupation=scale * float(dec.weights[k]),
                energy=float(ene[g].mean()),
                label="{}_c{}".format(label, k + 1)))

    lines = list(provenance) + [
        "Contents: EXACT real components of two-component spinor densities, not orbitals.",
        "  rho(r) = sum_k Occup_k * (component_k . chi(r))^2 ; phases are meaningless.",
        "  One group of components per {} ; Sym= labels them <spinors>_c<component>."
        .format({"kramers": "Kramers pair", "none": "spinor",
                 "degenerate": "degenerate block"}.get(group, "orbital group")
                if isinstance(group, str) else "orbital group"),
        "  Summing Occup_k * component_k^2 over the file reproduces the electron density.",
    ]
    if high_l.any():
        lines.append("  l > 4 functions: {}."
                     .format("INCLUDED, non-standard [11h] ordering" if include_high_l
                             else "DROPPED (molden defines up to [9g]); orbitals truncated"))

    info = write_molden(path, layout, orbitals, include_high_l=include_high_l,
                        provenance=lines)

    result = SpinorMoldenReport(
        path=str(path), n_groups=len(groups), n_orbitals=len(orbitals),
        dropped_ao=int(info["dropped_ao"]), leading_weight=leading, n_components=n_comp,
        truncated_weight=lost,
        written_electrons=float(sum(o.occupation for o in orbitals)))
    if report:
        out.subsection(log, "Molden spinor-density dump")
        result.report()
    return result


def _group_label(columns: np.ndarray, g: np.ndarray) -> str:
    """Molden ``Sym=`` labels may not contain spaces, so this is not the population one."""
    idx = columns[g]
    return str(int(idx[0])) if idx.size == 1 else "{}-{}".format(int(idx.min()),
                                                                 int(idx.max()))


__all__ = ["DEFAULT_COMPONENT_TOLERANCE", "MoldenOrbital", "SpinorMoldenReport",
           "write_molden", "write_spinor_molden"]
