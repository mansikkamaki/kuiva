"""Slater-Condon radial parameters ``F^k``, ``G^k`` and ``R^k`` from a converged atomic solution.

What is extracted, and how
--------------------------
For a spherical atom every two-electron integral over the orbitals of a set of shells is a
sum of products of a *pure number* and a *radial parameter*:

.. math::

    (p\\,m_p,\\; q\\,m_q \\,|\\, r\\,m_r,\\; s\\,m_s)
        \\;=\\; \\sum_k A^k[m_p, m_q, m_r, m_s]\\; R^k(pr;\\, qs) ,

with the angular tensors ``A^k`` of :mod:`kuiva.extras.angular` and the Condon-Shortley radial
parameters ``R^k(ab;cd)``, of which the direct ``F^k(a,b) = R^k(ab;ab)`` and exchange
``G^k(a,b) = R^k(ab;ba)`` are the familiar special cases. The parameters are what this module
returns, in hartree.

**They are obtained by inversion rather than by radial quadrature**, and that is the design
decision worth stating. The alternative — build ``P_{nl}(r)`` on a radial grid and integrate
``r_<^k / r_>^{k+1}`` numerically — needs a second radial integrator, a grid whose accuracy is
its own study, and the primitive normalization convention of the integral library restated
correctly somewhere. Inverting the expansion above instead reuses the factorized two-electron
integrals the rest of the code already produces, and hands back a **diagnostic for free**: the
system is enormously overdetermined — a single ``4f``/``4f`` class is 2401 equations for four
unknowns — so the residual of the least-squares solution says whether the expansion the
parameters are *defined* by holds at all.

⚠ **What that residual does and does not measure, stated because the obvious reading is
wrong.** The shell orbitals it works on are one radial function placed in each ``m`` channel,
so they are **exactly** a radial function times a real harmonic whether or not the solution
they came from was spherical — and the expansion above is then exact by the Laplace expansion,
identically. The residual therefore sees the *machinery*: an angular convention that does not
match the orbitals, an index placement that is wrong, orbitals that are not in fact one radial
function per shell, or a two-electron factorization whose error is anisotropic (which is what
plain column-pivoted Cholesky produces and complete symmetry orbits do not). It does **not**
see a non-spherical SCF. That is what the shell anisotropy of
:class:`kuiva.extras.shells.AtomicShells` measures, on the Fock operator, and the two numbers
are reported side by side because neither implies the other.

⚠ **The residual is never used to correct anything.** A large one means the input was not what
the extraction assumes; rescaling parameters to absorb it would produce a plausible set of
numbers for a state that does not exist.

What the parameters contain
---------------------------
The two-electron integrals here are the **bare Coulomb integrals over the X2C radial
functions**: the scalar-relativistic contraction of the radial functions is in them, and no
two-electron picture-change correction is, which is the standard content of a parameter set
from a one-electron-transformed relativistic Hamiltonian. The picture-change correction that
the atomic mean field supplies is a *one-electron* operator in this scheme and reaches the
spin-orbit constants, not these parameters.

⚠ **Two rules bind any splitting quoted from a parameter extracted here**, and both have
produced convincing-looking method errors elsewhere in this project. State which construction
produced it — these are frozen average-of-configuration orbitals of one fixed configuration,
not a self-consistent optimization of any particular state — and never compare against a value
obtained in a different basis set, because no radial parameter can recover a basis truncation.

References
----------
* E. U. Condon, G. H. Shortley, *The Theory of Atomic Spectra*, Cambridge University Press
  (1935), Chapter VI — the definitions of ``R^k``, ``F^k`` and ``G^k`` and their ordering
  convention, which is the one used here.
* J. C. Slater, "The Theory of Complex Spectra", Phys. Rev. **34**, 1293 (1929) — the original
  radial integrals.
* R. D. Cowan, *The Theory of Atomic Structure and Spectra*, University of California Press
  (1981), Chapters 6 and 14 — parameter conventions and the configuration-average energy
  expressions these parameters reproduce.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from kuiva.extras.angular import admissible_k, angular_tensor
from kuiva.extras.shells import (AtomicShells, ShellConfiguration, ShellOrbitals,
                                 extract_shells, shell_label)
from kuiva.extras.spin_orbit import SpinOrbitConstants, extract_spin_orbit
from kuiva.integrals.transform import (DEFAULT_CHOLESKY_TOL, ThreeIndexAO, assemble_4c,
                                       transform_3c)
from kuiva.util import output as out
from kuiva.util import resources as res
from kuiva.util.logging import get_logger
from kuiva.util.timing import timer

log = get_logger(__name__)

#: Bumped whenever the *meaning* of anything already in the parameter file changes — a unit, a
#: sign convention, the ordering convention of ``R^k``, the name of a column. Adding a section
#: or a header key does not require a bump. A consumer that does not recognise the version must
#: refuse the file rather than guess; :func:`read_parameters` does.
#:
#: ⚠ Independent of the code version and of every other format version in the project.
#:
#: **2** — the radial functions carry a stated phase convention (positive in the outer region),
#: which gives the sign of every genuine cross parameter ``R^k`` a defined meaning. In version 1
#: those signs were whatever the eigensolver returned: reproducible for one calculation and
#: unrelated between two. ``F^k``, ``G^k`` and ``zeta`` are unaffected, being quadratic in every
#: radial function they involve.
FORMAT_VERSION = 2

#: Largest relative least-squares residual a parameter class may show before extraction warns,
#: scaled by the largest two-electron integral of the class.
#:
#: ⚠ **It is a consistency diagnostic, not an accuracy one, and not a sphericity one** — see
#: the module docstring for what it can and cannot see. It is also **not** a measure of the
#: Cholesky threshold: that shifts every integral of a class nearly together and cancels out of
#: the residual, so a loose threshold moves the parameters without moving this number.
#:
#: Measured on converged average-of-configuration atoms (O, C, Ti(3+)) with the one-centre
#: orbit-pivoted Cholesky factorization: **1e-13 and below** throughout, independent of the
#: threshold. Against **1e-1 to 1** for shell orbitals that are deliberately not one radial
#: function per shell — a little of the 2s mixed into a 2p channel, or two ``m`` channels of a
#: 3d rotated into each other. Twelve orders of separation, so the bound is set well above the
#: floor and is not delicate.
PARAMETER_RESIDUAL_TOLERANCE = 1e-8


def parameter_integral_memory_gb(n_orbitals: int) -> float:
    """Memory [GB] of the four-index array over ``n_orbitals`` shell orbitals (exact).

    ``n^4`` doubles: no permutational packing, because the array is indexed directly by four
    magnetic quantum numbers while the design matrices are built and the whole point is that it
    is small — the orbitals are one shell's worth each, never a correlation space.
    """
    return float(n_orbitals) ** 4 * 8.0 / res.BYTES_PER_GB


# --- Parameter identity ----------------------------------------------------------------------

def _class_label(kind: str, shells: Sequence[str], k: int) -> str:
    if kind in ("F", "G"):
        return "{}{}({},{})".format(kind, k, shells[0], shells[1])
    return "R{}({} {};{} {})".format(k, shells[0], shells[1], shells[2], shells[3])


@dataclass(frozen=True)
class SlaterParameter:
    """One radial parameter, with the quality of the fit it came from.

    Attributes
    ----------
    kind : str
        ``"F"``, ``"G"`` or ``"R"``. The first two are the direct and exchange special cases;
        ``"R"`` is a genuine cross parameter, involving three or four distinct shells or two
        shells in a pairing that is neither.
    k : int
        The rank of the multipole.
    shells : tuple of str
        Two labels for ``F`` and ``G``, four for ``R`` — and for ``R`` they are in
        **Condon-Shortley order**, ``R^k(ab;cd)`` with ``a, c`` on electron 1 and ``b, d`` on
        electron 2.
    value : float
        The parameter, in hartree.
    rms_residual, max_residual : float
        Residuals of the least-squares solution for the whole class this parameter belongs to
        [Eh]; ``relative_residual`` scales the largest of them by the largest integral in the
        class. Every parameter of one class carries the same three numbers, because the fit is
        a property of the class rather than of a single ``k``.
    n_equations : int
        Integrals the class was fitted to. Always far larger than the number of parameters.
    """

    kind: str
    k: int
    shells: Tuple[str, ...]
    value: float
    rms_residual: float
    max_residual: float
    relative_residual: float
    n_equations: int

    @property
    def label(self) -> str:
        """``"F2(4f,4f)"``, ``"G3(4f,6s)"``, ``"R2(4f 5d;4f 6s)"`` — the identity, as text."""
        return _class_label(self.kind, self.shells, self.k)

    @property
    def value_cm(self) -> float:
        """The parameter in wavenumbers, the unit atomic parameters are quoted in."""
        from kuiva.props.multiplet import HARTREE_TO_CM

        return self.value * HARTREE_TO_CM

    def __repr__(self) -> str:
        return "SlaterParameter({} = {:.6f} Eh)".format(self.label, self.value)


@dataclass(frozen=True)
class RadialParameters:
    """Every nonvanishing parameter among a set of shells, with the shells they came from.

    Parameters that are zero by the angular selection rules are **absent rather than listed as
    zero**: ``(4f 5d | 6s 6s)`` admits no ``k`` at all, and printing it as ``0.0`` would invite
    the reading that it was computed and found small.
    """

    parameters: Tuple[SlaterParameter, ...]
    shells: AtomicShells

    def __len__(self) -> int:
        return len(self.parameters)

    def __iter__(self):
        return iter(self.parameters)

    def __getitem__(self, key) -> SlaterParameter:
        """By label (``params["F2(4f,4f)"]``) or by position."""
        if isinstance(key, int):
            return self.parameters[key]
        for parameter in self.parameters:
            if parameter.label == key:
                return parameter
        raise KeyError("{} is not among the {} parameters extracted: {}".format(
            key, len(self.parameters), ", ".join(p.label for p in self.parameters)))

    def as_dict(self) -> Dict[str, float]:
        """``{label: value in Eh}`` — for reference records and JSON provenance."""
        return {p.label: p.value for p in self.parameters}

    def of_kind(self, kind: str) -> Tuple[SlaterParameter, ...]:
        return tuple(p for p in self.parameters if p.kind == kind)

    @property
    def max_relative_residual(self) -> float:
        """The worst class residual in the set — the end-to-end consistency diagnostic."""
        return max((p.relative_residual for p in self.parameters), default=0.0)

    def report(self, logger=None) -> None:
        """The parameter table, through the output grammar.

        Both units, because both are read: hartree is what the numbers are computed in and
        wavenumbers are what atomic parameters are quoted in. The residual is per class, so it
        repeats down a class's rows.
        """
        logger = logger or log
        table = out.Table(logger, [out.Column("parameter", "{:s}", 20, align="<"),
                                   out.Column("k", "{:d}", 3),
                                   out.Column("value [Eh]", out.E_FMT, 21),
                                   out.Column("value [cm^-1]", out.CM_FMT, 15),
                                   out.col_sci("residual"),
                                   out.col_count("equations", 10)])
        table.start("Slater-Condon radial parameters, Condon-Shortley ordering")
        for p in self.parameters:
            table.row(p.label, p.k, p.value, p.value_cm, p.relative_residual, p.n_equations)
        table.end("parameters with no admissible k are absent, not zero; the residual is the "
                  "class fit, not an error bar")

    def __repr__(self) -> str:
        return "RadialParameters({} parameters over {}, worst residual {:.1e})".format(
            len(self.parameters), "/".join(self.shells.labels), self.max_relative_residual)


# --- The integrals over the shell orbitals -----------------------------------------------------

def shell_mo_integrals(shells: AtomicShells, factors: ThreeIndexAO, *,
                       buffer_gb: Optional[float] = None
                       ) -> Tuple[np.ndarray, Tuple[slice, ...]]:
    """Two-electron integrals over every orbital of every shell, plus each shell's slice.

    Returns ``(eri, slices)`` where ``eri[p, q, r, s]`` is the **chemists'** integral
    ``(pq|rs)`` and ``slices[i]`` selects shell ``i``'s ``2l+1`` columns, in ascending ``m``.

    The route is the project's ordinary factorized one — the three-index AO factors, the
    orbital transform, then the four-index assembly — which matters for one reason beyond
    reuse: with the one-centre orbit-pivoted Cholesky factorization the decomposition's
    spherical symmetry is exact by construction rather than accurate to a threshold. A
    factorization that split degeneracies at the threshold would put its own anisotropy into
    every parameter and into every residual reported beside them.

    The transform takes two-component coefficients with rows blocked ``[alpha; beta]``; the
    scalar shell orbitals enter as ``[C; 0]``, whose spin sum is the spatial transform exactly.
    """
    orbitals = list(shells)
    columns = np.hstack([shell.coefficients for shell in orbitals])
    n_total = int(columns.shape[1])
    nao = factors.nao
    if columns.shape[0] != nao:
        raise ValueError("the shell orbitals span {} AO functions and the integral factors "
                         "{}; they are not from the same calculation"
                         .format(columns.shape[0], nao))

    two_component = np.zeros((2 * nao, n_total), dtype=np.float64)
    two_component[:nao] = columns
    b = transform_3c(factors, two_component, two_component, dtype=np.float64,
                     buffer_gb=buffer_gb)
    res.require("shell two-electron integrals", parameter_integral_memory_gb(n_total),
                note="{} shell orbitals".format(n_total),
                advice=["extract parameters for fewer shells at a time"])
    eri = assemble_4c(b)

    slices, start = [], 0
    for shell in orbitals:
        slices.append(slice(start, start + shell.size))
        start += shell.size
    return eri, tuple(slices)


# --- Enumerating the classes -------------------------------------------------------------------

def _canonical_classes(shells: Sequence[ShellOrbitals]
                       ) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, ...]]]:
    """Every distinct parameter class among the shells, as ``(pair1, pair2, ks)``.

    A radial parameter ``R^k(ab;cd)`` depends on its two **electron pairs** ``{a, c}`` and
    ``{b, d}`` and on nothing else: the integrand is symmetric in each pair separately and in
    the exchange of the two electrons, which is the group of order eight that makes ``F`` and
    ``G`` the only names most parameters ever need. So a class is an unordered pair of
    unordered shell pairs, and enumerating those enumerates each parameter exactly once.

    Classes admitting no ``k`` are dropped here: they are identically zero by symmetry and are
    not parameters.
    """
    n = len(shells)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    classes = []
    for a, first in enumerate(pairs):
        for second in pairs[a:]:
            ks = admissible_k(shells[first[0]].l, shells[first[1]].l,
                              shells[second[0]].l, shells[second[1]].l)
            if ks:
                classes.append((first, second, ks))
    return classes


def _name_class(labels: Sequence[str], first: Tuple[int, int], second: Tuple[int, int]
                ) -> Tuple[str, Tuple[str, ...]]:
    """``(kind, shell labels)`` for a class, in the convention of :class:`SlaterParameter`.

    The three cases are the whole naming scheme. ``F^k(x,y)`` is the class whose two electron
    pairs are each a repeated shell; ``G^k(x,y)`` is the one whose two electron pairs are the
    *same* mixed pair; everything else keeps the general four-label form. ⚠ ``G^k(x,x)`` is not
    a separate parameter — it is ``F^k(x,x)`` — and falls into the first case, which is why the
    order of these tests matters.
    """
    i, j = first
    p, q = second
    if i == j and p == q:
        return "F", (labels[i], labels[p])
    if first == second:
        return "G", (labels[i], labels[j])
    return "R", (labels[i], labels[p], labels[j], labels[q])


# --- The extraction ------------------------------------------------------------------------------

def extract_parameters(shells: AtomicShells, data=None, *,
                       factors: Optional[ThreeIndexAO] = None,
                       cholesky_tol: float = DEFAULT_CHOLESKY_TOL,
                       residual_tol: float = PARAMETER_RESIDUAL_TOLERANCE,
                       buffer_gb: Optional[float] = None,
                       report: bool = False) -> RadialParameters:
    """Fit ``F^k``, ``G^k`` and ``R^k`` to the two-electron integrals over ``shells``.

    Parameters
    ----------
    shells : AtomicShells
        The ``m``-aligned shell orbitals of a converged atomic solution
        (:func:`kuiva.extras.shells.extract_shells`). ⚠ Their column order — ascending ``m``
        from ``-l`` to ``+l`` — is what the angular tensors are written in, and it is not the
        integral library's within-shell order.
    data
        The ingested solution the shells came from, used to build the integral factors.
        Optional if ``factors`` is given directly, which is what a caller extracting several
        parameter sets from one solution should do.
    cholesky_tol : float
        Threshold of the Cholesky factorization, an error bound on a single two-electron
        integral. It propagates to the parameters roughly one-for-one and is recorded by
        whoever stores them.
    residual_tol : float
        Relative residual above which a class warns. See
        :data:`PARAMETER_RESIDUAL_TOLERANCE`.

    Returns
    -------
    RadialParameters

    Notes
    -----
    Each class is solved by ordinary least squares over **all** of its magnetic quantum number
    combinations at once, including the ones whose angular coefficients vanish — they are rows
    of zeros in the design matrix and nonzero entries in the right-hand side if anything is
    wrong, which is exactly what a residual should be able to see. The system is consistent to
    machine precision whenever the shell orbitals really are one radial function per shell, so
    the least squares is a diagnostic device rather than a statistical one and no weighting is
    applied.
    """
    if factors is None:
        if data is None:
            raise ValueError("extraction needs either the ingested solution it should build "
                             "integral factors from, or the factors themselves")
        # ⚠ ``release_eri=False``: this borrows a container it does not own. The pipeline
        # releases the stored integral array once the factors replace it, but an extraction
        # is one analysis among several a caller may run over the *same* ingested atom —
        # a second shell set, a second threshold — and dropping the array under it would
        # make the second call refuse.
        factors = ThreeIndexAO.from_scalar_data(data, cholesky_tol, report=report,
                                                release_eri=False)

    orbitals = list(shells)
    if not orbitals:
        raise ValueError("no shells to extract parameters among")
    labels = [shell.label for shell in orbitals]

    with timer("Slater-Condon extraction"):
        eri, slices = shell_mo_integrals(shells, factors, buffer_gb=buffer_gb)
        extracted: List[SlaterParameter] = []
        for first, second, ks in _canonical_classes(orbitals):
            block = eri[slices[first[0]], slices[first[1]],
                        slices[second[0]], slices[second[1]]]
            design = np.stack([
                angular_tensor(orbitals[first[0]].l, orbitals[first[1]].l,
                               orbitals[second[0]].l, orbitals[second[1]].l, k).ravel()
                for k in ks], axis=1)
            rhs = np.asarray(block, dtype=float).ravel()
            values, _, _, _ = np.linalg.lstsq(design, rhs, rcond=None)
            deviation = design @ values - rhs
            rms = float(np.sqrt(np.mean(deviation ** 2)))
            worst = float(np.max(np.abs(deviation))) if deviation.size else 0.0
            scale = max(float(np.max(np.abs(rhs))), 1e-300)
            relative = worst / scale
            kind, names = _name_class(labels, first, second)
            if relative > residual_tol:
                log.warning(
                    "the %s^k(%s) class is reproduced by its radial parameters only to a "
                    "relative %.2e (bound %.0e), where the expansion they are defined by is "
                    "exact. Either the shell orbitals are not one radial function per shell "
                    "or the two-electron factorization does not preserve the atomic symmetry; "
                    "the parameters describe the integrals only as well as this number.",
                    kind, ", ".join(names), relative, residual_tol)
            for k, value in zip(ks, values):
                extracted.append(SlaterParameter(
                    kind=kind, k=int(k), shells=names, value=float(value),
                    rms_residual=rms, max_residual=worst, relative_residual=relative,
                    n_equations=int(rhs.size)))

    order = {"F": 0, "G": 1, "R": 2}
    extracted.sort(key=lambda p: (order[p.kind], p.shells, p.k))
    return RadialParameters(parameters=tuple(extracted), shells=shells)


# --- The whole calculation: result, driver, report, file ---------------------------------------

#: The convention statement, written verbatim into every file and quoted nowhere else, so the
#: file and the code cannot drift apart. It is what a reader needs to know before using a
#: single number below it: the definition, the ordering, and what the parameters contain.
CONVENTION = """\
R^k(ab;cd) = Int Int P_a(r1) P_c(r1) [r_<^k / r_>^(k+1)] P_b(r2) P_d(r2) dr1 dr2
F^k(a,b)   = R^k(ab;ab)   (direct)        G^k(a,b) = R^k(ab;ba)   (exchange)

Condon-Shortley ordering: the FIRST and THIRD labels sit on electron 1 and the second and
fourth on electron 2, with P_nl(r) = r R_nl(r) the radial functions of the average-of-
configuration solution described in the header. In the chemists' notation of a two-electron
integral, (p q | r s) = sum_k A^k R^k(p r; q s) -- note the reindexing.

PHASE CONVENTION, which fixes the SIGN of every R^k that is not an F^k or a G^k: each radial
function is taken POSITIVE IN ITS OUTER REGION, P_nl(r) > 0 as r -> infinity. F^k, G^k and
zeta are quadratic in every radial function they involve and do not depend on it; a genuine
cross parameter R^k(ab;cd) is linear in two of them and changes sign with either, so without a
stated convention its sign would be an artifact of the eigensolver and could not be compared
between two ions. The other common convention, P_nl(r) > 0 as r -> 0, differs by (-1)^(n-l-1).

The two-electron parameters are the bare Coulomb integrals over the X2C radial functions: the
scalar-relativistic contraction of the radial functions is in them and no two-electron
picture-change correction is. The spin-orbit constants zeta contain the one-electron X2C
operator plus the two-electron screening if and only if the PROVENANCE section says so.

These are frozen-orbital quantities of one fixed average-of-configuration reference, not
self-consistent values for any particular state, and they may not be compared against values
obtained in another basis set."""


def _provenance(data) -> Dict[str, object]:
    """What Hamiltonian produced this, as JSON-ready data — **never empty**.

    A solution ingested without spin-orbit coupling has no
    :class:`~kuiva.interface.pyscf_bridge.SpinOrbitX2C` to ask, and the answer is then a
    record that says exactly that rather than an absent section: "no screening record" and
    "the screening record says none" must not look alike in a stored file, and neither may be
    silence.
    """
    if getattr(data, "soc", None) is not None:
        record = dict(data.soc.provenance())
    else:
        record = {
            "method": "",
            "one_electron": "X2C (scalar; the two-component operator was not ingested)",
            "screening": {"method": "none",
                          "note": "the reference was run with with_soc=False"},
        }
    record["radial_parameters_contain"] = (
        "bare Coulomb integrals over the X2C radial functions; no two-electron picture change")
    record["construction"] = (
        "frozen average-of-configuration orbitals of one fixed configuration")
    return record


@dataclass(frozen=True)
class SlaterCondonResult:
    """Everything one Slater-Condon calculation produced, and what it may be trusted for.

    The container the driver returns: the parameters, the spin-orbit constants, the shells
    they were extracted from, the solution underneath, and the two diagnostics that say
    whether any of it means anything — the **shell anisotropy** (is the solution spherical?)
    and the **worst class residual** (does the expansion the parameters are defined by
    hold?). ⚠ Neither implies the other and both are reported, in the log and in the file.

    Attributes
    ----------
    element : str
    charge : int
        Derived from the configuration's electron count, never stated separately.
    configuration : ShellConfiguration
    basis : object
        As given to the driver — a family name or a per-element mapping.
    shells : AtomicShells
    parameters : RadialParameters
    spin_orbit : SpinOrbitConstants
        Empty when the reference was run without spin-orbit ingestion.
    data : ScalarX2CData
        The average-of-configuration solution itself, kept so that a caller can extract
        further shells or parameters without repeating the SCF.
    cholesky_tol : float
        The two-electron factorization threshold the parameters were obtained at. It
        propagates to them roughly one-for-one and is recorded in the file.
    provenance : dict
        :func:`_provenance` of the solution — mandatory, and mandatory in the file header.
    """

    element: str
    charge: int
    configuration: ShellConfiguration
    basis: object
    shells: AtomicShells
    parameters: RadialParameters
    spin_orbit: SpinOrbitConstants
    data: object
    cholesky_tol: float = DEFAULT_CHOLESKY_TOL
    provenance: Dict[str, object] = field(default_factory=dict)

    # -- construction -----------------------------------------------------------------------

    @classmethod
    def from_solution(cls, data, configuration, element: str = "", *, basis=None,
                      shells: Optional[Sequence[str]] = None,
                      cholesky_tol: float = DEFAULT_CHOLESKY_TOL,
                      spin_orbit: bool = True,
                      buffer_gb: Optional[float] = None) -> "SlaterCondonResult":
        """Extract everything from a solution that has already been converged.

        The half of :func:`slater_condon_parameters` that is not the SCF, so a caller who has
        a :class:`~kuiva.interface.pyscf_bridge.ScalarX2CData` in hand — from a checkpoint, or
        from one run being analysed several ways — does not repeat it.

        ``spin_orbit`` asks for the constants and is silently skipped when the solution
        carries no two-component operator, since that is a property of how it was run.
        """
        configuration = ShellConfiguration.coerce(configuration)
        atomic = extract_shells(data, configuration, shells=shells)
        parameters = extract_parameters(atomic, data, cholesky_tol=cholesky_tol,
                                        buffer_gb=buffer_gb)
        constants = (extract_spin_orbit(atomic, data)
                     if spin_orbit and getattr(data, "soc", None) is not None
                     else SpinOrbitConstants(constants=(), provenance={}))
        return cls(element=element or "", charge=configuration.charge(element) if element
                   else 0, configuration=configuration,
                   basis=getattr(data, "basis_meta", {}) if basis is None else basis,
                   shells=atomic, parameters=parameters, spin_orbit=constants, data=data,
                   cholesky_tol=float(cholesky_tol), provenance=_provenance(data))

    # -- what it is -------------------------------------------------------------------------

    @property
    def anisotropy(self) -> float:
        """Sphericity of the solution, on the Fock operator. The diagnostic no residual
        replaces (:mod:`kuiva.extras.spin_orbit` and this module's docstring both say why)."""
        return self.shells.anisotropy

    @property
    def max_relative_residual(self) -> float:
        """The worst class residual over the radial parameters *and* the constants."""
        return max(self.parameters.max_relative_residual,
                   self.spin_orbit.max_relative_residual)

    @property
    def labels(self) -> Tuple[str, ...]:
        return self.shells.labels

    def as_dict(self) -> Dict[str, float]:
        """``{label: value in Eh}`` over the parameters and the constants together.

        The constants are keyed ``zeta(4f)`` so that the two families cannot collide in a
        record that stores both.
        """
        values = dict(self.parameters.as_dict())
        values.update({"zeta({})".format(c.shell): c.zeta for c in self.spin_orbit})
        return values

    # -- output -----------------------------------------------------------------------------

    def report(self, logger=None) -> None:
        """The whole result, through the output grammar (a section, then three tables)."""
        logger = logger or log
        out.section(logger, "Slater-Condon parameters")
        screening = self.provenance.get("screening", {})
        out.entries(logger, [
            ("element", "{} ({:+d})".format(self.element, self.charge) if self.charge
             else self.element),
            ("configuration", self.configuration.label, "", self.configuration.canonical),
            ("reference", "average of configuration (spin-restricted, scalar X2C)"),
            ("SCF energy", float(getattr(self.data, "e_scf", 0.0)), "Eh", "", out.E_FMT),
            ("SCF converged", bool(getattr(self.data, "converged", False))),
            ("shells", ", ".join(self.shells.labels)),
            ("two-electron screening in zeta",
             str(screening.get("method", "none")) if self.spin_orbit else "n/a (no zeta)"),
            ("Cholesky threshold", self.cholesky_tol, "", "on one integral", out.SCI_FMT),
            ("shell anisotropy", self.anisotropy, "", "sphericity of the solution",
             out.SCI_FMT),
            ("worst class residual", self.max_relative_residual, "",
             "consistency of the extraction", out.SCI_FMT),
        ])
        out.blank(logger)
        self.shells.report(logger)
        out.blank(logger)
        self.parameters.report(logger)
        out.blank(logger)
        self.spin_orbit.report(logger)

    def write(self, path, *, title: str = "") -> Path:
        """Write the machine-readable parameter file. See :func:`write_parameters`."""
        return write_parameters(path, self, title=title)

    def __repr__(self) -> str:
        return "SlaterCondonResult({}{}, {}, {} parameters, {} constants)".format(
            self.element, "{:+d}".format(self.charge) if self.charge else "",
            self.configuration.canonical, len(self.parameters), len(self.spin_orbit))


def slater_condon_parameters(element: str, configuration, *, basis,
                             shells: Optional[Sequence[str]] = None, zeta: bool = True,
                             screening: Optional[str] = None,
                             cholesky_tol: Optional[float] = None,
                             memory_gb: Optional[float] = None,
                             file=None, title: str = "", report: bool = True,
                             buffer_gb: Optional[float] = None,
                             **scf_options) -> SlaterCondonResult:
    """Slater-Condon parameters and spin-orbit constants of one atom or ion, start to finish.

    Runs the average-of-configuration scalar X2C SCF, extracts the shells, inverts the angular
    expansion for ``F^k``, ``G^k`` and ``R^k``, fits ``zeta_nl``, prints the report and
    optionally writes the file::

        from kuiva.extras import slater_condon_parameters

        result = slater_condon_parameters(
            "Dy", "[Xe] 4f9 5d1 6s1", basis="x2c-TZVPall-2c",
            shells=("4f", "5d", "6s"), file="dy_i.scp")

    Parameters
    ----------
    element : str
        Chemical symbol. The **charge is derived** from the configuration's electron count.
    configuration : str or ShellConfiguration
        ``"[Xe] 4f9 5d1 6s1"``. ⚠ An oxidation state is **refused**: which shells the
        electrons of Dy(1+) occupy is chemistry, and the answer changes every number here.
    basis : str or dict
        Through the basis registry, as the rest of the front end.
    shells : sequence of str, optional
        Which shells to compute parameters among (``("4f", "5d", "6s")``). The default is the
        configuration's **open** shells — the ones a parameter set is normally quoted for —
        falling back to every occupied shell when the configuration is closed.
    zeta : bool
        Extract the spin-orbit constants. ⚠ **It is what makes the run expensive**: it needs
        the two-component operator, whose default screening costs one four-component atomic
        solve per element — sub-second for a light atom, tens of minutes for a lanthanide,
        paid once ever thanks to the on-disk cache. ``zeta=False`` defaults ``screening`` to
        ``"none"`` and keeps the whole run to the SCF.
    screening : str, optional
        The two-electron picture change (:mod:`kuiva.amf`). ``None`` resolves to the project
        default when ``zeta`` is on and to ``"none"`` when it is not.
    cholesky_tol : float, optional
        Two-electron factorization threshold; ``None`` is
        :data:`kuiva.integrals.transform.DEFAULT_CHOLESKY_TOL`.
    file : path-like, optional
        Where to write the parameter file. ``None`` writes none.
    report : bool
        Print the tables. The file and the return value do not depend on it.
    **scf_options
        Passed to :func:`kuiva.interface.pyscf_bridge.run_scalar_aoc` — ``conv_tol``,
        ``max_cycle``, and the convergence aids ``level_shift``, ``damp``, ``init_guess``
        that an open-shell lanthanide actually needs.

    Returns
    -------
    SlaterCondonResult

    Notes
    -----
    ⚠ **Two numbers decide whether the output means anything and both are printed**: the shell
    anisotropy, which says whether the solution is spherical, and the worst class residual,
    which says whether the expansion the parameters are defined by holds. Neither implies the
    other — a symmetry-broken solution gives a residual at roundoff — so a run is read with
    both in hand.
    """
    from kuiva.interface.pyscf_bridge import run_scalar_aoc

    configuration = ShellConfiguration.coerce(configuration)
    if screening is None and not zeta:
        # ⚠ Spin-orbit ingestion changes no scalar quantity, so with no zeta wanted it is pure
        # cost — and the cost is a four-component atomic solve per element.
        screening = "none"
    if shells is None:
        wanted = configuration.open_shells() or configuration.shells
        shells = tuple(shell_label(n, l) for n, l, _ in wanted)

    data = run_scalar_aoc(element, configuration, basis=basis, with_soc=zeta,
                          screening=screening, memory_gb=memory_gb, **scf_options)
    result = SlaterCondonResult.from_solution(
        data, configuration, element, basis=basis, shells=shells, spin_orbit=zeta,
        cholesky_tol=DEFAULT_CHOLESKY_TOL if cholesky_tol is None else float(cholesky_tol),
        buffer_gb=buffer_gb)
    if report:
        result.report()
    if file is not None:
        result.write(file, title=title)
    return result


# --- The file ------------------------------------------------------------------------------

_PARAMETER_FMT = "{:<4s}{:3d}  {:<5s} {:<5s} {:<5s} {:<5s}  {:+.12e}  {:+18.4f}  {:.3e} {:8d}\n"
_ZETA_FMT = "{:<5s} {:2d}  {:+.12e}  {:+18.4f}  {:+18.4f}  {:.3e}\n"
_SHELL_FMT = "{:<5s} {:2d}  {:12.8f}  {:+.12e}  {:.3e} {:5d}\n"


def write_parameters(path, result: SlaterCondonResult, *, title: str = "") -> Path:
    """Write the parameter file and return its path.

    A sibling of the property dump and deliberately the same dull shape: ``#`` comments, a
    versioned ``[HEADER]``, a ``[PROVENANCE]`` block of JSON, then one whitespace-separated
    record per line under ``[SHELLS]``, ``[PARAMETERS]`` and ``[ZETA]``. It is a stored
    product that will outlive the session, so three things are not optional:

    * the **convention statement** (:data:`CONVENTION`) verbatim, because ``R^k(ab;cd)`` is
      ordered differently by different authors and a file of numbers without it is ambiguous;
    * the **provenance record**, which says whether the two-electron screening is in ``zeta``
      — a 5-30% difference — and is written even when it says ``"none"``;
    * **both units on every value**: hartree at full precision, wavenumbers at four decimals.
      The file deliberately carries more digits than the log, because a parameter is refitted
      and recombined by whoever reads it.

    Written whole and moved into place: a file truncated by an interrupt is worse than no
    file, because it parses.
    """
    from kuiva import __version__

    path = Path(path)
    configuration = result.configuration
    screening = result.provenance.get("screening", {})
    header = [
        ("format", "KUIVA_SLATER_CONDON"),
        ("format_version", str(FORMAT_VERSION)),
        # Two versions, two questions: `format_version` says whether a parser may read this
        # file at all, `code_version` says which Kuiva produced the numbers in it.
        ("code_version", str(__version__)),
        ("element", result.element or "unspecified"),
        ("charge", "{:+d}".format(result.charge)),
        ("electrons", str(configuration.n_electrons)),
        ("configuration", configuration.label),
        ("configuration_canonical", configuration.canonical),
        ("basis", str(result.basis)),
        ("reference", "average of configuration (spin-restricted, scalar X2C)"),
        ("scf_energy_hartree", "{:+.12e}".format(float(getattr(result.data, "e_scf", 0.0)))),
        ("scf_converged", "yes" if getattr(result.data, "converged", False) else "NO"),
        ("cholesky_tol", "{:.3e}".format(result.cholesky_tol)),
        ("shell_anisotropy", "{:.3e}".format(result.anisotropy)),
        ("max_relative_residual", "{:.3e}".format(result.max_relative_residual)),
        ("spin_orbit_screening", str(screening.get("method", "none"))),
        ("energy_unit", "Eh"),
        ("wavenumber_unit", "cm^-1"),
        ("hartree_to_cm", "{:.7f}".format(_hartree_to_cm())),
    ]

    lines: List[str] = []
    w = lines.append
    w("# Kuiva Slater-Condon parameters of a free atom or ion.\n")
    if title:
        w("# {}\n".format(title))
    w("#\n")
    for line in CONVENTION.splitlines():
        w("# {}\n".format(line) if line else "#\n")
    w("#\n")

    w("[HEADER]\n")
    for key, value in header:
        w("{:26s} {}\n".format(key, value))
    w("[END]\n\n")

    w("# Which Hamiltonian produced this. A stored spin-orbit constant that does not say\n"
      "# whether the two-electron screening is in it is not interpretable: the difference is\n"
      "# 5-30%. Written even when it says none.\n")
    w("[PROVENANCE]\n")
    w(json.dumps(result.provenance, sort_keys=True, indent=2))
    w("\n[END]\n\n")

    w("# The shells the parameters are over. 'electrons' is projected out of the converged\n"
      "# density; 'spread' is the shell's radial eigenvalue against the mean orbital energy\n"
      "# of the orbitals spanning it; 'block' is the degenerate group they form (2l+1 in a\n"
      "# clean solution).\n")
    w("[SHELLS]\n")
    w("# shell  l   electrons          eps [Eh]     spread [Eh] block\n")
    for shell in result.shells:
        w(_SHELL_FMT.format(shell.label, shell.l, shell.occupation, shell.energy,
                            shell.energy_spread, shell.degenerate_block))
    w("[END]\n\n")

    w("# One record per parameter. The four shell columns are the Condon-Shortley order\n"
      "# R^k(a b; c d); F and G name only two shells and leave the rest '-'. 'residual' is\n"
      "# the relative least-squares residual of the whole class, shared by its k values, and\n"
      "# it is a consistency diagnostic rather than an error bar.\n")
    w("[PARAMETERS]\n")
    w("# kind  k  a     b     c     d              value [Eh]        value [cm^-1]"
      "   residual      neq\n")
    for p in result.parameters:
        names = list(p.shells) + ["-"] * (4 - len(p.shells))
        w(_PARAMETER_FMT.format(p.kind, p.k, names[0], names[1], names[2], names[3],
                                p.value, p.value_cm, p.relative_residual, p.n_equations))
    w("[END]\n\n")

    w("# One-electron spin-orbit constants, H_SO = zeta l.s, fitted to the two-component\n"
      "# one-electron Hamiltonian over the same shell orbitals. 'splitting' is the\n"
      "# (2l+1) zeta / 2 separation of the j = l +- 1/2 levels for ONE electron in the shell,\n"
      "# with frozen orbitals. An s shell has no constant and is absent rather than zero.\n")
    w("[ZETA]\n")
    w("# shell  l             zeta [Eh]         zeta [cm^-1]"
      "    splitting [cm^-1]   residual\n")
    for c in result.spin_orbit:
        w(_ZETA_FMT.format(c.shell, c.l, c.zeta, c.zeta_cm, c.splitting_cm,
                           c.relative_residual))
    w("[END]\n\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text("".join(lines))
    tmp.replace(path)
    out.blank(log)
    out.entry(log, "Slater-Condon parameters written to", str(path), "",
              "{} parameters, {} constants".format(len(result.parameters),
                                                   len(result.spin_orbit)))
    return path


def read_parameters(path) -> Dict[str, object]:
    """Parse a file written by :func:`write_parameters`. The round-trip test, and an example.

    Returns ``{"header": {...}, "provenance": {...}, "shells": [{...}], "parameters":
    {label: {...}}, "zeta": {shell: {...}}}``, with the parameter labels rebuilt from the
    stored columns by the same function that names them in the first place.

    Refuses a file whose ``format_version`` it does not know, rather than guessing — the
    version exists precisely so that a consumer can refuse.
    """
    header: Dict[str, str] = {}
    provenance: Dict[str, object] = {}
    shells: List[Dict[str, object]] = []
    parameters: Dict[str, Dict[str, object]] = {}
    zeta: Dict[str, Dict[str, object]] = {}

    section: Optional[str] = None
    buffer: List[str] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            tag = line[1:-1]
            if tag == "END":
                if section == "PROVENANCE":
                    provenance = json.loads("\n".join(buffer))
                section, buffer = None, []
            else:
                section = tag.split()[0]
            continue
        if section == "HEADER":
            key, _, value = line.partition(" ")
            header[key] = value.strip()
        elif section == "PROVENANCE":
            buffer.append(line)
        elif section == "SHELLS":
            f = line.split()
            shells.append({"shell": f[0], "l": int(f[1]), "electrons": float(f[2]),
                           "energy": float(f[3]), "energy_spread": float(f[4]),
                           "degenerate_block": int(f[5])})
        elif section == "PARAMETERS":
            f = line.split()
            names = tuple(name for name in f[2:6] if name != "-")
            label = _class_label(f[0], names, int(f[1]))
            parameters[label] = {"kind": f[0], "k": int(f[1]), "shells": names,
                                 "value": float(f[6]), "value_cm": float(f[7]),
                                 "relative_residual": float(f[8]),
                                 "n_equations": int(f[9])}
        elif section == "ZETA":
            f = line.split()
            zeta[f[0]] = {"shell": f[0], "l": int(f[1]), "zeta": float(f[2]),
                          "zeta_cm": float(f[3]), "splitting_cm": float(f[4]),
                          "relative_residual": float(f[5])}

    version = int(header.get("format_version", -1))
    if version != FORMAT_VERSION:
        raise ValueError(
            "{} declares format_version {} and this parser knows version {}; refusing to "
            "guess. The version exists so that a consumer can refuse rather than "
            "misinterpret.".format(path, version, FORMAT_VERSION))
    return {"header": header, "provenance": provenance, "shells": shells,
            "parameters": parameters, "zeta": zeta}


def _hartree_to_cm() -> float:
    from kuiva.props.multiplet import HARTREE_TO_CM

    return float(HARTREE_TO_CM)


__all__ = ["CONVENTION", "FORMAT_VERSION", "PARAMETER_RESIDUAL_TOLERANCE",
           "RadialParameters", "SlaterCondonResult", "SlaterParameter", "extract_parameters",
           "parameter_integral_memory_gb", "read_parameters", "shell_mo_integrals",
           "slater_condon_parameters", "write_parameters"]
