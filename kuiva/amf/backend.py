"""The four-component atomic backend abstraction.

Why this module exists
----------------------
X2CAMF needs a **four-component atomic Dirac-Hartree-Fock** solution per element: its
converged density and its two-electron mean field, in the ``LL/LS/SL/SS`` blocking, so that
the same X2C decoupling that produced the one-electron Hamiltonian can be applied to the
two-electron operator. That solution is obtained from PySCF today, but the project scope keeps
four-component methodology open as a later possibility, which makes this a **hard
requirement**: the atomic 4c solver sits behind an interface from day one, so
that a native Kuiva backend is a one-argument change rather than a rewrite of everything built
on top of it.

The contract, and what it deliberately forbids
----------------------------------------------
1. :class:`AtomicDiracSolution` contains **plain NumPy arrays and metadata only**. No PySCF
   object may appear in it or in any protocol signature. If a framework object leaks through
   here, every consumer downstream is free to reach into it and the abstraction is gone.
2. **All matrices are in the spin-blocked spin-orbital basis of kuiva/spinor/expand.py** — rows and
   columns ordered ``[alpha; beta]`` over the same real scalar basis. This is *not* the
   j-adapted 2-spinor basis a four-component code naturally works in; converting to it is the
   backend's job, precisely so that nothing downstream ever has to know which basis the solver
   used. kuiva/spinor/expand.py is the one statement of that convention this project has, and a second one here would be a
   permanent source of sign and ordering errors.
3. The solution carries **everything needed to reconstruct the X2C decoupling from it** — the
   core Hamiltonian blocks and the metric blocks, in the solver's own basis. That is why
   :mod:`kuiva.amf.decouple` can be pure linear algebra: it derives ``X`` and ``R`` from the
   blocks the backend returned rather than asking a second library for them, which makes the
   consistency requirement (*the same X and R as the one-electron Hamiltonian
   used*) structural instead of a warning in a docstring.

Block conventions
-----------------
:class:`~kuiva.x2c.decouple.FourComponentBlocks`, the block container, the restricted
kinetically balanced small-component normalization it assumes, and the metric threshold every
operation on that metric must share all live in :mod:`kuiva.x2c.decouple` — the layer both
this package and the molecular one-electron path of :mod:`kuiva.interface.pyscf_bridge` build
on. They are **re-exported here** so that a backend implementation needs one import, but they
are defined there and this module must not redefine them: a second definition of the block
convention is the one way a future backend could match the letter of this contract and not
its content.

References
----------
* Four-component Dirac-Hartree-Fock: I. P. Grant, "Relativistic Quantum Theory of Atoms and
  Molecules", Springer (2007); K. G. Dyall, K. Faegri, "Introduction to Relativistic Quantum
  Chemistry", Oxford University Press (2007), ch. 7 and 11.
* Restricted kinetic balance: R. E. Stanton, S. Havriliak, J. Chem. Phys. 81, 1910 (1984),
  doi:10.1063/1.447865; K. G. Dyall, K. Faegri, Chem. Phys. Lett. 174, 25 (1990),
  doi:10.1016/0009-2614(90)85321-3.
* X2CAMF: J. Liu, L. Cheng, J. Chem. Phys. 148, 144108 (2018), doi:10.1063/1.5023750.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

try: # Protocol is 3.8+; the project targets 3.9
    from typing import Protocol, runtime_checkable
except ImportError:                      # pragma: no cover - defensive, never taken on 3.9
    Protocol = object                    # type: ignore

    def runtime_checkable(cls):          # type: ignore
        return cls

from ..util import output as out
from ..util import resources as res
from ..util.logging import get_logger
from ..x2c.decouple import (FourComponentBlocks, LIGHT_SPEED, METRIC_LINDEP_THRESHOLD,
                            blocks_memory_gb, metric_keep_mask)
from .configuration import SHELL_LETTERS, AtomicConfiguration

log = get_logger(__name__)

#: The two-electron interactions a backend may be asked for, in increasing completeness.
#: ``"coulomb"`` is the Dirac-Coulomb operator; ``"gaunt"`` adds the Gaunt (spin-other-orbit
#: and spin-spin) term; ``"breit"`` adds the full Breit interaction (Gaunt + gauge).
INTERACTIONS = ("coulomb", "gaunt", "breit")


# --- Sizing functions ---------------------------------------------------

def solution_memory_gb(nao: int) -> float:
    """Size [GB] of the arrays an :class:`AtomicDiracSolution` holds resident.

    Four block groups (core Hamiltonian, metric, density, mean field). The contraction matrix
    is ``nao * nao_target`` reals and is negligible beside them, but it is counted so that the
    figure is the size of the object rather than the size of most of it.
    """
    return 4.0 * blocks_memory_gb(nao) + res.array_gb((nao, nao), np.float64)


def dirac_scf_memory_gb(nao: int) -> float:
    """Size [GB] of the dominant arrays a four-component atomic SCF holds.

    A 4c SCF over ``nao`` scalar functions works with ``n4c = 4 * nao`` (two spinor components
    of ``2 * nao`` each), and keeps of that order: the core Hamiltonian, the metric, the
    density, the mean field, the Fock matrix, the orbital coefficients, and a DIIS subspace.
    Counted as **ten** ``(n4c, n4c)`` ``complex128`` matrices, which is a deliberate,
    documented over-estimate of a small number: it is the one figure here that is a guess
    about someone else's allocation pattern rather than a shape Kuiva controls, and the
    resource-accounting rule requires those to err high. It grows as ``nao^2``, so the pessimism costs nothing.
    """
    return 10.0 * res.array_gb((4 * nao, 4 * nao), np.complex128)


# --- The data the abstraction is built on -------------------------------------------------

@dataclass(frozen=True)
class AtomicDiracSolution:
    """A converged four-component atomic calculation: **plain arrays and metadata only**.

    This is the whole interface between a four-component solver and the rest of Kuiva. See the
    module docstring for what it forbids and why.

    Attributes
    ----------
    element : str
        Element symbol, capitalized.
    atomic_number, charge : int
        ⚠ **There is deliberately no ``spin`` field.** An earlier version carried one and it
        was always ``0``. Under average-of-configuration it could not be
        restored honestly: the reference is an *average* over a degenerate manifold and has no
        definite ``S`` — an open ``f`` shell at occupation 9/14 is not a spin state. What
        replaces it is ``configuration``, which says exactly what was averaged over.
    basis : str
        A human-readable label for the basis actually used. Provenance only.
    basis_spec : object
        Whatever the backend needs to rebuild the solver's basis, as **plain data** (for the
        PySCF backend: the parsed basis, i.e. nested lists of floats). Opaque to everything
        except the backend that produced it, and the reason
        :meth:`AtomicDiracBackend.coulomb_mean_field` can be a stateless function of a
        solution rather than requiring the solver object to still be alive. It is data, not a
        framework object, so the rule of point 1 in the module docstring is intact.
    configuration : AtomicConfiguration
        The atomic reference configuration the mean field was taken over — canonical and
        hashable, so two spellings of the same reference share a cache entry and two different
        references cannot alias. See :mod:`kuiva.amf.configuration`.
    interaction : str
        One of :data:`INTERACTIONS`.
    light_speed : float
        The speed of light [a.u.] the solution was obtained at. Recorded because the
        non-relativistic limit ``c -> inf`` is a first-class test of the subtracted operator,
        and a solution obtained at a modified ``c`` must never be mistaken for a
        physical one — nor cached alongside one.
    hcore, overlap, density, veff : FourComponentBlocks
        The one-electron Hamiltonian, the metric, the converged density and the two-electron
        mean field, in the conventions of the module docstring.
    contraction : ndarray (nao, nao_target) or None
        Maps the solver's scalar basis onto the basis the correction is wanted in:
        ``A_target = contraction^T A_solver contraction``. ``None`` when they coincide.
    uncontracted : bool
        Whether the solver's basis is the fully uncontracted (primitive) one. X2C decoupling
        is done uncontracted, so this is normally ``True``.
    mo_energy, mo_occ : ndarray
        Four-component spinor energies [Eh] and occupations, positronic branch included. Not
        needed by the correction itself — it is built from the density and the mean field —
        but they are *the* four-component reference quantity for validation: the j-splitting
        of a valence shell read straight off them is what an X2C+AMF treatment has to
        reproduce, with no external program involved.
    e_tot : float
        Total electronic + nuclear energy [Eh].
    converged : bool
    backend, backend_version : str
        Which implementation produced this, and at what version.
    nuclear_model : str
        The nuclear charge distribution the solution was obtained over (:mod:`kuiva.x2c.nuclear`).
        Recorded for the same reason ``light_speed`` is: it is part of what the solution *is*,
        and a solution over one nucleus must never be mistaken for — nor cached alongside — a
        solution over another.
    """

    element: str
    atomic_number: int
    charge: int
    basis: str
    basis_spec: object
    configuration: AtomicConfiguration
    interaction: str
    light_speed: float
    hcore: FourComponentBlocks
    overlap: FourComponentBlocks
    density: FourComponentBlocks
    veff: FourComponentBlocks
    contraction: Optional[np.ndarray]
    uncontracted: bool
    mo_energy: np.ndarray
    mo_occ: np.ndarray
    e_tot: float
    converged: bool
    backend: str
    backend_version: str
    #: Defaulted so a backend written before the model was an option still constructs, and so
    #: that what it reports — a point nucleus — is what it actually solved.
    nuclear_model: str = "point"

    def occupied_energies(self) -> np.ndarray:
        """Occupied four-component spinor energies [Eh], ascending.

        The positronic branch is excluded by the occupations, not by index arithmetic — see
        :func:`kuiva.amf.pyscf_dhf._average_of_configuration_occupation` for why that
        distinction is not academic once the metric has been projected.

        ⚠ Under average of configuration the **count is not the electron count**: an occupied
        spinor may carry a fraction, so an open ``f`` shell contributes all fourteen of its
        spinors at 9/14 rather than nine at one. Slicing this by an electron count is wrong,
        and was.
        """
        e = np.asarray(self.mo_energy, dtype=float)
        return np.sort(e[np.asarray(self.mo_occ, dtype=float) > 0])

    def shell_splitting(self, degeneracy: int = 6) -> float:
        """Spread [Eh] of the highest ``degeneracy`` occupied spinors — the four-component
        j-splitting of the valence shell, and the reference an X2C treatment must reproduce.

        For a closed p shell ``degeneracy = 6`` spans ``p_1/2`` (2) and ``p_3/2`` (4), so the
        spread *is* the spin-orbit splitting. This is a one-particle quantity read off a
        mean-field calculation; it is not a spectroscopic term splitting, and comparing it to
        one is a 30%-level statement (experiment anchors against gross error, never an accuracy claim).
        """
        e = self.occupied_energies()[-degeneracy:]
        return float(e[-1] - e[0]) if e.size else 0.0

    def __post_init__(self) -> None:
        # Coerce rather than merely check: the contract is that ``configuration`` is the
        # canonical hashable object, and enforcing it *here* means a second backend cannot
        # accidentally reintroduce free-form strings into the cache key downstream.
        if not isinstance(self.configuration, AtomicConfiguration):
            object.__setattr__(self, "configuration", AtomicConfiguration.coerce(
                self.configuration, self.element))
        naos = {b.nao for b in (self.hcore, self.overlap, self.density, self.veff)}
        if len(naos) != 1:
            raise ValueError("hcore, overlap, density and veff must be over the same basis; "
                             "got scalar dimensions {}".format(sorted(naos)))
        if self.interaction not in INTERACTIONS:
            raise ValueError("unknown two-electron interaction {!r}; expected one of {}"
                             .format(self.interaction, INTERACTIONS))
        if self.contraction is not None and self.contraction.shape[0] != self.nao:
            raise ValueError("contraction matrix {} does not map the solver basis of {} "
                             "functions".format(self.contraction.shape, self.nao))

    @property
    def nao(self) -> int:
        """Scalar basis functions in the **solver's** basis (uncontracted, normally)."""
        return self.hcore.nao

    @property
    def nao_target(self) -> int:
        """Scalar basis functions the correction will be expressed in."""
        return self.nao if self.contraction is None else int(self.contraction.shape[1])

    @property
    def n_electrons(self) -> int:
        return self.atomic_number - self.charge

    def contract(self, a: np.ndarray) -> np.ndarray:
        """Express a ``(2*nao, 2*nao)`` two-component operator in the target basis.

        The contraction acts on the scalar basis and is spin-free, so it is applied
        block-diagonally over the ``[alpha; beta]`` layout. Returns ``a`` unchanged when the
        solver already worked in the target basis.
        """
        if self.contraction is None:
            return np.ascontiguousarray(a, dtype=np.complex128)
        c = np.asarray(self.contraction)
        n, m = c.shape
        if a.shape != (2 * n, 2 * n):
            raise ValueError("expected a ({0}, {0}) two-component operator, got {1}".format(
                2 * n, a.shape))
        big = np.zeros((2 * n, 2 * m), dtype=c.dtype)
        big[:n, :m], big[n:, m:] = c, c
        return np.ascontiguousarray(big.T @ a @ big)

    def report(self, logger=None) -> None:
        """Log the standard atomic-solution block (the standard output grammar)."""
        logger = logger or log
        out.entries(logger, [
            ("element", "{} (Z = {}, charge {:+d})".format(
                self.element, self.atomic_number, self.charge)),
            ("reference configuration", "{} ({})".format(self.configuration.label,
                                                         self.configuration.canonical)),
            ("open shells", ", ".join(
                "{}^{}".format(SHELL_LETTERS[l], q)
                for l, q in self.configuration.open_shells()) or "none (closed shell)",
             "", "occupied fractionally (average of configuration)"
             if not self.configuration.is_closed_shell else ""),
            ("two-electron interaction", self.interaction),
            ("solver basis functions", self.nao,
             "", "uncontracted" if self.uncontracted else "as given"),
            ("target basis functions", self.nao_target),
            ("four-component dimension", 4 * self.nao),
            ("4c total energy", self.e_tot, "Eh", "", out.E_FMT),
            ("4c SCF converged", self.converged),
            ("backend", "{} {}".format(self.backend, self.backend_version)),
        ])
        if self.light_speed != LIGHT_SPEED:
            out.entry(logger, "speed of light", self.light_speed, "a.u.",
                      note="NOT physical; non-relativistic-limit test", fmt="{:.6e}")

    def __repr__(self) -> str:
        return ("AtomicDiracSolution({} q{:+d}, {}, {}, nao={}->{}, E={:.8f} Eh, conv={}, "
                "backend={})".format(self.element, self.charge, self.basis, self.interaction,
                                     self.nao, self.nao_target, self.e_tot, self.converged,
                                     self.backend))


# --- The protocol -------------------------------------------------------------------------

@runtime_checkable
class AtomicDiracBackend(Protocol):
    """What a four-component atomic solver must provide.

    Two methods, because the atomic mean-field correction needs two things from the same
    basis: the four-component solution, and the **non-relativistic** two-electron mean field
    of a given two-component density in that same basis — the term the correction subtracts.
    The second cannot be derived from the first — it needs the two-electron integrals — and it
    must not be computed from a separately built basis, or the subtraction is between two
    slightly different things.
    """

    #: Registry name, e.g. ``"pyscf"``.
    name: str

    @property
    def version(self) -> str:
        """Version of the underlying implementation, for provenance."""
        ...

    def solve(self, element: str, basis: object, *, charge: int = 0,
              configuration: Optional[str] = None, interaction: str = "coulomb",
              uncontract: bool = True, light_speed: Optional[float] = None,
              conv_tol: float = 1e-11, max_cycle: int = 100,
              spherical: bool = True,
              nuclear_model: str = "point") -> AtomicDiracSolution:
        """Converge a four-component atomic calculation and return it as plain arrays.

        ``spherical`` asks the backend to constrain the solution to spherically symmetric
        densities, which an atom's is. It is **on by default and every implementation is
        expected to honour it**: for an open shell the symmetric solution is an unstable fixed
        point of the SCF, so a backend that merely occupies the frontier shell fractionally
        will drift into a symmetry-broken solution and return a mean field with an arbitrary
        spatial orientation in it.

        ``nuclear_model`` is the nuclear charge distribution to solve over (``"point"``, the
        default, or ``"gaussian"`` — :mod:`kuiva.x2c.nuclear`). ⚠ **An implementation that
        cannot honour it must raise rather than fall back to a point nucleus**: the correction
        this solution produces is added to a molecular Hamiltonian built over the model the
        caller asked for, and the difference between two nuclei is a Hermitian, plausible,
        wrong contribution concentrated exactly where the correction matters. The model is
        recorded on the solution for the same reason ``light_speed`` is.
        """
        ...

    def coulomb_mean_field(self, solution: AtomicDiracSolution,
                           dm: np.ndarray) -> np.ndarray:
        """Non-relativistic Coulomb (``J - K``) mean field of a two-component density.

        ``dm`` is ``(2*nao, 2*nao)`` in the **solver's** basis and the spin-blocked row layout; the
        result is in the same basis and blocking. This is the *untransformed* operator the
        molecular Hamiltonian already carries, evaluated over the atomic
        two-component density — not a relativistic mean field.
        """
        ...


# --- The registry -------------------------------------------------------------------------

_BACKENDS: Dict[str, Callable[[], AtomicDiracBackend]] = {}


def register_backend(name: str, factory: Callable[[], AtomicDiracBackend]) -> None:
    """Register a backend under ``name``. ``factory`` is called once, lazily, per lookup.

    Lazy because a backend's import may be expensive or may fail on a machine where its
    dependency is absent, and a registry lookup for a *different* backend must not be affected
    by that.
    """
    key = name.lower()
    if key in _BACKENDS:
        log.debug("replacing already-registered atomic 4c backend %r", key)
    _BACKENDS[key] = factory


def unregister_backend(name: str) -> None:
    """Remove a backend (tests)."""
    _BACKENDS.pop(name.lower(), None)


def available_backends() -> Tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def get_backend(name: str = "pyscf") -> AtomicDiracBackend:
    """Instantiate a registered backend by name."""
    key = name.lower()
    if key not in _BACKENDS:
        raise ValueError("unknown atomic four-component backend {!r}; registered: {}".format(
            name, ", ".join(available_backends()) or "(none)"))
    return _BACKENDS[key]()


def _register_default_backends() -> None:
    """Register the backends that ship with Kuiva.

    ``"pyscf"`` is the only implementation today. ``"kuiva"`` is deliberately **not**
    registered as a stub: a name that resolves to something non-functional is worse than a
    name that does not resolve, because the error arrives later and further from the cause.
    It is reserved by convention for future native four-component work.
    """
    def _pyscf() -> AtomicDiracBackend:
        from .pyscf_dhf import PySCFDiracBackend
        return PySCFDiracBackend()

    register_backend("pyscf", _pyscf)


_register_default_backends()


__all__ = ["INTERACTIONS", "LIGHT_SPEED", "METRIC_LINDEP_THRESHOLD",
           "AtomicDiracBackend", "AtomicDiracSolution",
           "FourComponentBlocks", "available_backends", "get_backend", "register_backend",
           "unregister_backend", "blocks_memory_gb", "metric_keep_mask",
           "solution_memory_gb", "dirac_scf_memory_gb"]
