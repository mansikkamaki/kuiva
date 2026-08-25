"""Per-element driver and cache for the atomic mean-field correction.

Sequences the three things a correction needs — the four-component solve, the X2C decoupling,
and the subtracted non-relativistic mean field — and caches the result.

Why caching is worth having here, and why it is sound
-----------------------------------------------------
The correction depends only on ``(element, basis, charge, configuration, interaction,
backend, speed of light)``. It does **not** depend on the molecular geometry, on the other
atoms, or on anything the SCF does: it is a property of an isolated atom in its own basis. So
one solve serves every geometry of a potential-energy surface, every atom of the same element
in a molecule, and every test in the suite that touches that element.

Three consequences that the key has to respect, and all three are traps:

* ⚠ **The speed of light is part of the key.** The ``c -> inf`` test runs the
  same element in the same basis at a different ``c``, and a cache that ignored it would
  return the physical correction and the test would pass for the wrong reason — the worst
  possible failure of a test whose entire job is to be unfoolable.
* ⚠ **The basis is keyed by its content, not its name.** Two ``Mole`` objects can name the
  same family and carry different parsed data (a per-atom override, a decontraction, a
  registry that resolved a name through Basis Set Exchange at a different time). Keying on the
  name would silently reuse a correction built for different functions. The parsed basis is
  hashed instead — a few hundred floats per element, once.
* ⚠ **The reference configuration is canonicalized, not stored as written.** For an open shell
  the configuration is a real choice that changes the correction, and the sensitivity study of
  neutral-vs-ionic references *depends* on the two not aliasing.
  Equally, ``"[Ar]3d1"`` and ``"1s2 2s2 2p6 3s2 3p6 3d1"`` are the same reference and must
  share one entry. A free-form string gets both of those wrong; the canonical per-``l``
  occupation vector of :class:`~kuiva.amf.configuration.AtomicConfiguration` gets both right,
  and it is hashable by construction.

**Two lifetimes, and they hold different things.** The **four-component solutions** live in a
module-level dictionary and die with the process: they are large (258 MB for a decontracted
lanthanide) and are an intermediate. The **corrections** are additionally written to disk by
:mod:`kuiva.amf.cache`, keyed by :func:`cache_key`, because they are small, they are what
everything downstream wants, and they are the reason the in-process cache was not enough — a
potential-energy surface run as N separate jobs used to pay a lanthanide's ~35-minute atomic
solve N times over, for a quantity that has one value. See that module for the three ways a
persistent cache can silently return a wrong answer and what closes each; the one worth naming
here is that :data:`kuiva.amf.cache.FORMULA_VERSION` **must be bumped whenever the numerical
content of an** :class:`~kuiva.amf.decouple.AtomicAMF` **changes for an unchanged request**.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..util import output as out
from ..util.logging import get_logger
from ..util.timing import timer
from ..x2c.nuclear import pyscf_nucmod, resolve_nuclear_model
from . import cache as disk_cache
from .backend import AtomicDiracSolution, get_backend
from .configuration import AtomicConfiguration
from .decouple import AtomicAMF, amf_atomic_correction

log = get_logger(__name__)


@dataclass(frozen=True)
class AtomicRequest:
    """Everything that determines an atomic correction — and therefore the cache key.

    Frozen and hashable. ``basis_digest`` is a hash of the *parsed* basis rather than its
    name; see the module docstring for why that distinction is not pedantic.
    """

    element: str
    basis_digest: str
    charge: int
    configuration: AtomicConfiguration
    interaction: str
    backend: str
    light_speed: Optional[float]
    uncontract: bool
    #: ⚠ **Part of the key, because it is part of the physics.** A mean field solved over a
    #: point nucleus and one solved over a finite nucleus are different operators of the same
    #: shape, and the difference is largest exactly where this correction matters most — so
    #: two requests differing only in this must not share an entry. The default matches the
    #: program's, so a request built without the argument says what it means.
    #: (Entries written before the field existed do not need it: the same change moved
    #: :data:`kuiva.amf.cache.FORMULA_VERSION`, which makes all of them inert.)
    nuclear_model: str = "point"

    def __str__(self) -> str:
        c = "" if self.light_speed is None else ", c={:.4e}".format(self.light_speed)
        n = "" if self.nuclear_model == "point" else ", {} nucleus".format(self.nuclear_model)
        return "{}{:+d} [{}] {} {} via {}{}{}".format(
            self.element, self.charge, self.basis_digest[:8],
            self.configuration.canonical, self.interaction, self.backend, c, n)


def basis_digest(basis: object) -> str:
    """A stable hash of a parsed basis specification.

    ``json`` with ``sort_keys`` gives a canonical text form for the nested lists of floats
    PySCF parses a basis into, and for a plain family name. Floats are serialized with
    ``repr`` semantics, so two bases that differ in the last bit hash differently — which is
    the intended behaviour: they *are* different bases.
    """
    try:
        text = json.dumps(basis, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        text = repr(basis)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_request(element: str, basis: object, *, charge: int = 0,
                 configuration=None, interaction: str = "coulomb",
                 backend: str = "pyscf", light_speed: Optional[float] = None,
                 uncontract: bool = True,
                 nuclear_model: Optional[str] = None) -> AtomicRequest:
    """Build the cache key for an atomic correction.

    ⚠ ``configuration`` is **canonicalized here**, not stored as given. Two spellings of the
    same reference state (``"[Ar]3d1"`` and ``"1s2 2s2 2p6 3s2 3p6 3d1"``) must share a cache
    entry, and two different references must not alias — neither of which a free-form string
    can guarantee. See :mod:`kuiva.amf.configuration`.

    ⚠ ``nuclear_model`` is **normalized here**, for the same reason: ``"gauss"`` and
    ``"gaussian"`` are one model and must share an entry, and an unknown spelling is refused
    rather than defaulted, because a typo that silently selects a point nucleus caches a
    result under a key that does not describe it.

    ⚠ ``charge`` is likewise **derived from the configuration**, not taken as given, because
    that is what the backend actually solves. Keying on the requested charge instead would let
    two requests that produce the identical calculation occupy two cache entries — and, worse,
    would key on a number that does not describe the stored result.
    """
    from pyscf import gto

    element = element.capitalize()
    config = AtomicConfiguration.coerce(configuration, element)
    return AtomicRequest(element=element, basis_digest=basis_digest(basis),
                         charge=int(gto.charge(element)) - config.n_electrons,
                         configuration=config,
                         interaction=interaction, backend=backend,
                         light_speed=None if light_speed is None else float(light_speed),
                         uncontract=bool(uncontract),
                         nuclear_model=resolve_nuclear_model(nuclear_model))


# The two caches. Solutions are kept as well as corrections because a solution is the
# expensive part and is what a future consumer (a different interaction, a diagnostic, a
# term-by-term comparison against another implementation) would want to reuse.
_SOLUTIONS: Dict[AtomicRequest, AtomicDiracSolution] = {}
_CORRECTIONS: Dict[AtomicRequest, AtomicAMF] = {}

#: Counts of what actually ran, for tests that verify caching by **call count** rather than by
#: timing — timing is not evidence on a thermally throttled machine.
#: ``disk_hits`` and ``disk_writes`` count the persistent cache of :mod:`kuiva.amf.cache`, and
#: are what makes "the second process did not solve anything" a countable statement.
_STATS = {"solves": 0, "corrections": 0, "solution_hits": 0, "correction_hits": 0,
          "disk_hits": 0, "disk_writes": 0}


def cache_statistics() -> Dict[str, int]:
    """Copy of the solve/hit counters. Reset by :func:`clear_cache`."""
    return dict(_STATS)


def clear_cache() -> None:
    """Forget every **in-process** cached atomic solution and correction, reset the counters.

    ⚠ It does not touch the persistent cache of :mod:`kuiva.amf.cache`, and must not: this is
    called by generators and tests that want a cold *in-process* start, and silently deleting
    a user's accumulated lanthanide corrections would be a spectacular overreach for a function
    with this name. :func:`kuiva.amf.cache.purge` is the one that removes files, and tests that
    need a cold disk point ``$KUIVA_AMF_CACHE`` at a temporary directory instead.
    """
    _SOLUTIONS.clear()
    _CORRECTIONS.clear()
    for k in _STATS:
        _STATS[k] = 0


def cache_key(request: AtomicRequest) -> str:
    """A filesystem-safe identifier for a request — the natural name for an on-disk cache
    entry, should one ever be wanted (see the module docstring).

    ⚠ **This list is written out by hand and a field added to** :class:`AtomicRequest`
    **does not join it on its own.** Two requests that differ only in a forgotten field then
    name one file; the stored-attribute check in :func:`kuiva.amf.cache.load` catches it and
    warns, so nothing wrong is ever *served* — but the entry is rewritten on every run and
    the cache silently stops working. Add the field here in the same change.
    """
    parts = (request.element, request.basis_digest[:16], str(request.charge),
             request.configuration.canonical.replace(" ", ""), request.interaction,
             request.backend,
             "c{:.6e}".format(request.light_speed) if request.light_speed else "c-physical",
             "unc" if request.uncontract else "con",
             "nuc-{}".format(request.nuclear_model))
    return "-".join(parts)


def atomic_solution(element: str, basis: object, *, charge: int = 0,
                    configuration: Optional[str] = None, interaction: str = "coulomb",
                    backend: str = "pyscf", light_speed: Optional[float] = None,
                    uncontract: bool = True, nuclear_model: Optional[str] = None,
                    **solver_kwargs) -> AtomicDiracSolution:
    """A converged four-component atomic solution, from the cache if it is there.

    ``light_speed`` is applied for the whole of the solve (see
    :func:`kuiva.amf.pyscf_dhf.light_speed` — it is a PySCF process-global, not an argument).

    ``nuclear_model`` (``"point"``, the default, or ``"gaussian"``) is the charge distribution
    the nucleus is solved over. ⚠ It must be the **same** model the molecule this solution
    will correct was built with; :func:`kuiva.amf.correction.amf_correction` is where that is
    checked rather than assumed.
    """
    request = make_request(element, basis, charge=charge, configuration=configuration,
                           interaction=interaction, backend=backend,
                           light_speed=light_speed, uncontract=uncontract,
                           nuclear_model=nuclear_model)
    cached = _SOLUTIONS.get(request)
    if cached is not None:
        _STATS["solution_hits"] += 1
        log.debug("atomic 4c solution for %s served from cache", request)
        return cached

    impl = get_backend(backend)
    with _light_speed_for(backend, light_speed):
        # The *canonical* configuration is passed on, not the caller's spelling of it, so the
        # solution is provably the one this key describes.
        solution = impl.solve(element, basis, charge=charge,
                              configuration=request.configuration,
                              interaction=interaction, uncontract=uncontract,
                              nuclear_model=request.nuclear_model,
                              **solver_kwargs)
    _SOLUTIONS[request] = solution
    _STATS["solves"] += 1
    return solution


def atomic_correction(element: str, basis: object, *, charge: int = 0,
                      configuration: Optional[str] = None, interaction: str = "coulomb",
                      backend: str = "pyscf", light_speed: Optional[float] = None,
                      uncontract: bool = True, nuclear_model: Optional[str] = None,
                      report: bool = False, **solver_kwargs) -> AtomicAMF:
    """The atomic mean-field correction ``(delta h_sf, delta w)`` for one element.

    Everything from the four-component solve to the subtraction happens inside a **single**
    speed-of-light scope. That is not tidiness: the transformed mean field and the subtracted
    one must be computed at the same ``c``, or their difference is meaningless instead of
    zero, and PySCF's ``c`` is a process-global that a per-call argument cannot express.
    """
    request = make_request(element, basis, charge=charge, configuration=configuration,
                           interaction=interaction, backend=backend,
                           light_speed=light_speed, uncontract=uncontract,
                           nuclear_model=nuclear_model)
    if request.configuration.n_electrons <= 1:
        # ⚠ Hydrogen, and every one-electron reference. There is no second electron to screen,
        # so there is no two-electron mean field and no picture change of one: the correction
        # is exactly zero **by definition**, the same statement
        # :func:`kuiva.amf.correction.amf_correction` makes for a one-electron *molecule*, made
        # here per element because a molecule containing hydrogen reaches this function and
        # not that branch. Computing it instead would picture-change a Hartree-Fock
        # self-interaction — an artefact of the method, not physical screening.
        #
        # Deliberately **not** cached: it costs nothing to rebuild and caching it would put an
        # entry in the statistics that no solve corresponds to, which is exactly what the
        # unique-element caching test reads.
        return _zero_correction(element, basis, request.configuration,
                                request.nuclear_model)
    cached = _CORRECTIONS.get(request)
    if cached is not None:
        _STATS["correction_hits"] += 1
        log.debug("atomic mean-field correction for %s served from cache", request)
        if report:
            _report(request, cached)
        return cached

    # The persistent cache, before the four-component solve rather than after it: the whole
    # point is to avoid the solve, so a disk hit must short-circuit it. A miss for any reason
    # returns None and costs one stat() (kuiva.amf.cache).
    disk_cache.announce()
    key = cache_key(request)
    cached = disk_cache.load(request, key)
    if cached is not None:
        _STATS["disk_hits"] += 1
        _CORRECTIONS[request] = cached
        log.debug("atomic mean-field correction for %s served from the persistent cache",
                  request)
        if report:
            _report(request, cached)
        return cached

    impl = get_backend(backend)
    with _light_speed_for(backend, light_speed):
        solution = _SOLUTIONS.get(request)
        if solution is None:
            with timer("atomic 4c solve"):
                solution = impl.solve(element, basis, charge=charge,
                                      configuration=request.configuration,
                                      interaction=interaction,
                                      uncontract=uncontract,
                                      nuclear_model=request.nuclear_model,
                                      **solver_kwargs)
            _SOLUTIONS[request] = solution
            _STATS["solves"] += 1
        else:
            _STATS["solution_hits"] += 1

        with timer("atomic mean-field decoupling"):
            correction = amf_atomic_correction(
                solution, lambda dm: impl.coulomb_mean_field(solution, dm))

    _CORRECTIONS[request] = correction
    _STATS["corrections"] += 1
    # ⚠ Written only at the physical speed of light. A ``c``-scaled solve is a test artefact
    # (the ``c -> inf`` check), and although ``light_speed`` is in the key
    # and in the stored request so it *could* not alias, filling a user's cache with entries
    # nothing will ever ask for again is not what the directory is for.
    if light_speed is None and disk_cache.store(request, cache_key(request), correction):
        _STATS["disk_writes"] += 1
    if report:
        _report(request, correction, solution)
    return correction


def _zero_correction(element: str, basis: object, configuration: AtomicConfiguration,
                     nuclear_model: str = "point") -> AtomicAMF:
    """An exactly-zero :class:`~kuiva.amf.decouple.AtomicAMF` over ``basis``'s functions.

    The size is taken from a probe ``Mole`` built exactly as
    :meth:`kuiva.amf.pyscf_dhf.PySCFDiracBackend._build_mole` builds the target basis, so the
    zero block has the same shape a computed one would have had and the molecular assembly
    cannot tell the two apart. The nuclear model changes no dimension and is passed for that
    reason alone — so "built exactly as" stays a true statement rather than one with an
    exception nobody wrote down.
    """
    import numpy as np
    from pyscf import gto

    probe = gto.M(atom=[(element, (0.0, 0.0, 0.0))], basis={element: basis},
                  charge=int(gto.charge(element)) - configuration.n_electrons,
                  spin=configuration.n_electrons % 2,
                  nucmod=pyscf_nucmod(nuclear_model), verbose=0)
    nao = int(probe.nao)
    return AtomicAMF(h_sf=np.zeros((nao, nao)), w=np.zeros((3, nao, nao)),
                     configuration=configuration, scale=0.0, tr_residual=0.0,
                     tr_residual_rel=0.0, transformed_scale=0.0, subtracted_scale=0.0)


def _light_speed_for(backend: str, c: Optional[float]):
    """The backend's speed-of-light scope.

    Only the PySCF backend has one, because only PySCF keeps ``c`` in a process-global. A
    backend that takes it as an argument needs no scope, and a backend that has neither cannot
    support the ``c -> inf`` test — which is a real limitation and is refused loudly rather
    than silently producing a physical result where a non-relativistic one was asked for.
    """
    import contextlib

    if c is None:
        return contextlib.suppress()
    if backend.lower() == "pyscf":
        from .pyscf_dhf import light_speed
        return light_speed(c)
    raise NotImplementedError(
        "backend {!r} does not support overriding the speed of light, which the "
        "non-relativistic-limit test requires. Refusing rather than "
        "silently returning a correction at the physical c.".format(backend))


def _report(request: AtomicRequest, correction: AtomicAMF,
            solution: Optional[AtomicDiracSolution] = None) -> None:
    """The standard output block for one element's correction."""
    out.subsection(log, "Atomic mean-field correction: {}".format(request.element))
    if solution is not None:
        solution.report(log)
    out.entries(log, [
        ("spin-free correction, max |dh_sf|", correction.spin_free_scale, "Eh", "",
         "{:.6e}"),
        ("spin-orbit correction, max |dw|", correction.spin_orbit_scale, "Eh", "",
         "{:.6e}"),
        ("picture-changed mean field, max |G~|", correction.transformed_scale, "Eh", "",
         "{:.6e}"),
        ("subtracted Coulomb mean field, max |G|", correction.subtracted_scale, "Eh", "",
         "{:.6e}"),
        ("one-electron compensation, max |dh1e|", correction.compensation_scale, "Eh",
         "h1e(X_2e) - h1e(X_1e)", "{:.6e}"),
        ("cancellation factor", correction.cancellation, "", "|G| / |dG|", "{:.1f}"),
        ("discarded time-reversal-odd part", correction.tr_residual, "Eh",
         "{:.1e} relative".format(correction.tr_residual_rel), "{:.3e}"),
        ("persistent cache", disk_cache.describe(), "", "geometry-independent; "
         "${}".format(disk_cache.ENV_CACHE_DIR)),
    ])


def nuclear_model_of(mol) -> str:
    """Which nuclear charge model a built ``Mole`` actually uses.

    ⚠ **Read off the molecule's own atom table, never off a value someone carried alongside
    it.** The nuclear model belongs to whatever produced the integrals, exactly as the speed
    of light does (:mod:`kuiva.x2c.nuclear`): an atomic mean field solved over one nucleus and
    a molecular Hamiltonian built over another differ by a Hermitian, plausible, wrong amount
    that no output field would mention. The atom table is what the integral engine reads, so
    it is what this reads.

    ⚠ **A molecule with a mixed nuclear model is refused, not summarized.** Kuiva states the
    model once per molecule, so a mixture can only come from a ``Mole`` built elsewhere; there
    is no honest single answer to return, and every caller here is about to use the answer for
    an *atomic* calculation that has to match.
    """
    from pyscf import gto

    flags = {int(mol._atm[ia, gto.mole.NUC_MOD_OF]) for ia in range(mol.natm)}
    models = {gto.mole.NUC_GAUSS: "gaussian"}
    named = {models.get(f, "point") for f in flags}
    if len(named) > 1:
        mixed = ", ".join(
            "{} ({})".format(mol.atom_symbol(ia),
                             models.get(int(mol._atm[ia, gto.mole.NUC_MOD_OF]), "point"))
            for ia in range(mol.natm))
        raise NotImplementedError(
            "this molecule mixes nuclear charge models — {}. Kuiva states the model once for "
            "the whole molecule, and an atomic mean-field correction has to be solved over "
            "the same nucleus as the molecular integrals it corrects.".format(mixed))
    return named.pop() if named else "point"


def elements_by_label(mol) -> Dict[str, Tuple[str, object]]:
    """``{atom label: (element symbol, parsed basis)}``, one entry per **unique** label.

    The key is ``mol.atom_symbol(ia)`` — the *labelled* symbol, so ``Ti1`` and ``Ti2`` with
    different bases stay separate — and it is what the molecular assembly of
    :func:`kuiva.amf.correction.amf_correction` looks each atom up by. The value is what an
    atomic solve needs: the pure element symbol and the parsed basis.

    This is what makes the caching effective at the molecular level: a correction
    is solved once per element, not once per atom. The parsed basis is taken from
    ``mol._basis`` rather than from a family name so the atomic calculation uses exactly the
    functions the molecule does (see the module docstring).
    """
    from pyscf import gto

    seen = {}
    for ia in range(mol.natm):
        symbol = mol.atom_symbol(ia)
        if symbol in seen:
            continue
        pure = mol.atom_pure_symbol(ia)
        basis = mol._basis.get(symbol, mol._basis.get(pure))
        if basis is None:
            raise KeyError("no basis found for atom {} ({}) in the molecule".format(
                ia, symbol))
        seen[symbol] = (pure, basis)

    # ⚠ ``Mole._basis`` and the integral engine's actual shells can disagree, and it is
    # silent. ``Mole.decontract_basis()`` returns a molecule whose ``_bas``/``_env`` are
    # primitive while ``_basis`` still holds the **contracted** definition it started from —
    # measured on cc-pVDZ neon: ``x.nao`` is 26 and ``x._basis["Ne"]`` still rebuilds to 14.
    # Anything reading ``_basis`` there gets a different set of functions from the one the
    # molecule is using, and the atomic mean field would be computed over the wrong basis.
    # Rebuilding and counting is cheap and turns that into an error at the right place.
    per_element = {}
    for symbol, (pure, basis) in seen.items():
        probe = gto.M(atom=[(pure, (0.0, 0.0, 0.0))], basis={pure: basis},
                      spin=int(gto.charge(pure)) % 2, verbose=0)
        per_element[symbol] = int(probe.nao)
    total = sum(per_element[mol.atom_symbol(ia)] for ia in range(mol.natm))
    if total != int(mol.nao):
        raise ValueError(
            "the basis recorded on this molecule rebuilds to {} functions but the molecule "
            "has {}. Its stored basis specification and the shells it actually uses "
            "disagree — this is what Mole.decontract_basis() returns, for instance. The "
            "atomic mean-field correction would then be computed over different functions "
            "from the Hamiltonian it corrects.".format(total, mol.nao))
    return seen


def elements_and_bases(mol) -> Tuple[Tuple[str, object], ...]:
    """The ``(element symbol, parsed basis)`` pairs of a ``Mole``, one per unique element.

    :func:`elements_by_label` without the labels, for the callers that only need to know what
    has to be solved.
    """
    return tuple(elements_by_label(mol).values())


__all__ = ["AtomicRequest", "atomic_correction", "atomic_solution", "basis_digest",
           "cache_key", "cache_statistics", "clear_cache", "elements_and_bases",
           "elements_by_label", "make_request", "nuclear_model_of"]
