"""Fragment localization: which centre an orbital belongs to, as orbitals rather than words.

Why this exists
---------------
A polynuclear active space can be *selected* by orbital character — "the ten lowest spinors
of d character on the two titaniums" — and that is a reproducible physical statement. What it
cannot say is **which of the two titaniums** any one of those ten belongs to, and for two
equivalent centres the canonical orbitals answer "both": the SCF returns the symmetric and
antisymmetric combinations, each half on each metal. Everything that needs a *site* rather
than a set therefore needs a rotation first:

* a **broken-symmetry starting guess** has to put up-spin on one centre and down-spin on the
  other, which is not expressible in delocalized orbitals at all (the honest way to say "this
  centre" is a localized orbital set, not an atom-blocked density);
* a **multi-centre pseudospin export** needs each site in its own particle-number sector,
  i.e. a site-blocked mode ordering;
* a tensor network wants site-contiguous modes for the same reason.

⚠ **The rotation is inside the active space and therefore changes nothing.** A CASCI/CASSCF
energy is exactly invariant under a unitary mixing of active orbitals, so localizing is free
of physics — which is what makes it safe to do, and what the tests assert to machine
precision rather than to a tolerance. It changes the *labels*, and the labels are the point.

The method
----------
**Sequential fragment projection (SPADE).** For each site in turn, the Löwdin-orthogonalized
coefficients of the orbitals still unassigned are projected onto that site's AO rows and the
resulting Gram matrix is diagonalized; the eigenvectors of largest eigenvalue are the orbitals
that live on that site, and the decomposition continues on the orthogonal complement. The
last site takes the remainder, so the partition is an exact cover of the space by construction.

⚠ **Why this rather than Pipek-Mezey, which is the literature default.** Three reasons, and
they are recorded here because the choice is a departure worth being able to re-open:

1. it is **exact and non-iterative** — one eigendecomposition per site, no sweep, no
   convergence criterion, no maximum that a starting guess can miss;
2. it is **complex-safe unchanged**, which matters because the orbitals here are
   two-component spinors: Pipek-Mezey's Jacobi sweep needs a complex generalization whose
   convergence would then need validating on exactly the systems this is meant to make
   possible;
3. it produces the **site-blocked ordering with a stated count per site**, which is what both
   consumers above actually ask for. Pipek-Mezey localizes and leaves the assignment to be
   made afterwards, from populations, with no guarantee that the counts come out as asked.

Pipek-Mezey remains a reasonable second method for someone who wants the classical orbitals;
nothing here is in its way.

⚠ **A localization that did not localize is a silent failure**, so it is refused rather than
reported: every orbital's population on the site it was assigned to is measured, and a set
whose weakest orbital falls below ``min_population`` raises with the whole table in the
message. A broken-symmetry guess built from half-delocalized orbitals is a symmetric guess
wearing a fragment label, and it converges straight back to the closed-shell solution.

References
----------
The fragment-projection construction is the SPADE partition of D. Claudino, N. J. Mayhall,
"Automatic Partition of Orbital Spaces Based on Singular Value Decomposition in the Context
of Embedding Theories", *J. Chem. Theory Comput.* **15**, 1053-1064 (2019),
doi:10.1021/acs.jctc.8b01112 — there for one fragment and its environment, here applied
sequentially so that several sites partition one active space. The populations it is built on
are P.-O. Löwdin, "On the Non-Orthogonality Problem Connected with the Use of Atomic Wave
Functions in the Theory of Molecules and Crystals", *J. Chem. Phys.* **18**, 365 (1950),
doi:10.1063/1.1747632. The classical alternative named above is J. Pipek, P. G. Mezey,
"A fast intrinsic localization procedure applicable for ab initio and semiempirical linear
combination of atomic orbital wave functions", *J. Chem. Phys.* **90**, 4916 (1989),
doi:10.1063/1.456588.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..orth.canonical import sqrt_overlap
from ..util import output as out
from ..util.logging import get_logger

log = get_logger(__name__)

#: Fraction of an orbital's Löwdin population that must sit on the site it was assigned to.
#: ⚠ **Not a quality target — a definition.** An orbital that is less than half on "its" site
#: is not that site's orbital, and every consumer of this module treats it as one: a
#: broken-symmetry guess would put its spin flip on an orbital shared with the other centre,
#: and a site-blocked mode ordering would be a fiction. 0.5 is where the statement stops being
#: true, not where it stops being good; a covalently bridged pair that genuinely cannot reach
#: it is a case for stating a lower value deliberately, with the populations in front of you.
DEFAULT_SITE_POPULATION_MIN = 0.5


@dataclass(frozen=True)
class FragmentLocalization:
    """Localized orbitals and the site partition they define.

    ``coeff`` is the **whole** coefficient matrix with the localized columns written back in
    place, so it can be handed to a CASSCF as ``coeff=`` unchanged; ``columns`` are the
    columns that were rotated, in site-blocked order, and ``site[k]`` names the site of
    ``columns[k]``.

    ⚠ **The localized set is not Kramers paired.** A localizing rotation mixes the members of
    a pair like any other active-active rotation, exactly as a converged general-complex
    CASSCF's active orbitals are not pair-aligned either. The span of each *site* is
    time-reversal closed (the site projector is spin-free and real, so it commutes with
    ``T``), so the pairing is repairable per site and
    :func:`kuiva.interface.api.localize_active_space` does that; this kernel does not,
    because it is also used on scalar orbitals where the question does not arise.
    """

    coeff: np.ndarray
    columns: np.ndarray
    site: np.ndarray
    populations: np.ndarray
    site_labels: Tuple[str, ...]
    #: The unitary that was applied, ``(n_loc, n_loc)``. Kept because the rotation is a
    #: **right** multiplication and therefore basis-independent: a caller holding the same
    #: orbitals in the orthonormal working basis applies it to those instead, which is how
    #: the Kramers pairing gets repaired without this module leaving the AO basis.
    rotation: Optional[np.ndarray] = None
    method: str = "projection"

    @property
    def n_sites(self) -> int:
        return len(self.site_labels)

    @property
    def weakest(self) -> float:
        """Smallest own-site population over the localized orbitals — the quality figure."""
        if self.populations.size == 0:
            return 1.0
        own = self.populations[np.arange(self.site.size), self.site]
        return float(np.min(own))

    def site_columns(self, site: int) -> np.ndarray:
        """The columns of :attr:`coeff` that ended up on ``site``."""
        return np.asarray(self.columns)[np.asarray(self.site) == int(site)]

    def report(self, logger=None) -> None:
        logger = logger or log
        out.entry(logger, "localization", self.method,
                  note="rotation inside the localized set; no energy changes")
        for i, label in enumerate(self.site_labels):
            cols = self.site_columns(i)
            own = self.populations[np.asarray(self.site) == i, i]
            out.entry(logger, "  site {}".format(label), len(cols), "orbitals",
                      note="population {:.3f}-{:.3f} on this site".format(
                          float(np.min(own)) if own.size else float("nan"),
                          float(np.max(own)) if own.size else float("nan")))


def _site_rows(layout, sites, n_rows: int) -> List[np.ndarray]:
    """Row indices of each site's AO functions, in whichever row layout ``coeff`` has.

    ⚠ **A spinor's rows are spin-blocked, so an atom owns TWO row ranges** — the same trap
    the decoupling layer's ``spin_blocked_indices`` exists for. Getting this wrong halves
    every population and localizes on the alpha block alone, which looks like a merely
    disappointing localization rather than a bug.
    """
    from .casci import _atom_indices

    nao = int(len(layout.ao_atom))
    if n_rows == nao:
        blocks = (0,)
    elif n_rows == 2 * nao:
        blocks = (0, nao)
    else:
        raise ValueError("coefficients with {} rows match neither the scalar AO basis ({}) "
                         "nor the spin-blocked spinor basis ({})"
                         .format(n_rows, nao, 2 * nao))
    rows = []
    for spec in sites:
        atoms = _atom_indices(layout, spec)
        ao = np.concatenate([np.asarray(layout.atom_indices(int(a))) for a in atoms])
        rows.append(np.concatenate([ao + shift for shift in blocks]))
    return rows


def _site_label(layout, spec) -> str:
    from .casci import _atom_indices

    return "+".join(layout.atom_label(int(a)) for a in _atom_indices(layout, spec))


def _lowdin_columns(coeff: np.ndarray, s_ao: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """``S^{1/2} C`` over the selected columns, in the row layout ``coeff`` came in."""
    root = sqrt_overlap(np.asarray(s_ao))
    nao = root.shape[0]
    block = np.asarray(coeff)[:, columns]
    if block.shape[0] == nao:
        return root @ block
    return np.vstack([root @ block[:nao], root @ block[nao:]])


def fragment_populations(coeff: np.ndarray, s_ao: np.ndarray, layout, sites,
                         columns: Optional[Sequence[int]] = None) -> np.ndarray:
    """Löwdin population of each column on each site: ``(n_columns, n_sites)``.

    Rows sum to at most one — the remainder is the population on atoms no site claims, which
    is information rather than an error (a bridging ligand is exactly that).
    """
    coeff = np.asarray(coeff)
    cols = (np.arange(coeff.shape[1]) if columns is None
            else np.asarray(columns, dtype=int))
    t = _lowdin_columns(coeff, s_ao, cols)
    rows = _site_rows(layout, sites, coeff.shape[0])
    pops = np.empty((cols.size, len(rows)), dtype=float)
    for i, idx in enumerate(rows):
        pops[:, i] = np.sum(np.abs(t[idx]) ** 2, axis=0)
    return pops


def _counts_for(n_orb: int, sites, counts) -> List[int]:
    if counts is None:
        share, leftover = divmod(n_orb, len(sites))
        if leftover:
            raise ValueError(
                "{} orbitals do not divide evenly over {} sites; state the per-site counts "
                "(counts=[...]) — an uneven partition is a physical statement and this "
                "cannot guess it".format(n_orb, len(sites)))
        return [share] * len(sites)
    counts = [int(c) for c in counts]
    if len(counts) != len(sites):
        raise ValueError("counts has {} entries for {} sites".format(len(counts), len(sites)))
    if any(c < 1 for c in counts):
        raise ValueError("every site takes at least one orbital; got {!r}".format(counts))
    if sum(counts) != n_orb:
        raise ValueError("the per-site counts sum to {} but {} orbitals were given to "
                         "localize".format(sum(counts), n_orb))
    return counts


def localize(coeff: np.ndarray, s_ao: np.ndarray, layout, columns: Sequence[int], sites, *,
             counts: Optional[Sequence[int]] = None,
             min_population: float = DEFAULT_SITE_POPULATION_MIN,
             method: str = "projection") -> FragmentLocalization:
    """Rotate ``columns`` of ``coeff`` into orbitals that each belong to one site.

    Parameters
    ----------
    coeff : ndarray
        ``(nao, n)`` scalar MOs or ``(2*nao, n)`` spinors in the AO basis (spin-blocked rows).
        Returned with the localized columns written back in place.
    s_ao : ndarray ``(nao, nao)``
        The scalar AO overlap.
    layout : :class:`kuiva.basis.layout.AOLayout`
    columns : sequence of int
        Which columns to localize — normally an active space, but any set the caller owns.
    sites : sequence
        One entry per site, each an atom index, an element symbol unique in the molecule, or
        a sequence of either (a site may be a whole fragment: ``[["Fe", 2, 3], ...]``). The
        addressing is :func:`kuiva.mcscf.casci.active_space_by_character`'s, so "which atom"
        means the same thing everywhere.
    counts : sequence of int, optional
        Orbitals per site. Default: an equal split, refused when it does not divide.
    min_population : float
        See :data:`DEFAULT_SITE_POPULATION_MIN`. The refusal carries the whole table.
    method : str
        ``"projection"`` — sequential fragment projection (SPADE), the only method
        implemented; see the module docstring for what it is and why it rather than
        Pipek-Mezey.

    Returns
    -------
    :class:`FragmentLocalization`

    Raises
    ------
    ValueError
        If the counts do not partition the set, if a site claims no AO functions, or if the
        localization does not reach ``min_population`` — with the populations printed.
    """
    if method != "projection":
        raise ValueError("method must be \"projection\" (sequential fragment projection); "
                         "got {!r}. Pipek-Mezey is not implemented — see the module "
                         "docstring for why the projection is the one that is."
                         .format(method))
    coeff = np.asarray(coeff)
    columns = np.asarray(columns, dtype=int)
    sites = list(sites)
    if len(sites) < 2:
        raise ValueError("localization partitions orbitals over at least two sites; one site "
                         "is the space itself and needs no rotation")
    per_site = _counts_for(columns.size, sites, counts)
    rows = _site_rows(layout, sites, coeff.shape[0])
    for spec, idx in zip(sites, rows):
        if idx.size == 0:
            raise ValueError("site {!r} carries no basis functions in this molecule"
                             .format(spec))

    t = _lowdin_columns(coeff, s_ao, columns)
    n_orb = columns.size
    dtype = np.complex128 if np.iscomplexobj(t) else np.float64
    residual = np.eye(n_orb, dtype=dtype)     # basis of the not-yet-assigned subspace
    blocks: List[np.ndarray] = []
    for i, (idx, take) in enumerate(zip(rows, per_site)):
        if i == len(sites) - 1:
            # ⚠ The remainder, whole: the partition is an exact cover by construction rather
            # than by a threshold, so no orbital can be claimed twice or dropped.
            block = residual
            if block.shape[1] != take:
                raise AssertionError("the last site was left {} orbitals but asked for {}"
                                     .format(block.shape[1], take))
        else:
            ta = t[idx] @ residual                       # the site's part of what is left
            gram = ta.conj().T @ ta
            vals, vecs = np.linalg.eigh(0.5 * (gram + gram.conj().T))
            order = np.argsort(vals)[::-1]               # most localized first
            block = residual @ vecs[:, order[:take]]
            residual = residual @ vecs[:, order[take:]]
        blocks.append(block)
    u = np.hstack(blocks)
    unitarity = float(np.max(np.abs(u.conj().T @ u - np.eye(n_orb))))
    if unitarity > 1e-10:
        raise AssertionError("the localizing rotation is not unitary (residual {:.2e}); the "
                             "deflation lost orthogonality".format(unitarity))

    localized = np.ascontiguousarray(coeff.copy())
    localized[:, columns] = np.asarray(coeff)[:, columns] @ u
    site_of = np.concatenate([np.full(take, i, dtype=int)
                              for i, take in enumerate(per_site)])
    pops = fragment_populations(localized, s_ao, layout, sites, columns)
    labels = tuple(_site_label(layout, spec) for spec in sites)
    result = FragmentLocalization(coeff=localized, columns=columns, site=site_of,
                                  populations=pops, site_labels=labels, rotation=u,
                                  method=method)
    own = pops[np.arange(n_orb), site_of]
    if float(np.min(own)) < float(min_population):
        rows_text = "\n".join(
            "    {:>4d}  site {:<12s} " .format(int(columns[k]), labels[site_of[k]])
            + "  ".join("{:6.3f}".format(p) for p in pops[k])
            for k in range(n_orb))
        raise ValueError(
            "localization did not separate the sites: the weakest orbital carries only "
            "{:.3f} of its population on the site it was assigned to, against a "
            "min_population of {:.2f}.\n"
            "  orbital  assigned to    populations on [{}]\n{}\n"
            "  A set this delocalized is not a site partition, and a broken-symmetry guess "
            "or a site-blocked ordering built from it would be one in name only. Either the "
            "sites are genuinely covalently shared — state a lower min_population "
            "deliberately — or the selected orbitals are not the magnetic ones."
            .format(float(np.min(own)), float(min_population), ", ".join(labels), rows_text))
    log.debug("localized %d orbitals over %d sites; weakest own-site population %.3f",
              n_orb, len(sites), result.weakest)
    return result


__all__ = ["DEFAULT_SITE_POPULATION_MIN", "FragmentLocalization", "fragment_populations",
           "localize"]
