"""Loewdin population analysis for two-component spinors.

Why this exists
---------------
A spinor has no isosurface (:mod:`kuiva.spinor.density` says why), so the usual way of
checking an active space — look at the orbitals — is not directly available. Reduced
populations are the substitute that needs no picture at all: they say, in numbers, that a
given active spinor is 87% Ti 3d and 9% Cl 3p. For spinors that is not a second-best option
but often the *primary* one, and it is why this module exists alongside the molden dump
rather than instead of it.

The two-component generalization
--------------------------------
In the scalar case one symmetrically orthogonalizes the AOs (``chi~ = S^{1/2} chi``) and reads
the diagonal of the transformed density. With spinors the one-particle density matrix has
alpha-alpha, beta-beta **and off-diagonal** spin blocks, and the quantities split into two
kinds:

* **Charge** — spin trace, ``q_mu = D~^aa_mu,mu + D~^bb_mu,mu``. The off-diagonal blocks do
  not enter; the atomic charges are as well defined as in the scalar case.
* **Spin** — a **vector**, ``s_k,mu = (1/2) sum_{ss'} (sigma_k)_{ss'} D~^{s's}_mu,mu``, which
  in this module's row layout is::

      s_x = +Re D~^ab_mu,mu      s_y = -Im D~^ab_mu,mu      s_z = (1/2)(D~^aa - D~^bb)_mu,mu

  The off-diagonal spin blocks are the whole of ``s_x`` and ``s_y``, and they are exactly what
  a scalar analysis has no place to put.

⚠ **Three things about the spin density that will otherwise be read as bugs.**

1. **For any Kramers-degenerate density it vanishes identically.** A state-averaged Kramers
   pair has ``s = 0`` at every point, everywhere, exactly — time reversal maps the pair onto
   itself and reverses the spin. A zero spin population on a Dy(III) centre is therefore the
   *correct* output of a properly state-averaged calculation, not a lost magnetic moment.
   Look at a single state if the spin distribution is the question.
2. **The quantization axis is arbitrary** without a field or a pseudospin convention, so the
   individual components carry no absolute meaning while ``|s|`` does. That is why ``|s|``
   alone is the default output and the vector is opt-in.
3. **Spin is not the magnetic moment.** With spin-orbit coupling the moment is
   ``-(L + g_e S) mu_B`` and the orbital part is comparable to the spin part; ``props/dump.py``
   and :mod:`kuiva.props.multiplet` are where magnetic properties live. This module reports a
   spin population, and the number should not be quoted as a moment.

Invariance — which rows mean something
--------------------------------------
⚠ **The population of an individual spinor inside a degenerate manifold is not well defined**:
the manifold's basis is arbitrary up to a unitary mixing, and the populations change with it.
Sums over a whole degenerate block *are* invariant, because ``sum_{i in block} c_i c_i^dag``
is the block projector. This is the same discipline :mod:`kuiva.props.multiplet` imposes on
moment matrices, for the same reason, and it is why ``group="kramers"`` (sum over Kramers
pairs) is the default here rather than a per-spinor listing.

Why Loewdin rather than Mulliken
--------------------------------
Mulliken's partition assigns the overlap population half-and-half, which makes it notoriously
basis-dependent and lets a diffuse function on one atom carry population that physically sits
on its neighbour. Loewdin's symmetrically orthogonalized AOs do not overlap, so each AO's
population is its own; the symmetric orthogonalization is also the one that keeps the
orthogonalized functions as close as possible to the originals, which is what makes "Ti 3d"
still mean Ti 3d — and *that* is the quantity this module exists to produce.

⚠ **The atomic charges are a different matter, and they are withdrawn from every printed
report (measured decision, 2026-08-22).** Every population analysis is a basis-dependent
partition of a quantity that has no unique partition, and the Loewdin charge fails
*qualitatively*, not merely noisily — characterized across five systems in two basis sets
(Mulliken on the same converged density as the cross-check):

* **Wrong sign on three of five systems**: Ti in TiCl3 comes out −0.33 (SVP) / −0.26 (TZVP)
  against Mulliken's +0.91/+1.36; Ce in CeCl3 −0.16 (SVP) and **exactly 0.00** (TZVP) for a
  trivalent ion; H in HI comes out *negative* (−0.03/−0.05), the wrong side of the
  H–I electronegativity difference. TiF3 and TlH keep the right sign.
* **A better basis does not rescue it** — the sign failures survive SVP → TZVP unchanged —
  and ⚠ **the usual diffuse-function explanation does not hold**: the smallest primitive
  exponent on the metal barely moves between the two bases (it *rises* on Ti and Ce), while
  the Loewdin–Mulliken gap tracks the **ligand** (chlorides ~1.1–1.6 e, fluoride ~0.9 e,
  hydride ≤0.07 e).
* **The mechanism is measured, not conjectured** (ghost test: the metal's basis with no
  nucleus and no electrons over a pure Cl3(3-) density): Loewdin assigns **2.4 electrons of
  pure chloride density to functions merely labeled with the metal** where Mulliken assigns
  0.6 — a 1.8 e labelling difference with no metal in the system, larger than the whole
  molecular gap. The excess sits in the ligand-pointing d and p channels (not the diffuse s),
  and it scales with the overlap conditioning: the same TiCl3 gives a 0.45 e gap in STO-3G
  (smallest overlap eigenvalue 2.7e-01), 0.69 e in def2-SVP (2.4e-02) and 1.24 e — across
  the sign boundary — in x2c-SVPall-2c (1.5e-05). An all-electron relativistic basis lives
  permanently in the ill-conditioned regime where ``S^{1/2}`` is non-local, so the failure
  is structural for this code, not an unlucky molecule (cf. Mayer 2004, below).
* Kuiva's spinor implementation reproduces an independent scalar Loewdin on the same density
  to 1e-13, so all of this is the partition, not the code.

A number with the wrong sign on ionic textbook compounds misleads no matter what warning
stands next to it, so :meth:`AtomicPopulations.report` prints populations and spins only.
:meth:`AtomicPopulations.atomic_charge` remains as an accessor — the algebra is trivially
``Z − population`` and sum-rule tests legitimately probe the density through it — but it is a
diagnostic, never a chemical statement. The **reduced orbital populations are far more
robust**, because they ask which functions an orbital is built from rather than how to divide
space; they are what this module exists to produce.

.. warning::
   ``S^{1/2}`` comes from :func:`kuiva.orth.canonical.sqrt_overlap`, **not** from
   ``symmetric_orthogonalization``: the latter computes ``S^{-1/2}`` and refuses a linearly
   dependent basis, which is correct for a working basis and unnecessary here (the square root
   is well defined for a singular ``S``).

References
----------
* P.-O. Loewdin, "On the Non-Orthogonality Problem Connected with the Use of Atomic Wave
  Functions...", J. Chem. Phys. 18, 365 (1950), doi:10.1063/1.1747632; and Adv. Quantum Chem.
  5, 185 (1970) for the population analysis itself.
* Basis-set dependence of Mulliken versus Loewdin partitions: I. Mayer, Chem. Phys. Lett. 393,
  209 (2004), doi:10.1016/j.cplett.2004.06.031.
* Spin-density matrices in two-component theory: K. G. Dyall, K. Faegri, "Introduction to
  Relativistic Quantum Chemistry", Oxford University Press (2007), ch. 6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..basis.ghosts import is_ghost
from ..basis.layout import AOLayout
from ..orth.canonical import sqrt_overlap
from ..spinor.expand import spinor_indices
from ..util import output as out
from ..util import resources as res
from ..util.logging import get_logger

log = get_logger(__name__)

#: Minimum population for a contribution to appear in the reduced-AO table. 1% of one
#: electron: small enough to show real ligand admixture, large enough that a heavy-element
#: orbital does not print a hundred rows of tail.
DEFAULT_PRINT_TOLERANCE = 0.01

#: Occupations within this of each other are treated as degenerate when grouping orbitals.
DEFAULT_DEGENERACY_TOLERANCE = 1e-6


def population_memory_gb(nao: int, n_orb: int) -> float:
    """Size [GB] of the population workspace (exact sizing function): ``S^{1/2}`` plus the
    transformed coefficients plus the per-AO populations.

    ``nao^2`` real for the root, ``2*nao*n_orb`` complex for ``S^{1/2} C``, ``nao*n_orb`` real
    for the result. Small next to the ERI array — 0.01 GB for nao = 560 and 40 orbitals — but
    it is stated rather than assumed.
    """
    return (res.array_gb((nao, nao), np.float64)
            + res.array_gb((2 * nao, n_orb), np.complex128)
            + res.array_gb((nao, n_orb), np.float64))


# --- Grouping orbitals into the blocks whose populations are well defined ------------------

def kramers_pair_groups(n_orb: int) -> List[np.ndarray]:
    """``[[0, 1], [2, 3], ...]`` — the interleaved Kramers pairs."""
    if n_orb % 2:
        raise ValueError("a Kramers-paired orbital set has an even number of spinors, got {}"
                         .format(n_orb))
    return [np.array([2 * p, 2 * p + 1]) for p in range(n_orb // 2)]


def degenerate_groups(occupation: np.ndarray,
                      tolerance: float = DEFAULT_DEGENERACY_TOLERANCE) -> List[np.ndarray]:
    """Group **adjacent** orbitals whose occupations agree to ``tolerance``.

    Adjacent, not sorted: the caller's ordering is preserved so the group indices still refer
    to the orbitals they were given. Degeneracy in the occupation number is a necessary
    condition for being in one manifold, not a sufficient one — two unrelated orbitals can
    share an occupation — so this is a convenience, and an explicit grouping is better
    wherever the manifold structure is actually known.
    """
    occupation = np.asarray(occupation, dtype=float)
    groups: List[List[int]] = []
    for i, occ in enumerate(occupation):
        if groups and abs(occ - occupation[groups[-1][0]]) <= tolerance:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [np.asarray(g, dtype=int) for g in groups]


def resolve_groups(group: Union[str, Sequence[Sequence[int]]], n_orb: int,
                   occupation: Optional[np.ndarray]) -> List[np.ndarray]:
    """Turn a ``group`` specification into explicit index groups. The one definition of the
    grouping rules; :mod:`kuiva.props.molden` uses it too, so the two cannot drift."""
    if isinstance(group, str):
        if group == "none":
            return [np.array([i]) for i in range(n_orb)]
        if group == "kramers":
            return kramers_pair_groups(n_orb)
        if group == "degenerate":
            if occupation is None:
                raise ValueError("group='degenerate' needs occupations")
            return degenerate_groups(occupation)
        raise ValueError("unknown grouping {!r}; expected 'none', 'kramers', 'degenerate' "
                         "or an explicit list of index groups".format(group))
    return [np.asarray(g, dtype=int).ravel() for g in group]


def warn_if_groups_not_degenerate(groups, occupation, group_spec,
                                  tolerance: float = 1e-4) -> float:
    """Warn if the orbitals summed into a group do not share an occupation.

    Grouping is only meaningful over a **degenerate block** — that is what makes a summed
    population, and the molden dump's scaling of a group's density by one occupation, well
    defined. Unequal occupations inside a group mean the grouping does not describe the orbital
    set it was applied to. Returns the largest spread found.
    """
    occupation = None if occupation is None else np.asarray(occupation, dtype=float)
    if occupation is None or not isinstance(group_spec, str) or group_spec == "none":
        return 0.0
    spread = max((float(occupation[g].max() - occupation[g].min())
                  for g in groups if g.size > 1), default=0.0)
    if spread > tolerance:
        log.warning("group=%r sums orbitals whose occupations differ by up to %.2e. Grouping "
                    "is only well defined over a degenerate block; if this orbital set is not "
                    "Kramers paired in the interleaved ordering, the groups "
                    "are not degenerate blocks and what they report is basis-dependent.",
                    group_spec, spread)
    return spread


# --- The transformed coefficients, which everything else is built from ---------------------

def lowdin_coefficients(c_ao: np.ndarray, s_ao: np.ndarray) -> np.ndarray:
    """``S^{1/2} C`` for both spin blocks: the spinors over symmetrically orthogonalized AOs.

    ``c_ao`` is ``(2*nao, n)`` in the spin-blocked row layout; the result has the same shape. The
    scalar root is applied to each spin block, which is what makes it the right transformation:
    orthogonalizing the AOs is a spin-free operation.
    """
    c_ao = np.asarray(c_ao)
    s_ao = np.asarray(s_ao)
    nao = s_ao.shape[0]
    if c_ao.shape[0] != 2 * nao:
        raise ValueError("spinor coefficients of shape {} do not match a scalar basis of {} "
                         "functions".format(c_ao.shape, nao))
    root = sqrt_overlap(s_ao)
    out_c = np.empty_like(c_ao, dtype=np.complex128)
    out_c[:nao] = root @ c_ao[:nao]
    out_c[nao:] = root @ c_ao[nao:]
    return out_c


# --- Atomic populations: charge and spin --------------------------------------------------

@dataclass(frozen=True)
class AtomicPopulations:
    """Loewdin charges and spin populations of a two-component density."""

    layout: AOLayout
    #: (nao,) electron population per orthogonalized AO.
    ao_population: np.ndarray
    #: (3, nao) spin population per orthogonalized AO, in units of hbar.
    ao_spin: np.ndarray

    @property
    def n_electrons(self) -> float:
        return float(self.ao_population.sum())

    def _by_atom(self, per_ao: np.ndarray) -> np.ndarray:
        out_a = np.zeros(per_ao.shape[:-1] + (self.layout.natm,), dtype=per_ao.dtype)
        for ia in range(self.layout.natm):
            out_a[..., ia] = per_ao[..., self.layout.atom_indices(ia)].sum(axis=-1)
        return out_a

    def atomic_population(self) -> np.ndarray:
        """(natm,) electrons assigned to each atom."""
        return self._by_atom(self.ao_population)

    def atomic_charge(self) -> np.ndarray:
        """(natm,) ``Z_A - N_A`` — a diagnostic, never a chemical statement.

        ⚠ Deliberately absent from :meth:`report` (measured decision, 2026-08-22): the
        Loewdin charge comes out with the wrong *sign* on ionic textbook compounds (module
        docstring). It stays as an accessor because the arithmetic is trivially implied by
        ``atomic_population`` and because sum-rule and conjugation-trap tests legitimately
        probe the density through it.
        """
        return self.layout.atom_charges - self.atomic_population()

    def atomic_spin(self) -> np.ndarray:
        """(3, natm) spin vector per atom."""
        return self._by_atom(self.ao_spin)

    def atomic_spin_magnitude(self) -> np.ndarray:
        """(natm,) ``|s_A|`` — the orientation-independent number (see the module docstring)."""
        return np.linalg.norm(self.atomic_spin(), axis=0)

    def report(self, logger=None, *, spin_vector: bool = False,
               title: str = "Loewdin atomic populations") -> None:
        """Log the atomic table (output grammar).

        ``spin_vector`` prints ``s_x, s_y, s_z`` beside ``|s|``. Off by default: the
        individual components depend on an arbitrary quantization axis, ``|s|`` does not.
        """
        logger = logger or log
        out.subsection(logger, title)
        # ⚠ No charge column, deliberately (measured decision, 2026-08-22, module docstring):
        # the Loewdin charge carries the wrong sign on ionic textbook compounds, and a number
        # that misleads is not repaired by a caption. atomic_charge() stays as an accessor.
        cols = [out.Column("atom", "{}", 10, "<"), out.Column("Z", "{:.0f}", 5),
                out.Column("population", "{:.4f}", 11),
                out.Column("|s|", "{:.4f}", 8)]
        if spin_vector:
            cols += [out.Column("s_x", "{:+.4f}", 9), out.Column("s_y", "{:+.4f}", 9),
                     out.Column("s_z", "{:+.4f}", 9)]
        table = out.Table(logger, cols)
        table.start()
        pop = self.atomic_population()
        spin, mag = self.atomic_spin(), self.atomic_spin_magnitude()
        for ia in range(self.layout.natm):
            row = [self.layout.atom_label(ia), self.layout.atom_charges[ia],
                   pop[ia], mag[ia]]
            if spin_vector:
                row += [spin[0, ia], spin[1, ia], spin[2, ia]]
            table.row(*row)
        table.end("total electrons {:.6f}, total charge {:+.4f}, |total spin| {:.4f}".format(
            self.n_electrons, float(self.layout.atom_charges.sum() - self.n_electrons),
            float(np.linalg.norm(self.ao_spin.sum(axis=1)))))


def atomic_populations(c_ao: np.ndarray, s_ao: np.ndarray, layout: AOLayout, *,
                       dm: Optional[np.ndarray] = None,
                       occupation: Optional[np.ndarray] = None) -> AtomicPopulations:
    """Loewdin charge and spin populations of the density built from ``c_ao``.

    Parameters
    ----------
    c_ao : ndarray (2*nao, n) complex
        Spinor coefficients in the **AO** basis, spin-blocked row layout.
    s_ao : ndarray (nao, nao)
        AO overlap.
    layout : AOLayout
    dm : ndarray (n, n), optional
        The one-particle density matrix in the spinor basis, ``gamma_pq = <a_p^dag a_q>``.
    occupation : ndarray (n,), optional
        Occupation numbers, for the case where the spinors are already natural. Exactly one of
        ``dm`` and ``occupation`` must be given.

    .. warning::
       ⚠ **The AO density built from a general ``gamma`` is ``C gamma^T C^dag``, not
       ``C gamma C^dag``** — the conjugation trap, hit twice elsewhere in this code.

       ⚠ **Here it is quieter than in either previous instance: the charge population cannot
       see it at all.** The transpose conjugates the diagonal of each *same-spin* block, and
       the population takes the real part of a Hermitian diagonal, so ``daa`` and ``dbb`` are
       **identical** under either convention — not approximately, algebraically. Measured on
       water with a random Hermitian ``gamma``: every atomic charge agrees to 1e-27, and the
       **spin density differs by 68%**, being built from the off-diagonal spin block where no
       such cancellation happens. So neither a charge nor a sum rule can
       validate this line; ``tests/test_population.py`` asserts it through ``ao_spin`` with a
       non-diagonal complex ``gamma``, which is the only place it shows.
    """
    if (dm is None) == (occupation is None):
        raise ValueError("give exactly one of dm= (a spinor-basis 1-RDM) and occupation=")

    c_ao = np.asarray(c_ao)
    nao = s_ao.shape[0]
    n_orb = c_ao.shape[1]
    res.require("Loewdin population analysis", population_memory_gb(nao, n_orb),
                note="{} AOs, {} spinors".format(nao, n_orb),
                advice=["analyse fewer orbitals (level='active')"])

    ct = lowdin_coefficients(c_ao, s_ao)
    ca, cb = ct[:nao], ct[nao:]

    if occupation is not None:
        w = np.asarray(occupation, dtype=float)
        # D^{ss'} = sum_i n_i c^s_i c^{s'}_i^dag; diagonals only, so form them directly.
        daa = np.einsum("mi,i,mi->m", ca, w, ca.conj()).real
        dbb = np.einsum("mi,i,mi->m", cb, w, cb.conj()).real
        dab = np.einsum("mi,i,mi->m", ca, w, cb.conj())
    else:
        gamma = np.asarray(dm)
        if gamma.shape != (n_orb, n_orb):
            raise ValueError("density matrix of shape {} for {} spinors"
                             .format(gamma.shape, n_orb))
        # ⚠ gamma.T, not gamma — see the warning above.
        gt = gamma.T
        daa = np.einsum("mi,ij,mj->m", ca, gt, ca.conj()).real
        dbb = np.einsum("mi,ij,mj->m", cb, gt, cb.conj()).real
        dab = np.einsum("mi,ij,mj->m", ca, gt, cb.conj())

    ao_population = daa + dbb
    ao_spin = np.stack([dab.real, -dab.imag, 0.5 * (daa - dbb)])
    return AtomicPopulations(layout=layout,
                             ao_population=np.ascontiguousarray(ao_population),
                             ao_spin=np.ascontiguousarray(ao_spin))


# --- Atomic-reference charges: the robust partition ----------------------------------------
#
# The charge partition that replaced the withdrawn Loewdin charge. Populations are taken in
# an orthonormal basis built from FREE-ATOM reference orbitals (a spherically averaged
# average-of-configuration sfx2c1e SCF per element, in the molecule's own basis, computed in
# the front end and carried on ScalarX2CData.atomic_reference): the occupied atomic orbitals
# are orthogonalized first, weighted by their atomic occupations, and the atomic virtuals
# are projected behind them and Loewdin-orthogonalized among themselves. Ligand density in
# the bonding region is then attributed by *atomic character* rather than by function label,
# which is what the plain Loewdin charge got wrong.
#
# Chosen by measurement, not preference. The battery that killed the Loewdin charge was run
# on this scheme (five systems, two to four bases, ghost-basis test, ROHF-vs-UHF, geometry
# perturbation): signs correct everywhere, worst basis drift ~0.1 e where Mulliken drifts
# 0.45 e and Loewdin flips sign, and a nucleus-free ghost basis over a pure chloride density
# receives ~0.1 e against Loewdin's 2.4 e. The numbers are in the package validation record;
# the two-tier structure is essential — a single occupancy-weighted orthogonalization
# without the occupied/virtual split leaks 0.8 e onto a ghost centre and was rejected.
#
# References: P.-O. Loewdin, J. Chem. Phys. 18, 365 (1950) (symmetric orthogonalization);
# B. C. Carlson, J. M. Keller, Phys. Rev. 105, 102 (1957) (weighted symmetric
# orthogonalization); A. E. Reed, R. B. Weinstock, F. Weinhold, J. Chem. Phys. 83, 735
# (1985) (natural population analysis — the occupancy-weighting and the occupied/virtual
# separation this scheme borrows); Q. Sun, G. K.-L. Chan, J. Chem. Theory Comput. 10, 3784
# (2014) (meta-Loewdin: populations in a minimal reference set orthogonalized first).

#: Atomic occupations above this count as an occupied reference orbital. Average of
#: configuration fills whole shells equally, so any threshold below the smallest fractional
#: filling (1/7 for f^1) cuts between shells, never through one.
REFERENCE_OCC_THRESHOLD = 1e-8


def _weighted_lowdin(s: np.ndarray, weight: Optional[np.ndarray] = None) -> np.ndarray:
    """``M`` with ``(TM)^dag S (TM) = 1``: Loewdin for equal weights, Carlson-Keller
    weighted symmetric orthogonalization otherwise (``M = W (W s W)^{-1/2} ``, so functions
    of small weight bend to preserve those of large weight)."""
    if weight is not None:
        w = np.asarray(weight, dtype=float)
        s = w[:, None] * s * w[None, :]
    e, v = np.linalg.eigh(s)
    keep = e > 1e-14 * float(e.max())
    m = (v[:, keep] / np.sqrt(e[keep])) @ v[:, keep].conj().T
    if keep.size != int(keep.sum()):
        log.warning("the atomic-reference orthogonalization dropped %d direction(s) as "
                    "numerically null; the charge sum rule holds only over the retained "
                    "space", keep.size - int(keep.sum()))
    return (w[:, None] * m) if weight is not None else m


@dataclass
class AtomicReferenceCharges:
    """Atomic charges in the free-atom reference partition, with their provenance."""
    layout: AOLayout
    charge: np.ndarray                     #: (natm,)
    population: np.ndarray                 #: (natm,) electrons per atom
    configurations: Dict[str, str]         #: element -> reference-state label
    any_non_default: bool
    all_converged: bool

    def atomic_charge(self) -> np.ndarray:
        return self.charge

    def report(self, logger=None, *, title: str = "Atomic-reference charges") -> None:
        logger = logger or log
        out.subsection(logger, title)
        table = out.Table(logger, [out.Column("atom", "{}", 10, "<"),
                                   out.Column("Z", "{:.0f}", 5),
                                   out.Column("population", "{:.4f}", 11),
                                   out.Column("charge", "{:+.4f}", 10)])
        table.start()
        for ia in range(self.layout.natm):
            table.row(self.layout.atom_label(ia), self.layout.atom_charges[ia],
                      self.population[ia], self.charge[ia])
        table.end("total charge {:+.4f}".format(float(self.charge.sum())))
        for sym in sorted(self.configurations):
            out.entry(logger, "reference state, " + sym, self.configurations[sym])
        if self.any_non_default:
            log.warning("one or more atomic-reference states differ from the per-element "
                        "defaults (the atomic mean field's: neutral atom, trivalent ion on "
                        "the f block). These charges are NOT comparable with charges "
                        "computed against the default references.")
        if not self.all_converged:
            log.warning("an atomic reference SCF did not converge; the charges built on it "
                        "are not trustworthy.")


def atomic_reference_charges(c_ao: np.ndarray, s_ao: np.ndarray, layout: AOLayout,
                             reference, *,
                             dm: Optional[np.ndarray] = None,
                             occupation: Optional[np.ndarray] = None,
                             report: bool = False) -> AtomicReferenceCharges:
    """Atomic charges of the density built from ``c_ao``, in the free-atom partition.

    ``c_ao`` is either a scalar set (``nao`` rows) or a spin-blocked spinor set (``2*nao``
    rows); with a spinor set the *spin-traced* density is analysed, which is the sector a
    charge lives in. Exactly one of ``dm`` (a 1-RDM over the given orbitals; ⚠ the AO
    density is ``C gamma^T C^dag`` — the conjugation-trap convention stated at
    :func:`atomic_populations`) and ``occupation`` must be given. ``reference`` is the
    :class:`kuiva.basis.reference.AtomicReferenceSet` the front end ingested
    (``atomic_reference=True``); without one this raises and names that knob.
    """
    if reference is None:
        raise ValueError(
            "atomic-reference charges need the per-element free-atom orbitals, which only "
            "the front end can compute: re-run the scalar SCF with atomic_reference=True "
            "(they are cached per element, so the cost is one small atomic SCF per unique "
            "element, once per process).")
    if (dm is None) == (occupation is None):
        raise ValueError("give exactly one of dm= (a 1-RDM) and occupation=")

    c_ao = np.asarray(c_ao)
    nao = int(s_ao.shape[0])
    if occupation is not None:
        w = np.asarray(occupation, dtype=float)
        d_full = (c_ao * w) @ c_ao.conj().T
    else:
        gamma = np.asarray(dm)
        d_full = c_ao @ gamma.T @ c_ao.conj().T          # ⚠ gamma.T: the stated convention
    if d_full.shape[0] == 2 * nao:                       # spinor density: spin-trace
        d = (d_full[:nao, :nao] + d_full[nao:, nao:]).real
    elif d_full.shape[0] == nao:
        d = d_full.real
    else:
        raise ValueError("orbitals with {} rows against {} AOs".format(c_ao.shape[0], nao))

    # Block-diagonal placement of each atom's reference orbitals, occupied columns first.
    t = np.zeros((nao, nao))
    weight = np.zeros(nao)
    occupied = np.zeros(nao, dtype=bool)
    owner = np.zeros(nao, dtype=int)
    configurations: Dict[str, str] = {}
    all_converged = True
    for ia in range(layout.natm):
        idx = np.asarray(layout.atom_indices(ia))
        sym = str(layout.atom_symbols[ia])
        if is_ghost(sym):
            # ⚠ A ghost has no free atom, so it has no *occupied* reference to project onto —
            # and that is exactly what makes its population worth reporting. Its functions
            # enter as atomic virtuals only (tier 2 below), so the density that lands on them
            # is density the real atoms borrowed from a basis that carries no electrons of
            # its own: the basis-set superposition error, as a number, per centre. Its
            # nuclear charge is zero, so the reported "charge" of a ghost is minus that
            # leaked population.
            t[np.ix_(idx, idx)] = np.eye(idx.size)
            owner[idx] = ia
            configurations[str(sym)] = "ghost (no nucleus, no reference)"
            continue
        sym = sym.capitalize()
        try:
            entry = reference.entry_for_atom(ia, sym) \
                if hasattr(reference, "entry_for_atom") else reference[sym]
        except KeyError:
            raise ValueError("the ingested atomic reference has no entry for atom {} "
                             "({}); it was built for a different molecule".format(
                                 ia + 1, sym))
        if entry.c.shape[0] != idx.size:
            raise ValueError(
                "the atomic reference for {} spans {} functions but this molecule gives the "
                "atom {}: the reference was built in a different basis".format(
                    sym, entry.c.shape[0], idx.size))
        order = np.argsort(-(entry.occ > REFERENCE_OCC_THRESHOLD).astype(int),
                           kind="stable")
        t[np.ix_(idx, idx)] = entry.c[:, order]
        occ_sorted = entry.occ[order]
        n_occ = int((occ_sorted > REFERENCE_OCC_THRESHOLD).sum())
        occupied[idx[:n_occ]] = True
        weight[idx[:n_occ]] = occ_sorted[:n_occ]
        owner[idx] = ia
        keys = getattr(reference, "atom_keys", None)
        key = keys[ia] if keys and keys[ia] else sym
        configurations[key] = entry.configuration
        all_converged = all_converged and entry.converged

    # Tier 1: the occupied atomic orbitals, weighted by their atomic occupations.
    c1 = t[:, occupied]
    c1 = c1 @ _weighted_lowdin(c1.T @ s_ao @ c1, weight[occupied])
    # Tier 2: atomic virtuals, projected out of tier 1, Loewdin among themselves. The split
    # cuts between whole shells only (average of configuration fills a shell equally), so
    # the degenerate-group discipline holds by construction.
    c2 = t[:, ~occupied] - c1 @ (c1.T @ s_ao @ t[:, ~occupied])
    c2 = c2 @ _weighted_lowdin(c2.T @ s_ao @ c2)
    x = np.empty((nao, nao))
    x[:, occupied] = c1
    x[:, ~occupied] = c2

    pop_per_fn = np.einsum("mi,mn,ni->i", x, s_ao @ d @ s_ao, x)
    population = np.array([pop_per_fn[owner == ia].sum() for ia in range(layout.natm)])
    result = AtomicReferenceCharges(
        layout=layout, charge=np.asarray(layout.atom_charges, dtype=float) - population,
        population=population, configurations=configurations,
        any_non_default=bool(getattr(reference, "any_non_default", False)),
        all_converged=all_converged)
    if report:
        result.report()
    return result


# --- Reduced AO populations, per orbital or per degenerate block ---------------------------

@dataclass(frozen=True)
class OrbitalPopulations:
    """Reduced AO populations of a set of orbitals or degenerate blocks.

    ``ao[:, g]`` sums to the number of orbitals in group ``g`` (each normalized spinor
    contributes 1), which is what makes the percentages comparable between groups of
    different size.
    """

    layout: AOLayout
    #: (nao, ngroup) population of each orthogonalized AO in each group.
    ao: np.ndarray
    #: One label per group, e.g. ``"12-13"`` for a Kramers pair.
    labels: Tuple[str, ...]
    #: (ngroup,) mean occupation of the orbitals in the group, if it was supplied.
    occupation: Optional[np.ndarray] = None
    #: (ngroup,) mean orbital energy [Eh], if it was supplied.
    energy: Optional[np.ndarray] = None
    #: The index groups themselves, as given.
    groups: Tuple[np.ndarray, ...] = ()

    @property
    def n_groups(self) -> int:
        return len(self.labels)

    def normalized(self) -> np.ndarray:
        """(nao, ngroup) populations as fractions of the group, summing to 1 per column."""
        total = self.ao.sum(axis=0)
        return self.ao / np.where(total > 0.0, total, 1.0)

    def by_atom(self) -> np.ndarray:
        """(natm, ngroup)."""
        return np.stack([self.ao[self.layout.atom_indices(ia)].sum(axis=0)
                         for ia in range(self.layout.natm)])

    def by_ao_type(self) -> Dict[Tuple[int, str], np.ndarray]:
        """``{(atom, label): (ngroup,)}`` — the reduced *atomic orbital* populations."""
        return {key: self.ao[idx].sum(axis=0)
                for key, idx in self.layout.group_by_ao_type().items()}

    def by_angular_momentum(self) -> Dict[Tuple[int, int], np.ndarray]:
        """``{(atom, l): (ngroup,)}`` — the unambiguous grouping (no principal quantum number,
        which is basis-dependent; see :mod:`kuiva.basis.layout`)."""
        result: Dict[Tuple[int, int], np.ndarray] = {}
        for ia in range(self.layout.natm):
            on_atom = self.layout.ao_atom == ia
            for l in np.unique(self.layout.ao_l[on_atom]):
                result[(ia, int(l))] = self.ao[on_atom & (self.layout.ao_l == l)].sum(axis=0)
        return result

    def report(self, logger=None, *, tolerance: float = DEFAULT_PRINT_TOLERANCE,
               title: str = "Loewdin reduced orbital populations") -> None:
        """Log the reduced-AO table (output grammar), one row per contribution.

        Contributions below ``tolerance`` (as a fraction of the group) are omitted and counted
        in a "remainder" row, so the printed rows always account for the whole orbital — a
        table that silently drops 30% of an orbital is worse than no table.
        """
        logger = logger or log
        out.subsection(logger, title)
        frac = self.normalized()
        groups = self.layout.group_by_ao_type()
        table = out.Table(logger, [
            out.Column("spinor", "{}", 12, "<"),
            out.Column("occ", "{:.4f}", 8),
            out.Column("energy [Eh]", out.E_FMT, 16),
            out.Column("atom", "{}", 10, "<"),
            out.Column("AO", "{}", 9, "<"),
            out.Column("%", "{:.2f}", 8),
        ])
        table.start()
        for g in range(self.n_groups):
            contributions = sorted(
                ((float(frac[idx, g].sum()), key) for key, idx in groups.items()),
                key=lambda kv: -kv[0])
            shown = 0.0
            first = True
            for value, (ia, label) in contributions:
                if value < tolerance:
                    continue
                table.row(self.labels[g] if first else "",
                          self._occ(g) if first else "",
                          self._ene(g) if first else "",
                          self.layout.atom_label(ia), label, 100.0 * value)
                shown += value
                first = False
            if first:                       # everything fell below the tolerance
                table.row(self.labels[g], self._occ(g), self._ene(g), "", "(all below tol)",
                          0.0)
            elif 1.0 - shown > tolerance:
                table.row("", "", "", "", "remainder", 100.0 * (1.0 - shown))
        table.end("contributions below {:.1%} of a spinor are collected in 'remainder'"
                  .format(tolerance))

    def _occ(self, g: int):
        return "" if self.occupation is None else self.occupation[g]

    def _ene(self, g: int):
        return "" if self.energy is None else self.energy[g]


def orbital_populations(c_ao: np.ndarray, s_ao: np.ndarray, layout: AOLayout, *,
                        columns: Optional[Sequence[int]] = None,
                        group: Union[str, Sequence[Sequence[int]]] = "kramers",
                        occupation: Optional[np.ndarray] = None,
                        energy: Optional[np.ndarray] = None) -> OrbitalPopulations:
    """Reduced AO populations of selected orbitals.

    Parameters
    ----------
    c_ao : ndarray (2*nao, n) complex
        Spinor coefficients in the AO basis.
    columns : sequence of int, optional
        Which spinors to analyse; all of them by default. ``occupation`` and ``energy``, if
        given, are indexed by the **same** column list, i.e. they are arrays over ``c_ao``'s
        columns and are sliced here.
    group : {"kramers", "none", "degenerate"} or list of index groups
        How the selected columns are grouped before the populations are summed.
        **"kramers" (the default) sums adjacent pairs**, which is the interleaved partner ordering.
        ⚠ Per-spinor rows (``"none"``) are basis-dependent inside a degenerate manifold and
        are offered for inspection, not for quoting — see the module docstring. Group indices
        are relative to ``columns``.
    """
    c_ao = np.asarray(c_ao)
    columns = (np.arange(c_ao.shape[1]) if columns is None
               else np.asarray(columns, dtype=int).ravel())
    occ = None if occupation is None else np.asarray(occupation, dtype=float)[columns]
    ene = None if energy is None else np.asarray(energy, dtype=float)[columns]

    nao = s_ao.shape[0]
    res.require("Loewdin orbital populations", population_memory_gb(nao, columns.size),
                note="{} AOs, {} spinors".format(nao, columns.size),
                advice=["analyse fewer orbitals (level='active')"])

    ct = lowdin_coefficients(np.ascontiguousarray(c_ao[:, columns]), s_ao)
    # |S^{1/2} c|^2 summed over the two spin blocks: real, non-negative, columns sum to 1.
    per_orbital = (np.abs(ct[:nao]) ** 2 + np.abs(ct[nao:]) ** 2)

    groups = resolve_groups(group, columns.size, occ)
    ao = np.stack([per_orbital[:, g].sum(axis=1) for g in groups], axis=1)
    labels = tuple(_group_label(columns, g) for g in groups)

    warn_if_groups_not_degenerate(groups, occ, group)

    return OrbitalPopulations(
        layout=layout, ao=np.ascontiguousarray(ao), labels=labels,
        occupation=None if occ is None else np.array([occ[g].mean() for g in groups]),
        energy=None if ene is None else np.array([ene[g].mean() for g in groups]),
        groups=tuple(groups))


def _group_label(columns: np.ndarray, g: np.ndarray) -> str:
    idx = columns[g]
    if idx.size == 1:
        return str(int(idx[0]))
    if idx.size == 2:
        return "{}-{}".format(int(idx[0]), int(idx[1]))
    return "{}..{} ({})".format(int(idx.min()), int(idx.max()), idx.size)


# --- Selecting which orbitals to analyse ---------------------------------------------------

def frontier_columns(occupation: np.ndarray, n_below: int = 5, n_above: int = 5,
                     threshold: float = 0.5, *, pairs: bool = True) -> np.ndarray:
    """Columns around the HOMO-LUMO gap: the last ``n_below`` occupied and the first
    ``n_above`` empty, with "occupied" meaning an occupation above ``threshold``.

    For a spinor set an occupation runs 0..1, so 0.5 is the natural midpoint and a
    fractionally occupied active orbital counts as occupied.

    ⚠ **With ``pairs=True`` (the default) the counts are in Kramers pairs and whole pairs are
    always returned.** A frontier selected spinor by spinor cuts pairs in half — the HOMO and
    LUMO are adjacent columns from *different* pairs — and a half pair is exactly the
    basis-dependent object the grouping rules exist to avoid, so the default is not to produce
    one. Set ``pairs=False`` for a literal per-spinor window.
    """
    occupation = np.asarray(occupation, dtype=float)
    occupied = np.nonzero(occupation > threshold)[0]
    empty = np.nonzero(occupation <= threshold)[0]
    if not pairs:
        return np.concatenate([occupied[-n_below:] if n_below else occupied[:0],
                               empty[:n_above] if n_above else empty[:0]])
    # Work in spatial (Kramers-pair) indices, then expand both partners.
    occ_pairs = np.unique(occupied // 2)
    empty_pairs = np.setdiff1d(np.unique(empty // 2), occ_pairs)
    chosen = np.concatenate([occ_pairs[-n_below:] if n_below else occ_pairs[:0],
                             empty_pairs[:n_above] if n_above else empty_pairs[:0]])
    return spinor_indices(np.sort(chosen))


def select_columns(level: str, n_orb: int, *, active: Optional[Sequence[int]] = None,
                   occupation: Optional[np.ndarray] = None, n_frontier: int = 5
                   ) -> np.ndarray:
    """Resolve a printing ``level`` to a column list (the user-facing knob).

    ``"active"`` (the default everywhere) needs ``active``; ``"frontier"`` needs
    ``occupation``; ``"all"`` is the whole set and is deliberately not a default — for a heavy
    element it is hundreds of orbitals and thousands of table rows.
    """
    if level == "active":
        if active is None:
            raise ValueError("level='active' needs the active-space columns")
        return np.asarray(active, dtype=int).ravel()
    if level == "frontier":
        if occupation is None:
            raise ValueError("level='frontier' needs occupations")
        return frontier_columns(occupation, n_frontier, n_frontier)
    if level == "all":
        log.warning("level='all' prints a reduced-AO table for every one of the %d spinors; "
                    "this is a large output and is only meaningful when it was asked for "
                    "deliberately", n_orb)
        return np.arange(n_orb)
    raise ValueError("unknown analysis level {!r}; expected 'active', 'frontier' or 'all'"
                     .format(level))


# --- The driver ---------------------------------------------------------------------------

def lowdin_analysis(c_ao: np.ndarray, s_ao: np.ndarray, layout: AOLayout, *,
                    dm: Optional[np.ndarray] = None,
                    occupation: Optional[np.ndarray] = None,
                    energy: Optional[np.ndarray] = None,
                    level: str = "active", active: Optional[Sequence[int]] = None,
                    n_frontier: int = 5,
                    group: Union[str, Sequence[Sequence[int]]] = "kramers",
                    tolerance: float = DEFAULT_PRINT_TOLERANCE,
                    spin_vector: bool = False, report: bool = True
                    ) -> Tuple[AtomicPopulations, Optional[OrbitalPopulations]]:
    """Full Loewdin analysis: atomic charges and spin, plus reduced AO populations.

    The atomic table is always produced — it is one line per atom and there is no reason to
    suppress it. The reduced-AO table is governed by ``level``: ``"active"`` (default),
    ``"frontier"`` or ``"all"``, and by ``tolerance``, below which a contribution is not
    printed. ``level="active"`` without ``active=`` produces no orbital table rather than
    raising, so a caller that has no active space yet still gets the charges.

    ``dm`` or ``occupation`` supplies the density for the atomic part (exactly one); the
    reduced populations need neither, being properties of the orbitals themselves.
    """
    atomic = atomic_populations(c_ao, s_ao, layout, dm=dm, occupation=occupation)
    orbital = None
    if not (level == "active" and active is None):
        columns = select_columns(level, c_ao.shape[1], active=active,
                                 occupation=occupation, n_frontier=n_frontier)
        occ_for_table = occupation
        if occ_for_table is None and dm is not None:
            occ_for_table = np.clip(np.real(np.diag(dm)), 0.0, None)
        orbital = orbital_populations(c_ao, s_ao, layout, columns=columns, group=group,
                                      occupation=occ_for_table, energy=energy)
    if report:
        out.section(log, "Loewdin population analysis")
        atomic.report(spin_vector=spin_vector)
        if orbital is not None:
            orbital.report(tolerance=tolerance)
    return atomic, orbital


__all__ = ["DEFAULT_PRINT_TOLERANCE", "AtomicPopulations", "OrbitalPopulations",
           "atomic_populations", "degenerate_groups", "frontier_columns",
           "kramers_pair_groups", "lowdin_analysis", "lowdin_coefficients", "resolve_groups",
           "orbital_populations", "population_memory_gb", "select_columns",
           "warn_if_groups_not_degenerate"]
