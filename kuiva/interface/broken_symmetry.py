"""The broken-symmetry starting guess: "this centre is spin-up" turned into a density.

Why a guess needs building at all
---------------------------------
For antiferromagnetically coupled centres a single determinant is qualitatively wrong, and an
unrestricted reference is the cheapest thing that is not: it can put up-spin on one metal and
down-spin on the other, which is the classical picture of the coupling and a usable starting
point for everything after it. ⚠ **But an unrestricted SCF started from a closed-shell density
stays closed-shell**: the symmetric point is a stationary point of the energy, so the
iteration has nothing to push it off, and the user gets the restricted answer wearing the
letters UHF. The polarization has to be *put in* by the starting density.

What this builds, and why through localized orbitals
----------------------------------------------------
The recipe is the standard one (Noodleman): converge the **high-spin** state, which is easy
and unambiguous, take its singly occupied orbitals, and **flip the ones that belong to the
centres the user asked to be spin-down** into the beta set. The result is a density with the
requested spin pattern and the right electron count, and the SCF is run from it.

⚠ **"The ones that belong to a centre" is a localization, not an atom-blocked density**, and
that is why this module is written on :mod:`kuiva.mcscf.localize` rather than on a partition
of the density matrix. For two equivalent metals the high-spin SCF returns the symmetric and
antisymmetric combinations of the two magnetic orbitals — each half on each centre — so
flipping "the orbital on metal 2" is not a statement one can make about the canonical set at
all. Localize first, and it is exact.

⚠ **The result is a spin-contaminated single determinant, on purpose.** A broken-symmetry
solution is not an eigenfunction of ``S^2``: its ``<S^2>`` sits between the low-spin and
high-spin values, and that is the diagnostic that it *is* broken-symmetry rather than a
converged-back-to-symmetric run. Nothing here maps it onto a spin-projected energy; it is a
starting point for the multireference stage, which is where the physics is done.

References
----------
The construction is L. Noodleman, "Valence bond description of antiferromagnetic coupling in
transition metal dimers", *J. Chem. Phys.* **74**, 5737 (1981), doi:10.1063/1.440939, in the
practical form that converges the high-spin state first and flips localized magnetic orbitals
— see also L. Noodleman, E. R. Davidson, *Chem. Phys.* **109**, 131 (1986),
doi:10.1016/0301-0104(86)80192-6. The localization that makes "this centre" well defined is
:mod:`kuiva.mcscf.localize`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..basis.atommap import resolve_atom_assignments
from ..util import output as out
from ..util.logging import get_logger
from ..util.timing import timer

log = get_logger(__name__)


@dataclass(frozen=True)
class BrokenSymmetryGuess:
    """The starting density, and the high-spin calculation it was built from."""

    dm0: np.ndarray
    spins: Tuple[int, ...]
    site_labels: Tuple[str, ...]
    populations: np.ndarray
    e_high_spin: float
    s2_high_spin: float
    n_flipped: int

    @property
    def weakest(self) -> float:
        """Smallest own-site population over the magnetic orbitals that were assigned."""
        if self.populations.size == 0:
            return 1.0
        return float(np.min(np.max(self.populations, axis=1)))

    def report(self, logger=None) -> None:
        logger = logger or log
        out.entry(logger, "broken-symmetry guess",
                  " ".join("{}{:+d}".format(lab, s)
                           for lab, s in zip(self.site_labels, self.assigned)),
                  note="unpaired electrons per site, signed")
        out.entry(logger, "high-spin reference energy", self.e_high_spin, "Eh",
                  fmt=out.E_FMT, note="the solution the magnetic orbitals came from")
        out.entry(logger, "high-spin <S^2>", self.s2_high_spin, "", fmt="{:.4f}")
        out.entry(logger, "magnetic orbitals flipped to beta", self.n_flipped, "",
                  note="localized on their own site to {:.3f} or better"
                       .format(self.weakest))

    @property
    def assigned(self) -> Tuple[int, ...]:
        """The signed spin of each site, in :attr:`site_labels` order."""
        return tuple(s for s in self.spins if s)


def resolve_spins(spec, symbols: Sequence[str]) -> List[int]:
    """Per-atom signed unpaired-electron counts from a user's assignment map.

    The keys are :mod:`kuiva.basis.atommap`'s — an element symbol, an atom label (``"Fe2"``),
    or a 1-based atom number — so "which atom" means the same thing here as it does for a
    per-atom basis or a reference configuration. ⚠ A bare element symbol assigns **every**
    atom of that element the same sign, which for a homonuclear dimer is exactly the symmetric
    guess this exists to avoid; that case is refused rather than run.
    """
    values, _specific = resolve_atom_assignments(spec, symbols, what="broken-symmetry spin",
                                                 default=0, allow_scalar=False)
    spins = []
    for i, v in enumerate(values):
        try:
            spins.append(int(v))
        except (TypeError, ValueError):
            raise ValueError("broken_symmetry values are signed integer counts of unpaired "
                             "electrons; atom {} got {!r}".format(i + 1, v))
    if not any(spins):
        raise ValueError("broken_symmetry assigns no unpaired electrons to any atom; a "
                         "guess with nothing to flip is the symmetric one")
    if all(s >= 0 for s in spins) or all(s <= 0 for s in spins):
        raise ValueError(
            "broken_symmetry assigns the same sign to every centre ({}), which is the "
            "high-spin state and not a broken-symmetry guess. Give the centres opposite "
            "signs — {{'Fe1': +5, 'Fe2': -5}} is an antiferromagnetically coupled pair."
            .format({i + 1: s for i, s in enumerate(spins) if s}))
    return spins


def _somo_columns(mf) -> np.ndarray:
    """The singly occupied alpha orbitals of a converged high-spin UHF."""
    occ_a, occ_b = np.asarray(mf.mo_occ[0]), np.asarray(mf.mo_occ[1])
    n_a, n_b = int(np.sum(occ_a > 0)), int(np.sum(occ_b > 0))
    if n_a <= n_b:
        raise ValueError("the high-spin reference has no singly occupied orbitals to "
                         "localize (alpha {} = beta {}); there is nothing to flip"
                         .format(n_a, n_b))
    return np.arange(n_b, n_a)


def broken_symmetry_density(mol, spec, layout, *, controls: Optional[dict] = None,
                            conv_tol: Optional[float] = None,
                            max_cycle: Optional[int] = None,
                            min_population: Optional[float] = None,
                            embedding=None, report: bool = True) -> BrokenSymmetryGuess:
    """Build ``(dm_alpha, dm_beta)`` for the broken-symmetry state ``spec`` describes.

    Parameters
    ----------
    mol : PySCF ``Mole``
        The molecule as the *target* calculation will run it, i.e. already at the
        broken-symmetry ``spin`` (``2 Ms``).
    spec : mapping
        Signed unpaired-electron counts per atom — ``{"Fe1": +5, "Fe2": -5}``. See
        :func:`resolve_spins`.
    layout : :class:`kuiva.basis.layout.AOLayout`
    controls, conv_tol, max_cycle
        The convergence controls of the target SCF, applied to the high-spin solve too: the
        guess is only as good as the solution it is built from, and a high-spin state that
        needed a level shift will need it here as well.
    min_population : float, optional
        Own-site population every magnetic orbital must reach
        (:data:`kuiva.mcscf.localize.DEFAULT_SITE_POPULATION_MIN`). ⚠ The refusal is the
        useful outcome when it fails: flipping an orbital that is half on the other centre
        produces a density that is *not* the requested spin pattern, and the SCF then
        converges back to the symmetric solution with nothing having gone visibly wrong.

    Raises
    ------
    ValueError
        If the assignment contradicts the molecule's own ``spin``, if the high-spin state has
        the wrong number of singly occupied orbitals for it, or if the magnetic orbitals do
        not localize.
    RuntimeError
        If the high-spin SCF does not converge — the guess would be meaningless.
    """
    from ..mcscf.localize import DEFAULT_SITE_POPULATION_MIN, localize
    from .pyscf_bridge import _build_scf, _embed_scf, apply_scf_controls

    spins = resolve_spins(spec, [str(s) for s in layout.atom_symbols])
    n_up = sum(s for s in spins if s > 0)
    n_down = -sum(s for s in spins if s < 0)
    if n_up - n_down != int(mol.spin):
        raise ValueError(
            "the broken-symmetry assignment has {} alpha and {} beta unpaired electrons, "
            "i.e. 2 Ms = {}, but the molecule was built with spin (2S) = {}. The two are the "
            "same statement and must agree — set spin={} on the Molecule, or change the "
            "assignment.".format(n_up, n_down, n_up - n_down, int(mol.spin), n_up - n_down))

    n_somo = n_up + n_down
    if n_somo > int(mol.nelectron) or (int(mol.nelectron) - n_somo) % 2:
        raise ValueError(
            "the assignment asks for {} unpaired electrons in total, which no state of this "
            "molecule has: it carries {} electrons, so the high-spin solution the guess is "
            "built from would need 2S = {} and cannot. Check the oxidation states and the "
            "charge.".format(n_somo, int(mol.nelectron), n_somo))

    sites = [i for i, s in enumerate(spins) if s]
    counts = [abs(spins[i]) for i in sites]

    # -- the high-spin solution, which is the easy one and the one the flip is made from -----
    hs = mol.copy()
    hs.spin = n_up + n_down
    hs.build(False, False)
    mf_hs, _name = _build_scf(hs, "uhf")
    if embedding is not None:
        mf_hs = _embed_scf(mf_hs, embedding)
    if conv_tol is not None:
        mf_hs.conv_tol = conv_tol
    if max_cycle is not None:
        mf_hs.max_cycle = max_cycle
    if controls:
        mf_hs, _note = apply_scf_controls(mf_hs, **controls)
    with timer("high-spin SCF (broken-symmetry guess)"):
        e_hs = float(mf_hs.kernel())
    if not mf_hs.converged:
        raise RuntimeError(
            "the high-spin SCF the broken-symmetry guess is built from did not converge "
            "(E = {:.8f} Eh, 2S = {}). The flip needs its singly occupied orbitals, so the "
            "run stops here rather than flipping an unconverged set: try the convergence "
            "controls (level_shift=, damp=, diis='adiis', second_order=True), which reach "
            "this solve too.".format(e_hs, hs.spin))
    s2_hs = float(mf_hs.spin_square()[0])

    somo = _somo_columns(mf_hs)
    if somo.size != n_up + n_down:
        raise ValueError(
            "the high-spin solution has {} singly occupied orbitals but the assignment asks "
            "for {} ({}). The magnetic orbital count is a property of the state, so this is "
            "an assignment the molecule does not have — check the oxidation states and the "
            "charge.".format(somo.size, n_up + n_down,
                             {i + 1: s for i, s in enumerate(spins) if s}))

    c_a = np.asarray(mf_hs.mo_coeff[0])
    floor = DEFAULT_SITE_POPULATION_MIN if min_population is None else float(min_population)
    loc = localize(c_a, np.asarray(mf_hs.get_ovlp()), layout, somo, sites, counts=counts,
                   min_population=floor)

    # -- the flip: every site's magnetic orbitals go to the set its sign names ---------------
    n_beta_hs = int(np.sum(np.asarray(mf_hs.mo_occ[1]) > 0))
    core_a = c_a[:, :n_beta_hs]
    core_b = np.asarray(mf_hs.mo_coeff[1])[:, :n_beta_hs]
    up_cols, down_cols = [], []
    for k, i in enumerate(sites):
        cols = loc.site_columns(k)
        (up_cols if spins[i] > 0 else down_cols).extend(int(c) for c in cols)
    occ_a = np.hstack([core_a, loc.coeff[:, up_cols]]) if up_cols else core_a
    occ_b = np.hstack([core_b, loc.coeff[:, down_cols]]) if down_cols else core_b
    dm0 = np.array([occ_a @ occ_a.T, occ_b @ occ_b.T])

    n_elec = float(np.trace(dm0[0] @ mf_hs.get_ovlp()) + np.trace(dm0[1] @ mf_hs.get_ovlp()))
    if abs(n_elec - mol.nelectron) > 1e-8:
        raise AssertionError("the broken-symmetry density carries {:.6f} electrons against "
                             "the molecule's {}".format(n_elec, mol.nelectron))

    guess = BrokenSymmetryGuess(
        dm0=dm0, spins=tuple(spins),
        site_labels=tuple(layout.atom_label(i) for i in sites),
        populations=loc.populations, e_high_spin=e_hs, s2_high_spin=s2_hs,
        n_flipped=len(down_cols))
    if report:
        out.subsection(log, "Broken-symmetry starting guess")
        guess.report()
    return guess


__all__ = ["BrokenSymmetryGuess", "broken_symmetry_density", "resolve_spins"]
