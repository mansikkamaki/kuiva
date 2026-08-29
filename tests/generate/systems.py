"""Shared definitions of the Tier-1/Tier-2 validation systems.

**One source of truth.** Geometries, charges, spins, active spaces and state counts live
here and are consumed by every reference generator (``tier1_pyscf.py``, ``tier2_molcas.py``,
``tier2_dirac.py``) so that the scalar (Tier-1) and spin-orbit (Tier-2) references describe
*the same* systems. A Tier-1 entry is therefore the SOC-free counterpart of the Tier-2 entry
with the same key, which is what lets the suite separate "is the scalar CASSCF right?" from
"is the SOC treatment right?".

Selection rationale (heavy-element, strongly correlated, strongly relativistic,
including multi-site SMM targets), and what each system is here to catch:

===============  ==========================================================================
key              what it tests
===============  ==========================================================================
``ne``           light closed shell; cheapest end-to-end smoke test of SCF/CASSCF/NEVPT2.
``zn2p``         3d^10 closed shell; the CASCI-in-a-full-active-space == SCF invariance, and
                 a transition-metal NEVPT2 correlation energy.
``hi``           light-heavy bond; a molecular sigma/sigma* CAS at low cost.
``bi``           6p^3: genuinely multireference *and* strongly spin-orbit coupled. The 20
                 states of the p^3 manifold (4S3/2, 2D3/2, 2D5/2, 2P1/2, 2P3/2) are a
                 stringent structural test with well-known experimental energies.
``tlh``          molecular heavy diatomic; SOC-split Pi states. The classic 2c benchmark.
``ce3p``         4f^1 — the minimal f-element problem: one electron in 14 spinors. Exact
                 6+8 (2F5/2 / 2F7/2) structure and an analytic Lande g = 6/7.
``yb3p``         4f^13 — the one-*hole* counterpart of ce3p (ground 2F7/2, g = 8/7). Ce/Yb
                 together exercise particle-hole symmetry of the CI string addressing.
``dy3p``         4f^9 — strongly correlated many-electron f shell and *the* SMM ion; ground
                 6H15/2 with g = 4/3. Still a tiny CI (C(14,9) = 2002 determinants).
``cecl3``        single f^1 site in a ligand field: crystal-field splitting of 2F5/2.
``ticl3``        single d^1 site in a ligand field, and the monomer the dimers below must
                 factorise back into. Planar D3h Ti(III): a1' + e'' + e' -> 5 Kramers
                 doublets once SOC is on.
``ti2cl6``       **multi-site**: two coupled d^1 centres. The 100-state d^1 (x) d^1 manifold
                 is exactly the local-multiplet product structure the DMRG multi-site
                 machinery must reproduce.
``ti2cl6_far``   the same dimer at 25 A: the spectrum must factorise into the tensor product
                 of two monomer multiplets with energies E_A + E_B, giving 100 states in 15
                 blocks of size 4 (both sites on one level, 5 of them) and 8 (sites on
                 different levels, 10 of them). This is the sharpest available test of
                 local-multiplet construction and of size-consistency.
===============  ==========================================================================

Why d^1 and not f^1 for the dimer (a recorded decision)
-------------------------------------------------------------
The multi-site pair was originally Ce2Cl6 (two f^1 centres, CAS(2,14), 98 state-averaged
roots). It was correct but unusable: **one** scalar Tier-1 record cost 7.86 CPU-hours, and a
reference that takes half a day to regenerate cannot be re-run while the code that consumes
it is being developed. Ti(III) d^1 is the same problem with 5 orbitals per site instead of
7 — one electron per centre, no intra-site correlation — at CAS(2,10) and 50 roots, which
measured **569 CPU-seconds** for the same record: a ~50x reduction for identical test logic.

Cu(II) d^9 was considered and rejected: particle-hole symmetry makes its CI exactly as large
as d^1 (two holes in ten orbitals) while adding seven more electrons per centre to the SCF
and the integral transformation. A 4d/5d centre would give stronger SOC but a larger basis;
strong-SOC physics is already covered single-site by ``bi``/``tlh``/``dy3p``/``yb3p``/
``ce3p``/``cecl3``, and the factorisation this pair tests is exact at any SOC/ligand-field
ratio. ``cecl3`` is retained: it is a cheap monomer and remains the 4f ligand-field case.

Geometry provenance
-------------------
Diatomic bond lengths are experimental equilibrium values (see per-system ``geom_note``).
``CeCl3`` and ``TiCl3`` use gas-phase planar D3h structures. ``Ti2Cl6`` is a *fixed model*
D2h edge-sharing dimer built from typical Ti(III)-Cl distances — it is deliberately **not**
an optimised structure, because a test system only has to be well defined, reproducible and
representative of the physics; it must never be quoted as a prediction.

Basis policy (and why it matters here)
-------------------------------------------
Cross-code comparison is only informative if the basis is held fixed, so each system carries
*two* basis labels:

* ``basis`` — ``x2c-*all-2c``, the project default, exercising the registry/BSE path;
* ``basis_matched`` — the basis the external code uses, which PySCF also has bundled
  (``ano-rcc-vdzp`` for OpenMolcas, ``dyallv2z`` for DIRAC; ``tests/reference/
  basis_crosscheck.json`` already verifies the primitives agree). Running Tier 1 in the
  matched basis too means a Tier-1/Tier-2 discrepancy is attributable to the *Hamiltonian
  and method*, not to the basis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Atom = Tuple[str, Tuple[float, float, float]]


# --- geometry builders -----------------------------------------------------------------
def _atom(sym: str) -> List[Atom]:
    return [(sym, (0.0, 0.0, 0.0))]


def _diatomic(a: str, b: str, r: float) -> List[Atom]:
    return [(a, (0.0, 0.0, 0.0)), (b, (0.0, 0.0, r))]


def _cecl3(origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
           r: float = 2.569) -> List[Atom]:
    """Planar D3h CeCl3 (gas phase), Ce at ``origin``, Cl in the xy plane."""
    import math
    ox, oy, oz = origin
    atoms: List[Atom] = [("Ce", (ox, oy, oz))]
    for k in range(3):
        th = 2.0 * math.pi * k / 3.0
        atoms.append(("Cl", (ox + r * math.cos(th), oy + r * math.sin(th), oz)))
    return atoms


def _ticl3(origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
           r: float = 2.25) -> List[Atom]:
    """Planar D3h TiCl3 (gas phase), Ti at ``origin``, Cl in the xy plane."""
    import math
    ox, oy, oz = origin
    atoms: List[Atom] = [("Ti", (ox, oy, oz))]
    for k in range(3):
        th = 2.0 * math.pi * k / 3.0
        atoms.append(("Cl", (ox + r * math.cos(th), oy + r * math.sin(th), oz)))
    return atoms


def _ti2cl6() -> List[Atom]:
    """Fixed D2h edge-sharing model dimer: 2 bridging + 4 terminal Cl.

    Built from Ti...Ti = 3.50 A, Ti-Cl(bridge) = 2.45 A, Ti-Cl(terminal) = 2.20 A with an
    80 degree terminal Cl-Ti-Cl angle. Model geometry, not an optimised structure.
    """
    import math
    a = 1.75                                        # half the Ti...Ti separation
    r_b, r_t, half_ang = 2.45, 2.20, math.radians(40.0)
    b = math.sqrt(r_b ** 2 - a ** 2)                # bridging Cl offset along y
    dx, dz = r_t * math.cos(half_ang), r_t * math.sin(half_ang)
    atoms: List[Atom] = [("Ti", (+a, 0.0, 0.0)), ("Ti", (-a, 0.0, 0.0)),
                         ("Cl", (0.0, +b, 0.0)), ("Cl", (0.0, -b, 0.0))]
    for sx in (+1.0, -1.0):
        for sz in (+1.0, -1.0):
            atoms.append(("Cl", (sx * (a + dx), 0.0, sz * dz)))
    return atoms


def _ti2cl6_far(sep: float = 25.0) -> List[Atom]:
    """Two TiCl3 monomers at ``sep`` Angstrom — non-interacting by construction.

    Built from :func:`_ticl3` so the dimer spectrum must factorise back into exactly the
    ``ticl3`` monomer levels; that inversion is the multi-site test.
    """
    return _ticl3(origin=(0.0, 0.0, 0.0)) + _ticl3(origin=(sep, 0.0, 0.0))


def _tif3(origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
          r: float = 1.780) -> List[Atom]:
    """Planar D3h TiF3, Ti at ``origin``, F in the xy plane (gas-phase-like model)."""
    import math
    ox, oy, oz = origin
    atoms: List[Atom] = [("Ti", (ox, oy, oz))]
    for k in range(3):
        th = 2.0 * math.pi * k / 3.0
        atoms.append(("F", (ox + r * math.cos(th), oy + r * math.sin(th), oz)))
    return atoms


def _fecl2(r: float = 2.151) -> List[Atom]:
    """Linear D(inf)h FeCl2 (gas phase), Fe at the origin, Cl on the z axis.

    r(Fe-Cl) = 2.151 A, the gas-phase electron-diffraction value. ⚠ **The axis matters and
    it is z**: the ground state is the |Lambda = +-2, Sigma = +-2> pair of a 5-Delta term,
    whose moment lies along the molecular axis, and the spinor conventions quantize spin
    along z (a symmetry axis that is not z is reported and not used).
    """
    return [("Fe", (0.0, 0.0, 0.0)), ("Cl", (0.0, 0.0, r)), ("Cl", (0.0, 0.0, -r))]


def _ti3_far(sep: float = 25.0) -> List[Atom]:
    """Three bare Ti(3+) ions at ``sep`` Angstrom along x — the d^1 trimer that FITS.

    The fallback the memory refusal of ``ti3f9_far`` forced (14.8 GB against an 11 GB
    machine): free ions instead of fluorides. What survives is exactly what the trimer
    rung exists for — three d^1 sites, a topology beyond one bond, a 1000-state product
    manifold, dense-CI checkable — and the free-ion j = 3/2 / j = 5/2 site multiplets
    add the spherical-degeneracy stress the ladder's ce3p/yb3p tradition values.
    """
    return [("Ti", (0.0, 0.0, 0.0)), ("Ti", (sep, 0.0, 0.0)),
            ("Ti", (2.0 * sep, 0.0, 0.0))]


def _ti3f9_far(sep: float = 25.0) -> List[Atom]:
    """Three TiF3 monomers at ``sep`` Angstrom along x — the d^1 TRIMER, far limit.

    Fluoride, not chloride, by memory arithmetic on the integral path: Ti3Cl9 is
    270 AOs (ERI array ~4.9 GB plus the transform's B^P factors, past the 15 GB dev box),
    Ti3F9 is 234 AOs and fits. The physics under test — three d^1 sites, a 10^3-state
    local-multiplet product manifold, topology beyond a single bond — is identical.
    """
    return (_tif3(origin=(0.0, 0.0, 0.0)) + _tif3(origin=(sep, 0.0, 0.0))
            + _tif3(origin=(2.0 * sep, 0.0, 0.0)))


# --- system record ---------------------------------------------------------------------
@dataclass(frozen=True)
class System:
    """One validation system and the calculation protocol applied to it in both tiers.

    Attributes
    ----------
    key, label : str
        Stable identifier used as the JSON key, and a human-readable name.
    atoms, charge, spin : geometry / state
        ``spin`` is 2S (PySCF convention) of the *scalar* reference state.
    basis, basis_matched : str
        See the module docstring's basis policy.
    ncas, nelecas : int
        Active space: number of active *spatial* orbitals and active electrons (scalar
        picture). The spinor-basis equivalent Kuiva will use is ``2 * ncas`` spinors.
    active_l : str
        Angular-momentum letter (``"f"``, ``"d"``, ...) whose Loewdin population selects the
        active orbitals, or ``""`` to take the plain frontier window ``[ncore, ncore+ncas)``.

        Deterministic selection matters: it is what makes the reference reproducible by an
        independent implementation. Note it is the *angular momentum*, not an AO label like
        ``"4f"``: PySCF's principal-quantum-number labels count shells within the basis, so
        in a **segmented** set (Karlsruhe) ``"6p"`` is merely the fifth p shell and lands on
        core orbitals, while in a **general** set (ANO-RCC) it is the true valence 6p. Keying
        on the label would make the active space silently basis-dependent - a live instance
        of the contraction-type trap the basis policy warns about. Angular momentum is contraction-blind.
    active_skip_pairs : int
        ⚠ **How many Kramers pairs of that character to SKIP, and it is not optional where it
        is nonzero.** A plain ``(atom, l)`` selection takes the *lowest* pairs of that
        character, which is the valence shell only when nothing of the same ``l`` is filled
        below it. That holds for every ``d`` and ``f`` system here (3d and 4f are the lowest
        of their kind) and for boron's 2p - and for nothing else in the np^1 series: Al's 3p^1
        selects 2p, Ga's 4p^1 selects 2p, and so on down to Tl.

        ⚠ **The failure is silent in both directions that matter.** The calculation converges
        and reports an ordinary ^2P doublet - Ga's came out at 249 400 cm^-1 against an
        experimental 826 - and the *g values do not notice*, because a p^1 shell is Lande 2/3
        wherever it sits. A study whose only observable is ``g`` therefore cannot verify its
        own active space. Worse, the CASSCF sometimes repairs the wrong guess and sometimes
        does not: Al's 2p start relaxed to the valence answer, Ga's did not.

        Consume it through :func:`character_selection`, never by rebuilding the argument, and
        check the result against :data:`EXPERIMENTAL_SPLITTING_CM` where that is defined.
    nroots : dict
        ``{multiplicity: n_roots}`` for the state-averaged calculation; multiplicity is
        2S+1. The counts are the *complete* spin-free manifolds of the active space, which
        is what makes the degeneracy patterns meaningful.
    soc_states : int
        Number of spin-orbit states expected after coupling (sum of mult * nroots).
    scalar_nroots : dict or None
        ⚠ **The ensemble the Tier-1 SCALAR cross-check averages, which is not always
        ``nroots``.** That cross-check runs Kuiva with SOC off and compares against PySCF, so
        the two have to average the *same* set — but PySCF's average is spin adapted and
        Kuiva's is "the lowest N spinor roots", and those coincide only while the terms in
        ``nroots`` are the lowest states of the CAS **including the multiplicities PySCF never
        computes**. Where they do not (``dy3p``: quartets fall below the ⁶P term), this carries
        a smaller, complete sub-ensemble that they do share, and the record generator produces
        a second reference for it. ``None`` means ``nroots`` already works.
    tier2 : tuple of str
        Which external codes generate a Tier-2 reference for this system.
    tier1 : bool
        Whether the system gets a scalar Tier-1 record (the default, and
        ``test_tier1_pyscf`` asserts completeness over exactly these). ``False`` is for
        systems whose scalar counterpart is *derived* rather than generated — e.g.
        ``ti3f9_far``, whose scalar physics is three non-interacting copies of ``tif3``:
        a direct record would be a ~375-root scalar SA-CASSCF, exactly the
        half-day-reference trap the d^1-vs-f^1 note above exists to prevent.
    slow : bool
        Excluded from the default (laptop-fast) suite; the default suite stays laptop-fast.

        The two ``ti2cl6*`` entries are the expensive ones, and "expensive" here means
        order 10 CPU-minutes per Tier-1 record, not hours - that upper bound is a deliberate
        design constraint on this suite, see the d^1-vs-f^1 note above. Judge cost by the
        ``cpu_seconds`` recorded in the reference files, **not** by wall time: this
        development machine runs ``thermald`` in adaptive mode and clamps the CPU under
        sustained load, so wall time here can be a large multiple of the actual compute
        (see :mod:`thermal`).
    """
    key: str
    label: str
    atoms: List[Atom]
    charge: int
    spin: int
    basis: str
    basis_matched: str
    ncas: int
    nelecas: int
    active_l: str = ""
    active_skip_pairs: int = 0
    nroots: Dict[int, int] = field(default_factory=lambda: {1: 1})
    soc_states_override: Optional[int] = None
    scalar_nroots: Optional[Dict[int, int]] = None
    tier2: Tuple[str, ...] = ()
    tier1: bool = True
    slow: bool = False
    geom_note: str = ""
    physics_note: str = ""

    @property
    def elements(self) -> Tuple[str, ...]:
        return tuple(sorted({s for s, _ in self.atoms}))

    @property
    def soc_states(self) -> int:
        """Spinor roots to average over.

        ⚠ **``sum(mult * nroots)`` is a SPIN-FREE count, and a two-component state average
        must land on a SPINOR manifold boundary instead.** The two coincide only while the
        spin-free terms being counted stay below everything they do not include. Where they
        do not, ``soc_states_override`` carries the measured boundary and the system's note
        says why — see ``dy3p``, where the difference was a 44.85 cm^-1 splitting of
        multiplets that spherical symmetry makes exactly zero.
        """
        if self.soc_states_override is not None:
            return int(self.soc_states_override)
        return sum(mult * n for mult, n in self.nroots.items())

    @property
    def scalar_ensemble(self) -> Dict[int, int]:
        """``{multiplicity: n_roots}`` the **scalar** Tier-1 cross-check averages."""
        return dict(self.scalar_nroots if self.scalar_nroots is not None else self.nroots)

    @property
    def scalar_states(self) -> int:
        """Spinor roots the **scalar** Tier-1 cross-check averages.

        ⚠ **Not ``soc_states``, and the difference is not cosmetic.** ``soc_states`` is a
        boundary of the *two-component* spectrum, whose manifolds are ``2J+1`` multiplets;
        with SOC switched off the manifolds are spin multiplets instead, and a count measured
        on one is generally not a boundary of the other. Using ``soc_states`` here cut a
        4-fold block of ``dy3p``'s spin-free spectrum exactly in half, which broke Kramers
        degeneracy during the optimization — the mirror image of the C1 defect, and invisible to
        the odd-block state-averaging gate because half of four is even.
        """
        return sum(mult * n for mult, n in self.scalar_ensemble.items())

    @property
    def nelectron(self) -> int:
        from kuiva.basis import registry as reg
        return sum(reg.z_of(s) for s, _ in self.atoms) - self.charge


# --- the suite -------------------------------------------------------------------------
SYSTEMS: Tuple[System, ...] = (
    System(
        key="ne", label="Ne", atoms=_atom("Ne"), charge=0, spin=0,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=4, nelecas=6, active_l="",
        physics_note="cheapest end-to-end smoke test (SCF + CASSCF + SC-NEVPT2)",
    ),
    System(
        key="zn2p", label="Zn(2+)", atoms=_atom("Zn"), charge=2, spin=0,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=5, nelecas=10, active_l="",
        physics_note="3d^10: CASCI over a *full* active space must equal the SCF energy "
                     "exactly - an invariance check that no correct code can fail",
    ),
    System(
        key="hi", label="HI", atoms=_diatomic("H", "I", 1.609), charge=0, spin=0,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=2, nelecas=2, active_l="",
        geom_note="r(H-I) = 1.609 A, experimental r_e (Huber & Herzberg 1979)",
        physics_note="light-heavy bond; molecular sigma/sigma* CAS at minimal cost",
    ),
    System(
        key="bi", label="Bi", atoms=_atom("Bi"), charge=0, spin=3,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=3, nelecas=3, active_l="",
        nroots={4: 1, 2: 8},
        tier2=("molcas", "dirac"),
        physics_note="6p^3: multireference and strongly spin-orbit coupled. Spin-free terms "
                     "4S(1) + 2D(5) + 2P(3); 20 SOC states = C(6,3). Experimental levels "
                     "4S3/2 0, 2D3/2 11419, 2D5/2 15438, 2P1/2 21661, 2P3/2 33165 cm-1 "
                     "(NIST ASD)",
    ),
    # --- the np^1 isoelectronic series -----------------------------------------------------
    # ⚠ **One valence p electron all the way down group 13**, so the g factor of each level is
    # purely angular and the analytic Lande values 2/3 and 4/3 hold exactly for every member.
    # That makes the series a controlled scan of a *relativistic* effect against Z with
    # everything else held fixed - which is what a single atom can never establish, however
    # precisely it is computed. All five are sub-second at CAS(1, 3 spatial) with six roots.
    System(
        key="b", label="B", atoms=_atom("B"), charge=0, spin=1,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=3, nelecas=1, active_l="p", nroots={2: 3}, tier1=False,
        physics_note="2p^1, Z=5: the light end of the np^1 series. 2P1/2 (2) below 2P3/2 (4), "
                     "analytic Lande g = 2/3 and 4/3, splitting ~33 cm-1",
    ),
    System(
        key="al", label="Al", atoms=_atom("Al"), charge=0, spin=1,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=3, nelecas=1, active_l="p", active_skip_pairs=3,  # 2p lie below
        nroots={2: 3}, tier1=False,
        physics_note="3p^1, Z=13, np^1 series; experimental 2P splitting 112 cm-1 (NIST ASD)",
    ),
    System(
        key="ga", label="Ga", atoms=_atom("Ga"), charge=0, spin=1,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=3, nelecas=1, active_l="p", active_skip_pairs=6,  # 2p+3p lie below
        nroots={2: 3}, tier1=False,
        physics_note="4p^1, Z=31, np^1 series; experimental 2P splitting 826 cm-1 (NIST ASD)",
    ),
    System(
        key="in", label="In", atoms=_atom("In"), charge=0, spin=1,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=3, nelecas=1, active_l="p", active_skip_pairs=9,  # 2p+3p+4p lie below
        nroots={2: 3}, tier1=False,
        physics_note="5p^1, Z=49, np^1 series; experimental 2P splitting 2213 cm-1 (NIST ASD)",
    ),
    System(
        key="tl", label="Tl", atoms=_atom("Tl"), charge=0, spin=1,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=3, nelecas=1, active_l="p", active_skip_pairs=12,  # 2p+3p+4p+5p lie below
        nroots={2: 3}, tier1=False,
        physics_note="6p^1, Z=81, the heavy end of the np^1 series and where a relativistic "
                     "property correction should be largest; experimental 2P splitting "
                     "7793 cm-1 (NIST ASD)",
    ),
    System(
        key="tlh", label="TlH", atoms=_diatomic("Tl", "H", 1.872), charge=0, spin=0,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=4, nelecas=2, active_l="",
        nroots={1: 3, 3: 3},
        tier2=("molcas", "dirac"),
        geom_note="r(Tl-H) = 1.872 A, experimental r_e (Huber & Herzberg 1979)",
        physics_note="molecular heavy diatomic; sigma, sigma* and the Tl 6p pi orbitals give "
                     "strongly SOC-split Pi states - the classic two-component benchmark",
    ),
    System(
        key="ce3p", label="Ce(3+)", atoms=_atom("Ce"), charge=3, spin=1,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=7, nelecas=1, active_l="f",
        nroots={2: 7},
        tier2=("molcas", "dirac"),
        physics_note="4f^1, the minimal f-element problem. 7 degenerate spin-free 2F states "
                     "split by SOC into 2F5/2 (6) + 2F7/2 (8); experimental splitting "
                     "2253 cm-1. Analytic Lande g(2F5/2) = 6/7",
    ),
    System(
        key="yb3p", label="Yb(3+)", atoms=_atom("Yb"), charge=3, spin=1,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=7, nelecas=13, active_l="f",
        nroots={2: 7},
        tier2=("molcas", "dirac"),
        physics_note="4f^13, the one-hole counterpart of Ce(3+): ground 2F7/2 (8) below "
                     "2F5/2 (6), i.e. the *inverted* multiplet, experimental splitting "
                     "10214 cm-1. Analytic Lande g(2F7/2) = 8/7",
    ),
    System(
        key="dy3p", label="Dy(3+)", atoms=_atom("Dy"), charge=3, spin=5,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=7, nelecas=9, active_l="f",
        nroots={6: 21}, soc_states_override=134, scalar_nroots={6: 11},
        tier2=("molcas",),
        physics_note="4f^9, strongly correlated many-electron f shell and the archetypal SMM "
                     "ion. Sextet terms 6H(11) + 6F(7) + 6P(3) = 21 roots; SOC ground state "
                     "6H15/2 (16-fold) with analytic Lande g = 4/3. "
                     "⚠ AVERAGED OVER 134 SPINOR ROOTS, NOT 126: the spin-free sextet count "
                     "is not a spinor manifold boundary here. Roots 119-134 form a single "
                     "16-fold manifold, so a 126-state average cuts it exactly in half, the "
                     "averaged density stops being spherical, and the 6H15/2 ground manifold "
                     "splits by 44.85 cm^-1. At 134 the boundary sits in a 2058 cm^-1 gap and "
                     "the worst spread is 0.005 cm^-1 - a 9000x reduction. This was "
                     "the recorded critical defect C1 (a state average cutting a degenerate manifold). "
                     "⚠ AND THE SCALAR CROSS-CHECK AVERAGES 11 ROOTS (66 SPINOR STATES), NOT "
                     "EITHER OF THOSE: with SOC off the manifolds are spin multiplets, where "
                     "134 sits inside the 4-fold block of roots 133-136 and 126 inside 125-128. "
                     "Worse, quartet terms fall below the 6P here (25683 vs 25914 cm^-1 at the "
                     "reference orbitals), so no count of lowest spinor roots reproduces "
                     "PySCF's spin-adapted 21-sextet average at all. The 6H term alone is both "
                     "a complete manifold - a 5623 cm^-1 boundary gap - and an ensemble both "
                     "codes can average identically",
    ),
    System(
        key="cecl3", label="CeCl3", atoms=_cecl3(), charge=0, spin=1,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=7, nelecas=1, active_l="f",
        nroots={2: 7},
        tier2=("molcas",), slow=True,
        geom_note="planar D3h, r(Ce-Cl) = 2.569 A (gas-phase structure)",
        physics_note="single f^1 site in a ligand field: the 2F5/2 sextet splits into three "
                     "Kramers doublets. Tests crystal-field splitting on top of SOC",
    ),
    System(
        key="ticl3", label="TiCl3", atoms=_ticl3(), charge=0, spin=1,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=5, nelecas=1, active_l="d",
        nroots={2: 5},
        tier2=("molcas",),
        geom_note="planar D3h, r(Ti-Cl) = 2.25 A (gas-phase structure)",
        physics_note="single d^1 site in a ligand field: D3h splits the d shell into "
                     "a1' + e'' + e', giving 5 Kramers doublets once SOC is on. This is the "
                     "monomer the ti2cl6_far spectrum must factorise back into",
    ),
    System(
        key="ti2cl6", label="Ti2Cl6 (dimer)", atoms=_ti2cl6(), charge=0, spin=0,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=10, nelecas=2, active_l="d",
        nroots={1: 25, 3: 25},
        tier2=("molcas",), slow=True,
        geom_note="fixed D2h model: Ti...Ti 3.50 A, Ti-Cl(bridge) 2.45 A, "
                  "Ti-Cl(terminal) 2.20 A (NOT an optimised structure)",
        physics_note="MULTI-SITE: two coupled d^1 centres. 25 singlet + 25 triplet spin-free "
                     "roots span the one-electron-per-site manifold, giving 25 + 3*25 = 100 "
                     "SOC states = the full 10 x 10 local-multiplet product. This is the "
                     "structure dmrg/manifold.py must reproduce",
    ),
    System(
        key="tif3", label="TiF3", atoms=_tif3(), charge=0, spin=1,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=5, nelecas=1, active_l="d",
        nroots={2: 5},
        geom_note="planar D3h, r(Ti-F) = 1.780 A (gas-phase-like model structure)",
        physics_note="single d^1 site in a fluoride ligand field: the compact monomer "
                     "whose trimer (ti3f9_far) fits the memory the chloride trimer "
                     "cannot. Same 5-Kramers-doublet structure as ticl3",
    ),
    System(
        key="fecl2", label="FeCl2 (linear)", atoms=_fecl2(), charge=0, spin=4,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=5, nelecas=6, active_l="d",
        nroots={5: 5}, slow=True,
        geom_note="linear D(inf)h, r(Fe-Cl) = 2.151 A (gas-phase electron diffraction); "
                  "the molecular axis is z, which is where the spinor conventions put the "
                  "spin quantization axis",
        physics_note="⚠ THE INTEGER-SPIN SYSTEM, and the only one: every other reference "
                     "here is odd-electron and therefore Kramers protected. Fe(2+) d^6 in a "
                     "linear field gives a 5-Delta ground term, and spin-orbit coupling "
                     "splits it into the |Omega| = 4, 3, 2, 1, 0 ladder whose lowest member "
                     "is a NON-KRAMERS doublet: degenerate because the field is axial, not "
                     "because time reversal says so. Two things follow that no odd-electron "
                     "system can test. The transverse g values are ZERO (a Kramers doublet "
                     "always has a transverse moment; this pair cannot, since mu_x connects "
                     "|Omega| = 4 to |Omega| = 3), and g_z has an analytic target: "
                     "2 (Lambda + g_e Sigma) = 12.009 for a pure |+-2, +-2>, measured "
                     "12.0075 here. Break the axis and the pair splits by a tunnelling gap, "
                     "which is the quantity a Tb or Ho single-molecule magnet is about.",
    ),
    System(
        key="ti3_far", label="Ti(3+) x3 (25 A)", atoms=_ti3_far(25.0), charge=9, spin=3,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=15, nelecas=3, active_l="d",
        nroots={4: 125, 2: 250}, soc_states_override=1000,
        tier1=False, slow=True,
        geom_note="three bare Ti(3+) ions 25 A apart along x",
        physics_note="the d^1 trimer that fits the dev box (see ti3f9_far, whose memory "
                     "refusal is the recorded reason this variant exists): three "
                     "free-ion d^1 sites, ground j = 3/2 quartet per site, so the "
                     "ground product manifold is 4^3 = 64-fold and the full local space "
                     "10^3 = 1000. Additivity and dense CI (4060 determinants) are the "
                     "oracles; no direct Tier-1 record (additivity from a single ion)",
    ),
    System(
        key="ti3f9_far", label="Ti3F9 (25 A)", atoms=_ti3f9_far(25.0), charge=0, spin=3,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=15, nelecas=3, active_l="d",
        nroots={4: 125, 2: 250}, soc_states_override=1000,
        tier1=False, slow=True,
        geom_note="three TiF3 monomers (identical to the tif3 system) 25 A apart along x",
        physics_note="the d^1 TRIMER, far limit (P6.1 rung 3): the first system "
                     "whose topology is not a single bond, with a 10^3 = 1000-state "
                     "local-multiplet product manifold — the state-count exercise "
                     "beyond a dimer, still CI-checkable (C(30,3) = 4060 determinants "
                     "diagonalized densely). ADDITIVITY: the spectrum must factorise "
                     "into sums over three identical monomer spectra. No direct Tier-1 "
                     "record (tier1=False): its scalar counterpart is the tif3 monomer "
                     "by additivity, and a direct record would be a ~375-root scalar "
                     "SA-CASSCF — the half-day-reference trap",
    ),
    System(
        key="ti2cl6_far", label="Ti2Cl6 (25 A)", atoms=_ti2cl6_far(25.0), charge=0, spin=0,
        basis="x2c-SVPall-2c", basis_matched="ano-rcc-vdzp",
        ncas=10, nelecas=2, active_l="d",
        nroots={1: 25, 3: 25},
        tier2=("molcas",), slow=True,
        geom_note="two TiCl3 monomers (identical to the ticl3 system) 25 A apart",
        physics_note="ADDITIVITY: at 25 A the 100 SOC states must factorise into the tensor "
                     "product of two monomer multiplets, E = E_A + E_B, giving 15 blocks - "
                     "5 of size 4 (both sites on the same local level) and 10 of size 8 "
                     "(sites on different levels). The sharpest available test of local "
                     "multiplets and size consistency",
    ),
)

SYSTEMS_BY_KEY: Dict[str, System] = {s.key: s for s in SYSTEMS}


#: Experimental ^2P splittings of the np^1 series (NIST ASD; the same numbers each system's
#: ``physics_note`` quotes), cm^-1. ⚠ **The guard against a core-shell selection**: a computed
#: splitting that is not within a small factor of these did not land on the valence shell, and
#: it is the only cheap check that catches it — the g values cannot.
EXPERIMENTAL_SPLITTING_CM: Dict[str, float] = {
    "b": 33.0, "al": 112.0, "ga": 826.0, "in": 2213.0, "tl": 7793.0,
}


def character_selection(system: System, *, centres=None) -> Dict:
    """How this system's active space is stated to ``api.casscf`` / ``api.casci``.

    ⚠ **The one place the ordinal window is applied**, so a consumer cannot forget it. With
    ``active_skip_pairs`` zero this is the plain ``character=(atom, l)`` form; with it nonzero
    the selection becomes the fragment-list form ``[(atom, l, n_spinors, skip)]``, which is the
    *only* way to name a valence shell that has filled shells of the same ``l`` below it (see
    :class:`System`). Rebuilding either form at a call site is how the np^1 series came to be
    measured on core 2p orbitals.

    ``centres`` overrides which atoms carry the character; the default is every atom of the
    active element, which is what a multi-centre active space needs.
    """
    if not system.active_l:
        raise ValueError(
            "{} has no active_l: its active space is the frontier spinor window, not a "
            "character selection".format(system.key))
    n_active_spinor = 2 * system.ncas
    if centres is None:
        element = system.atoms[0][0]
        centres = [i for i, (sym, _) in enumerate(system.atoms) if sym == element]
    if system.active_skip_pairs:
        return dict(character=[(centres, system.active_l, n_active_spinor,
                                system.active_skip_pairs)],
                    n_active_elec=system.nelecas)
    return dict(character=(centres, system.active_l), n_active=n_active_spinor,
                n_active_elec=system.nelecas)


def get(key: str) -> System:
    if key not in SYSTEMS_BY_KEY:
        raise KeyError(f"unknown system {key!r}; known: {sorted(SYSTEMS_BY_KEY)}")
    return SYSTEMS_BY_KEY[key]


def for_tier2(code: str) -> Tuple[System, ...]:
    """Systems with a Tier-2 reference from ``code`` ("molcas" or "dirac")."""
    return tuple(s for s in SYSTEMS if code in s.tier2)


def fast() -> Tuple[System, ...]:
    """Systems in the default (laptop-fast) suite."""
    return tuple(s for s in SYSTEMS if not s.slow)


__all__ = ["Atom", "EXPERIMENTAL_SPLITTING_CM", "System", "SYSTEMS", "SYSTEMS_BY_KEY",
           "character_selection", "fast", "for_tier2", "get"]
