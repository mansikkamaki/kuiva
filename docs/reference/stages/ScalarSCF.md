# `ScalarSCF(molecule, **options)`

The front end: a scalar-relativistic (spin-free) X2C SCF [[10]](../../references.md#r10) on
a [`Molecule`](Molecule.md), plus everything ingested with it — the integrals and the full
two-component X2C Hamiltonian that spin–orbit coupling enters through. Options are those of
`kuiva.interface.api.scalar_x2c_reference`, validated eagerly by name.

**After `.run()`:** `.energy` (Eh), `.converged`, `.stable` (⚠ `None` when `stability=` was
not asked for — `None` is neither `True` nor `False`; compare explicitly), `.data` (the
PySCF-free `ScalarX2CData` everything downstream builds on).

## Quick reference

**The Hamiltonian**

| option | default | meaning |
|---|---|---|
| `method` | `"X2C-AMF"` | the Hamiltonian by name: `"X2C-AMF"`, `"X2C-1e"`, `"X2C-AMF-DLU"`, `"X2C-1e-DLU"`, `"X2C-mmf"` — resolves to the two axes below; a name and an axis that contradict are refused |
| `x2c_approx` | `"1e"` | the one-electron decoupling: `"1e"` (exact molecular), `"1e-dlu"` (local/DLU [[27]](../../references.md#r27), warns), `"atom1e"` |
| `screening` | `"x2camf"` | the two-electron spin–orbit picture change: `"x2camf"` (X2CAMF [[16]](../../references.md#r16)), `"none"`, `"mmf"` (benchmark only, warns), `"x2camf-external"` |
| `screening_options` | `None` | dict: `interaction` (`"coulomb"` default, `"gaunt"`, `"breit"`), `backend`, `uncontract`, … |
| `decoupling_options` | `None` | dict tuning the DLU path: `partition` (`"atoms"`, `"single"`), `source` (`"diagonal"`, `"isolated"`) |
| `with_soc` | `True` | `False` ingests no two-component Hamiltonian at all: a scalar-relativistic calculation, warned about every time |
| `configuration` | per-element default | atomic reference states for the mean field **and** the atomic-reference charges: `{atom: "+3"}` or `{atom: "[Xe]4f1"}`, keyed by element / label / 1-based number |

**SCF convergence** (checked at construction through the same implementation that applies
them)

| option | default | meaning |
|---|---|---|
| `reference` | `"auto"` | `"rhf"`, `"rohf"`, `"uhf"`; `"auto"` is RHF closed-shell, ROHF otherwise. RHF on an open shell is refused, never promoted |
| `conv_tol` | `1e-10` | SCF energy convergence |
| `max_cycle` | `200` | SCF iteration budget |
| `level_shift` | `0.0` | energy added to the virtuals — stops occupations swapping |
| `damp` | `0.0` | mixes the previous Fock matrix in; slower, steadier |
| `diis` | `"cdiis"` | `"adiis"`, `"ediis"`, or `False`; with `diis_space=`, `diis_start_cycle=` |
| `init_guess` | `"minao"` | `"atom"`, `"1e"`/`"hcore"`, `"huckel"`, `"mod_huckel"`, `"sap"` — ⚠ an unrecognized name is **refused**, not silently substituted |
| `second_order` | `False` | the CIAH Newton solver; does not use the four levers above, and asking for both warns |
| `stability` | `None` | `"check"` runs the internal stability analysis; `"follow"` also rotates into the unstable mode and re-solves, up to three times |
| `allow_unconverged_scf` | `False` | ⚠ an SCF that runs out of cycles **refuses**; this proceeds deliberately and warns |

**The starting density** (three mutually exclusive ways)

| option | default | meaning |
|---|---|---|
| `init_guess` | `"minao"` | a standard starting density (above) |
| `guess_from` | `None` | a finished `ScalarSCF` (or its data): same basis — used as-is; different basis — projected [[50]](../../references.md#r50) |
| `broken_symmetry` | `None` | `{"Fe1": +5, "Fe2": -5}`: signed unpaired-electron counts per centre; builds the spin-polarized guess [[196]](../../references.md#r196)[[197]](../../references.md#r197); needs `reference="uhf"`; `bs_min_population=` is the localization floor |

**The two-electron route**

| option | default | meaning |
|---|---|---|
| `fitting` | `"auto"` | `"cholesky"` (store the ERI array, factorize, release), `"cholesky-direct"` (never form the array), `"df"` (density fitting); `"auto"` lets the memory plan decide |
| `auxbasis` | `None` | the DF auxiliary basis (giving one selects DF); ⚠ a Coulomb-fitting auxiliary warns — not accurate enough for correlated integrals |
| `cholesky_tol` | `1e-8` | the Cholesky error bound [[74]](../../references.md#r74) — decided on *relative* energies; see below |
| `orbit_pivots` | `True` | pivot on complete symmetry orbits [[78]](../../references.md#r78), keeping the factorization exactly rotationally invariant; off = plain pivoting, for measurement only |
| `one_centre` | `True` | the one-centre (atomic) decomposition construction |
| `factors` | `"auto"` | where the finished factor rows live: `"in-core"`, `"scratch"` (spill + stream back, bitwise), `"streamed"` (out-of-core decomposition, ⚠ not bitwise), `"auto"` |

**Memory and planning**

| option | default | meaning |
|---|---|---|
| `memory_gb` | configured | the memory limit for this calculation; with no configured value either, the run refuses to start |
| `n_active`, `n_active_elec`, `n_states`, `nevpt2` | `None`/`False` | *planning only*: sharpen the pre-flight with the multireference stage's dimensions, so a CAS the CI cannot hold is refused before the SCF is paid for. The CASSCF still states its own space |

**Properties, analysis, symmetry**

| option | default | meaning |
|---|---|---|
| `gauge_origin` | centre of mass | where `L` is defined about — fixed **here**, at ingestion; five forms below |
| `property_picture_change` | `False` | picture-change-transform the property operators (`L+2S` and `r` together, never one alone) |
| `anomaly_picture_change` | `False` | the further small-component term of the `g_e − 2` anomaly |
| `hyperfine` | `None` | build the hyperfine field operator of the **named** nuclei — never a default, and always picture-changed whatever `property_picture_change` says; see below |
| `atomic_reference` | `False` | compute the free-atom reference orbitals at ingestion — required for AVAS and the atomic-reference charges |
| `point_group` | from the `Molecule` | abelian double-group symmetry; `classification=` likewise (see [Molecule](Molecule.md)) |

## The Hamiltonian, by name

| method | one-electron decoupling | two-electron picture change | use it for |
|---|---|---|---|
| **`X2C-AMF`** *(default)* | exact, molecular | atomic mean field | **everything** |
| `X2C-AMF-DLU` | local (atom-blocked, DLU) | atomic mean field | systems where the exact decoupling is prohibitive |
| `X2C-1e` | exact, molecular | none | SOC-free comparisons, diagnostics |
| `X2C-1e-DLU` | local | none | diagnostics; doubly approximate |
| `X2C-mmf` | exact, molecular | **molecular** mean field | ⚠ benchmark only |

A name resolves to the two independent axes `x2c_approx` and `screening`, which may also be
set directly; a valid combination with no canonical name is synthesized, so the provenance
is never empty. ⚠ Selecting a DLU method warns — it is the escape hatch for systems where
the exact decoupling will not fit, not a cheaper default.

**Two one-electron operators, deliberately.** The SCF that produces the starting orbitals
always runs on the *spin-free* X2C operator (unaffected by both knobs); the correlated
Hamiltonian — whose expectation value is the energy — uses the full two-component one,
selected by `x2c_approx`, with the two-electron picture change selected by `screening`.
Orbitals are a basis the CASSCF re-optimizes, so the scalar set is a guess.

**`screening` values.** `"x2camf"` (default) adds the atomic mean-field picture change —
without it, atomic j-splittings come out 5–30% too large. It costs one four-component atomic
solve per unique element, cached persistently (see
[configuration](../../guide/configuration.md#the-atomic-mean-field-cache)). `"none"` is a
statement about cost, not correctness — right for anything not about spin–orbit coupling
(⚠ it does **not** turn SOC off; that is `with_soc=False`). `"mmf"` takes the same
subtraction from a full molecular four-component SCF — the two are values of one axis and
cannot be combined, because combining them would double-count the identical correction; its
cost grows as the fourth power of the basis, which is why X2CAMF exists.
`"x2camf-external"` is the original authors' plugin [[18]](../../references.md#r18), a
bisection tool (always neutral-atom reference; refuses a finite nucleus).

**Gaunt and Breit.** The four-component reference behind `"x2camf"`/`"mmf"` uses the
Dirac–Coulomb operator by default; `screening_options={"interaction": "gaunt"}` (or
`"breit"`) changes it [[20]](../../references.md#r20). ⚠ The interaction is **not** part of
the Hamiltonian's name — what records it is the screening record's `interaction` field in
`scf.data.soc.provenance()`, which every stored product carries. When two calculations
disagree, compare that field, not the method line.

⚠ `scf.data.soc.screening` states what the Hamiltonian already **contains** — it is a
record, not a request. The correction is applied in exactly one place, in the AO basis,
before any change of basis; anything adding a correction to a Hamiltonian whose record is
not `"none"` double-counts it.

## When the SCF will not converge

This is where a real calculation first stops: an ROHF on a metal ion with several shells
within an eV of each other oscillates between occupations, and no `max_cycle` fixes it. The
levers, roughly in the order worth trying: `level_shift=0.2`; `damp=0.5`; `diis="adiis"`
(much better in the first iterations of a hard open-shell case than the default Pulay
commutator DIIS); `init_guess="atom"`; and `second_order=True`, which converges cases the
DIIS iteration cannot, at more cost per iteration and far fewer of them.

```python
scf = kuiva.ScalarSCF(complex_, level_shift=0.2, diis="adiis").run()
scf = kuiva.ScalarSCF(complex_, second_order=True).run()      # when that is not enough
```

⚠ An SCF that does not converge **refuses** rather than handing on whichever iterate the
budget stopped at — everything downstream is built on those orbitals, and "the CASSCF
re-optimizes them" is a hope, not a property. `allow_unconverged_scf=True` proceeds
deliberately and warns.

## Is the converged solution a minimum at all?

`stability="check"` runs the internal stability analysis on the converged SCF;
`"follow"` also rotates into the unstable mode and re-solves. Off by default (it costs a
Davidson over the orbital Hessian), and worth that cost whenever the reference is in doubt:
⚠ an unstable SCF is a **saddle point** — converged flag, small gradient, plausible energy,
and a lower solution of the same reference one rotation away. Measured on a Ni atom: the
ROHF converges cleanly, and `stability="follow"` lands 0.30 Eh lower.

`scf.stable` is `True`, `False`, or `None` (not measured) — ⚠ compare explicitly; `if not
scf.stable` reads "never measured" as "unstable". External stability (RHF → UHF, real →
complex) is deliberately not run: the answer to it is a different `reference=`, which is
your decision.

## Starting from another calculation

`guess_from=` and `broken_symmetry=` are covered, with the measured caveats (a warm start
can land on a different — unstable — SCF solution; pair it with `stability=`), in
[workflows](../../guide/workflows.md#carrying-orbitals-between-calculations) and
[workflows: broken symmetry](../../guide/workflows.md#antiferromagnetically-coupled-centres-broken-symmetry).

## The gauge origin

Fixed **here**, not at the property dump, because `L` is defined relative to it and the
multireference layer never calls PySCF again. Five forms:

```python
kuiva.ScalarSCF(mol)                                     # centre of mass, the default
kuiva.ScalarSCF(mol, gauge_origin="charge")              # or "origin"
kuiva.ScalarSCF(mol, gauge_origin=("atom", 1))           # 1-based number, or a unique label
kuiva.ScalarSCF(mol, gauge_origin=("angstrom", 0, 0, 1.5))
kuiva.ScalarSCF(mol, gauge_origin=("bohr", 0, 0, 2.835))
```

`("atom", k)` uses the standard per-atom addressing; an element symbol naming *several*
atoms is refused rather than resolved to the first. ⚠ **A bare `(x, y, z)` tuple means
bohr**, and your geometry is in Angstrom — a coordinate copied out of the geometry lands
1.89× too far out, moves the point every orbital moment is defined about, and every number
stays plausible. The bare form warns; the tagged forms say nothing.

## Hyperfine nuclei: `hyperfine=`

Names the nuclei whose **hyperfine field operator** is built at ingestion, in the same
per-atom addressing as `basis` and `configuration` (element symbol, atom label, 1-based
number; most specific wins):

```python
kuiva.ScalarSCF(mol, hyperfine={"Tb1": True})          # the most abundant isotope with I > 0
kuiva.ScalarSCF(mol, hyperfine={"Dy": 163})            # a mass number, or "163Dy"
kuiva.ScalarSCF(mol, hyperfine={"Cu": 63, "Cu2": 65})  # two atoms of one element, differing
kuiva.ScalarSCF(atom, hyperfine=True)                  # a ONE-atom molecule only
```

Values are `True` (the most abundant isotope that has a moment — not simply the most
abundant, since the commonest nuclide of carbon, oxygen and half the transition metals has
`I = 0`), a mass number, an isotope label, or a
`kuiva.util.nuclei.NuclearMoment(spin=…, g=…, quadrupole_barn=…)` for an isomer, a revised
moment, or an element the curated table does not carry.

⚠ **It is never a default and there is no "all magnetic nuclei".** One
four-component-transformed operator is built and *stored* per nucleus, and a complex has
dozens of `1`H, so a bare `True` on anything but a one-atom molecule is refused, as is a
`"default"` key. Ghost atoms, unknown isotopes and `I = 0` are each refused by name.

⚠ **These operators always carry the X2C picture change**, independently of
`property_picture_change`, which keeps governing `mu` and `d` alone: the bare hyperfine
operator is wrong by a factor of 4–10 wherever s character carries spin density
[[209]](../../references.md#r209). That is the one place the "one flag governs both property
operators" rule does not hold, and what replaces its guarantee is that every stored file
states the treatment of **each operator family** separately.

⚠ **Selecting a nucleus warns about what no later number can show.** The isotropic (contact)
part of the coupling comes from core-s spin polarization, which a *valence* active space
does not carry [[212]](../../references.md#r212). For a 4f ion that is minor — the orbital
mechanism dominates — but for s/d spin density, for ligand nuclei and for spin-only ions
(Gd(III), Eu(II), Mn(II)) the isotropic part is qualitatively wrong, by 25% and worse on
transition metals. The remedy is core s shells in the active space, not a correction term.

Three further approximations are recorded in every file rather than hidden: the
transformation uses the **unperturbed** `X` and `R` [[205]](../../references.md#r205) rather
than their response to the nuclear moment [[207]](../../references.md#r207), which is what
makes the result an *operator* usable between different states; no two-electron picture
change is applied to it; and the nuclear magnetization distribution is the nuclear *charge*
distribution of the molecule's own nuclear model [[208]](../../references.md#r208). ⚠ The
isotope chosen here does **not** change that Gaussian exponent, which comes from the integral
library's main-isotope masses — recorded rather than reconciled.

What the operators become downstream is the `[NUCLEI]` table and `T_<k>_u` blocks of
**either** formatted product — [`PropertyDump`](PropertyDump.md) on the conventional-CI route
and [`PseudospinExport`](PseudospinExport.md) on the tensor-network one, written under the
same names, in the same units and with the same table by the same code. Neither has a switch
of its own: naming the nuclei here *is* the request.

## Reference configurations: `configuration=`

One statement per atom, feeding **both** consumers of a reference state — the atomic mean
field and the atomic-reference charges. Keys are element symbols, atom labels or 1-based
numbers (most specific wins); values are an oxidation state (`"+3"`, `2`, `-1`) or an
explicit configuration (`"[Xe]4f1"`, `"1s2 2s2 2p5"`). The default is the neutral atom,
except the f block, which defaults to M(3+); open shells are occupied by average of
configuration, not aufbau. An oxidation state resolves through a curated common-states table
[[31]](../../references.md#r31) to exactly one canonical configuration; an explicit
configuration outside the accepted set warns as unusual; an impossible one is refused. Two
atoms of one element may carry different states (they get decorated labels, `"O1"`/`"O2"`).
⚠ A scalar value on a heteronuclear molecule raises — `"+3"` almost always means "the metal
is trivalent", so say which atom.

## The two-electron route

Which Cholesky route runs is decided by the memory plan by default (`fitting="auto"`): the
stored-array route wherever its plan fits the configured limit, `"cholesky-direct"`
[[76]](../../references.md#r76) where it does not — same decomposition, same threshold, same
error bound, with each integral column evaluated when the pivoting asks for it. The two
routes cost the same CPU within a few per cent beyond ~160 basis functions, so the
`O(nao⁴/8)` array is the entire decision; the output's "two-electron route" line states the
choice and why. Either route changes no result (measured identical on a spin–orbit spectrum,
3e-15 on the integrals). ⚠ On the direct route the decomposition happens **in this stage**,
so `cholesky_tol` and `orbit_pivots` belong here rather than to `Reference`.

`factors=` is the second, independent choice — where the finished factor rows live — a
ladder of two claims: `"scratch"` spills the rows after the decomposition and streams them
back (bitwise identical; frees their RAM for later stages), `"streamed"` runs the
decomposition itself out of core so the factor array is never allocated (⚠ not bitwise: the
same subtractions in a different order, agreeing to rounding, with ties between
symmetry-equivalent columns possibly broken the other way — a different but equally valid
factorization). `"auto"` takes each rung only where it lowers the planned peak. Both need a
configured scratch directory; ⚠ neither applies to DF factors (declined with a warning).

`cholesky_tol` was decided on **relative** energies, where the factorization error cancels
by a measured ~500×. ⚠ An absolute total is bounded by the *basis*, not by this threshold —
tightening to `1e-10` (~25% more vectors) is worth it only to compare two totals in the same
basis.

## Cross-references

The spin-free route (`with_soc=False` vs `screening="none"`):
[workflows](../../guide/workflows.md#turning-spin-orbit-coupling-off). Point-group symmetry
and what it can and cannot promise: the symmetry method page (in preparation) and the
[CASSCF](CASSCF.md) page's per-irrep selection. `atomic_reference=True` feeds AVAS
([CASSCF](CASSCF.md)) and the atomic-reference charges ([Reference](Reference.md)).
