# `Reference(scf, **options)`

The multireference starting point, built from a finished [`ScalarSCF`](ScalarSCF.md): the
orthonormal working basis (linear dependence removed), the Kramers-paired spinor guess in
it, and the factorized two-electron integrals. The result container — everything the
multireference layer starts from — is `.reference`, a `SpinorReference`. (The stage is named
`Reference` and the container `SpinorReference` precisely so the two are not confused: the
stage runs the step, the container holds the data.)

**After `.run()`:** `.reference`, `.nspinor`, and the inspection helpers
`.population_analysis()`, `.atomic_reference_charges()`, `.write_molden()`.

## Quick reference

| option | default | meaning |
|---|---|---|
| `threshold` | `1e-7` | overlap-eigenvalue cutoff for linear-dependence removal [[47]](../../references.md#r47) |
| `scheme` | `"canonical"` | the orthogonalization [[44]](../../references.md#r44)[[45]](../../references.md#r45): `"canonical"` (the default and the only one that can drop a dimension), `"symmetric"` (raises rather than return a rank-deficient basis), `"cholesky"` [[49]](../../references.md#r49) (no linear-dependence detection) |
| `cholesky_tol` | `1e-8` | the integral-factorization error bound — ⚠ on the *stored* route only; on the direct route the decomposition already happened in `ScalarSCF`, and a different value given here is reported and not silently applied |
| `orbit_pivots` | `True` | symmetry-orbit pivoting, as on `ScalarSCF` |

Dropping a working-basis vector is a deliberate, **reported** reduction of the variational
space — the summary says how many went. A basis that drops directions drops whole degenerate
groups (see [notation](../../notation.md#degenerate-blocks-and-arbitrary-phases)).

## The integral array is released

⚠ On the stored route this stage **releases the SCF's two-electron integral array** the
moment the factors that replace it exist: nothing downstream reads it again, it is the
largest thing the container holds (`O(nao⁴/8)` — 7.7 GB at 300 basis functions, against
factor rows of 1.2 GB), and releasing it means the memory plan stops charging every later
stage for it. The output says what was released. A script that wants the array afterwards —
an exactness check, a second factorization at another threshold — takes its own handle to
`scf.data.eri` first, or factorizes through
`kuiva.integrals.transform.ThreeIndexAO.from_scalar_data` with `release_eri=False`; a second
factorization of an already-released container is refused with exactly that advice.

## Inspection: is this the active space I meant?

Both helpers report **degenerate blocks, not individual spinors**, by default — a single
spinor's density and populations are basis-dependent inside a degenerate manifold, and block
sums are invariant.

### `population_analysis(level="active", ...)`

Löwdin populations [[44]](../../references.md#r44) of a spinor set:

```python
atomic, orbital = ref.population_analysis(level="active", active=active_columns)
atomic.report(spin_vector=True)          # charges and spin; |s| only by default
orbital.report(tolerance=0.01)           # reduced AO populations, contributions above 1%
```

`level` is `"active"` (default), `"frontier"` (a window around the HOMO–LUMO gap) or
`"all"` (thousands of rows on a heavy element — warns). ⚠ Two things not to misread: the
spin density of a state-averaged Kramers pair is exactly **zero** everywhere (time-reversal
symmetry, not a lost moment — look at a single state instead); and **no atomic charge is
printed**, deliberately — the Löwdin charge was measured sign-wrong on ionic textbook
compounds and was withdrawn from every report [[174]](../../references.md#r174)
(`atomic_charge()` remains as an accessor, a diagnostic and never an oxidation state). The
reduced *orbital* populations are the robust half and are what this is for.

### `atomic_reference_charges()`

The supported charge: populations in occupation-weight-orthogonalized free-atom orbitals,
computed in the molecule's own basis with the same relativistic treatment — measured stable
in sign and to ~0.1 e across bases where Löwdin fails qualitatively. Needs
`atomic_reference=True` on the `ScalarSCF` stage (the free-atom orbitals need the integral
library, which nothing downstream has; forgetting it is reported with exactly that advice).
The per-element reference state is the same default the atomic mean field uses (neutral;
M(3+) on the f block), so one element has one default across the program; an overridden
reference is honoured and warns that its charges are not comparable with default-reference
ones.

### `write_molden(path, columns=..., ...)`

```python
ref.write_molden("active.molden", columns=active_columns,
                 occupation=occ, energy=energies)      # one entry per Kramers pair
```

⚠ **A Kuiva molden file [[176]](../../references.md#r176)[[177]](../../references.md#r177)
does not contain orbitals; it contains spinor densities, decomposed exactly into real
components** [[108]](../../references.md#r108) — a viewer can only draw real orbitals, a
two-component spinor is complex with two components, and the square root of its density is
not expandable in the basis (the tempting per-coefficient square root produces an entirely
plausible, wrong picture). Consequently: each entry is one component of a density (`Sym=`
labels it `<spinors>_c<component>`); the **number of components is the diagnostic** (one
means the spinor really is a real orbital; two or more means no single isosurface represents
it); phases and signs are meaningless; occupations are exact and sum to the electron count;
Kramers partners have identical densities, so one entry per pair is complete. ⚠ h functions
(l = 5) are outside the molden standard and are dropped by default with the discarded weight
recorded in the header; `include_high_l=True` writes them anyway (non-standard — many
viewers will refuse or misread the file).
