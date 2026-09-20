# The stored products

Everything a run writes, as contracts a consumer can rely on. Three version numbers exist
and are **independent**, because they answer different questions: `kuiva.__version__` (which
code produced a file — provenance, never a compatibility check), `schema_version` (whether a
checkpoint can be read at all), and `format_version` (whether a plain-text product can be
read at all — bumped only when the *meaning* of a stored field changes, never on an
addition, so a consumer can refuse rather than misinterpret).

## The output file

`kuiva.util.logging.add_file_handler(path)` mirrors the run's output stream to a file. It is
the human-readable output and nothing else — machine-readable matrices never enter it. It
reads like a conventional quantum-chemistry output: a banner (version, threads, BLAS
verdict, kernel backend), sections, label/value blocks, fixed-width iteration tables, ASCII
only. `WARNING` and `ERROR` lines are prefixed `*** WARNING [subsystem]` — greppable, and
deliberately breaking the visual flow. Physical quantities print with units at fixed
precision (energies 1e-8 Eh, moments 1e-5 μ_B).

## The property dump

Written by [`PropertyDump`](stages/PropertyDump.md); parsed by `kuiva.read_dump` and
`kuiva.PropertyMatrices.from_dump` (a format nobody has ever read back is a format with an
undetected ambiguity in it). Plain text, line-oriented, `#` comments, `[SECTION]` markers,
one `i j Re Im` record per matrix element — deliberately dull, optimized for being
trivially parseable in any language.

**Sections:** a versioned `[HEADER]`; a `[PROVENANCE]` block of JSON; `[NUCLEI]` when
hyperfine operators were requested; `[ENERGIES]`; one section per matrix — `H`, `mu_x`,
`mu_y`, `mu_z` (μ_B), `d_x`, `d_y`, `d_z` (e·a₀), `T_<k>_x`, `T_<k>_y`, `T_<k>_z` (Eh per
nuclear magneton, per treated nucleus `k`, which is named in the block by its **atom label** —
`T_Tb1_z` — and carried as a row of `[NUCLEI]`), and the bare `L`/`S` blocks when requested;
`[INACTIVE]` with the inactive contributions.

**Header fields a consumer must read:**

| field | meaning |
|---|---|
| `format_version` | refuse an unknown one rather than guess |
| `code_version` | provenance only |
| the provenance JSON | the screening and decoupling records, the nuclear model, the environment digest — a matrix that does not say which Hamiltonian produced it is not interpretable (the screened/unscreened difference is 5–30% on every splitting) |
| the gauge origin | what `L` and `r` are defined about |
| the active space | as a physical statement, never an index window |
| `picture_change_on_properties`, `picture_change_on_dipole` | ⚠ what `mu` and `d` *mean*: the correction changes the operators while `format_version` stays put, so these fields are what distinguish the files — reading the header is obligatory, not optional. In a picture-changed file the `L`/`S` blocks are the bare operators `mu` was *not* built from, written for reference only |
| `molecular_charge`, `dipole_origin_dependence` | ⚠ for a charged molecule the diagonal dipole obeys `d(R_G) = d(0) − q·R_G`; transition elements between distinct states do not move |
| the nuclear dipole vector | so the electronic and nuclear parts stay separable (the nuclear term is on the diagonal only) |
| `hyperfine_unit`, `hyperfine_operator` | what a `T_<k>_u` block *is*: the isotope-independent hyperfine **field** operator, and the one sentence that turns it into an interaction |
| `hyperfine_picture_change`, `hyperfine_decoupling`, `hyperfine_nuclear_model`, `hyperfine_x2c_response`, `hyperfine_2e_picture_change` | ⚠ the treatment of the hyperfine operators, stated **separately** from `mu` and `d` because it is not the same: these are always picture-changed whatever the two fields above say. The rest are the three approximations no number in the file can show — the unperturbed transformation, the absent two-electron picture change, the nuclear magnetization model |
| `n_hyperfine_nuclei` | checked against the `[NUCLEI]` row count; a disagreement is refused, because a `T` matrix matched to the wrong nucleus is Hermitian, plausible and wrong |

The `[NUCLEI]` table is the other half of the hyperfine contract: one whitespace-delimited
row per nucleus — site index, atom number, atom label, element, `Z`, isotope, `2I`, `g_N`,
`Q` in barn, position in bohr, and the source of the nuclear data — in the order the `T`
matrices are written and in the order the consumer is to build its nuclear sites, each a
pseudospin site of dimension `2I+1` with `M_I = −I … +I` ascending. ⚠ `Q` is written `none`
where no **signed** value is tabulated, and must be refused rather than replaced by a
magnitude: the sign of `Q` is the sign of every quadrupole splitting computed from it. A row
whose field count does not match is refused rather than read — that is how a
whitespace-delimited table misparses into plausible numbers.

⚠ **Kuiva never forms the electron–nuclear product space**, which is the whole point of
storing electronic matrices and a nuclear table separately: the isotope, or the subset of
nuclei, can be changed without re-running the electronic calculation, and the product
dimension is *reported* rather than refused because the allocation is the consumer's.

⚠ `H` is **diagonal** (the CI roots are the spin–orbit eigenstates — stated in the header
because a reader from a two-step workflow expects otherwise); ⚠ **phases are arbitrary and
never canonicalized** — validation and comparison go through the phase-invariant reductions
only ([properties](../methods/properties.md#phase-invariant-analysis)). A dump written from
a finished `NEVPT2` records the **hybrid protocol** in the header (`H` from perturbation
theory, `mu` from the CASSCF states).

## The pseudospin export

Written by [`PseudospinExport`](stages/PseudospinExport.md); read by
`kuiva.read_pseudospin` / `kuiva.PseudospinModel.from_file` (an unknown `format_version` is
refused). The same shape as the dump — versioned header, `[SECTION]` markers, `i j Re Im`
records, atomic write, the same provenance obligation (an empty provenance warns) — with
⚠ **one deliberate difference stated in the header: `H` is *not* diagonal.** It is the
effective Hamiltonian over the pseudospin **product** basis; `[ENERGIES]` lists its
eigenvalues and `[MATRIX U]` the diagonalizing unitary (columns = ab initio states over
product-basis rows). The `M` convention and the storage order (per site `M = -S..+S`
ascending, site 0 slowest) are OuluSpin's, restated in every file, so no permutation is
needed on the way in; an applied frame rotation is recorded. Spin operator matrices are
deliberately not written — the format is confirmed against what OuluSpin reads, and nothing
else widens it.

When the reference ingested hyperfine nuclei the file additionally carries `[NUCLEI]` and
`T_<k>_x/y/z` — **the same section, the same twelve columns and the same block names as the
property dump**, written and parsed by the same code, so a consumer meets one vocabulary
whichever file it opened — over the pseudospin product basis and in the frame the header
states. Its header gains the same `hyperfine_*` fields, plus `nuclear_site_order` (nuclear
sites follow the electronic sites, in `[NUCLEI]` order, each `M_I = −I … +I` ascending — the
`[BASIS]` order extended, not a second convention) and `product_dim`, the electron–nuclear
dimension `D × ∏_k (2I_k+1)`. ⚠ `product_dim` is **reported and never refused**, in the header
and above a stated size as a warning: Kuiva does not form that space, which is exactly why the
isotope is the consumer's to change. ⚠ There are no site-projected `T` matrices beside the
site-projected moments — a nucleus feels every electronic site, so no per-site `T` is written
and none is computed.

## The CASSCF checkpoint

Schema-versioned HDF5 [[3]](../references.md#r3), written by
`CASSCF(..., checkpoint=path)`. What is in it, in three tiers:

- **always, every macro-iteration** (cheap, precious): the orbitals and orbital partition,
  the 1-RDM, the converged state energies, the run metadata, the Hamiltonian provenance,
  the **state average** (`n_states` and the requested weights — a restart that disagrees is
  refused), the kernel backend, and a **system fingerprint** — molecule, basis,
  configurations, nuclear model, Hamiltonian axes, gauge origin — which a restart against
  the wrong file is refused on;
- **by policy** (large), dropped in this order when the adaptive budget bites — ⚠ ordered
  by what it costs to get the thing back, not what it costs to store: the 2-RDM (one
  contraction from the vectors), the CI vectors (one CI solve — the Davidson warm start),
  the orbital curvature (not regenerable, only discardable at a few extra
  macro-iterations). ⚠ A **converged** checkpoint inverts the ladder and stores **no
  curvature** — nothing resumes a converged run, and on a large basis the curvature is most
  of the file, while the CI vectors are what makes materialization cheap. The inversion
  keys on convergence, never on a forced write: a deadline stop *will* be resumed, so its
  curvature is kept;
- **never**: three- and four-index integrals, 3-/4-RDMs, and ⚠ transition densities — all
  regenerate deterministically from the orbitals (the expensive atomic solves have their
  own persistent cache), and transition densities are *derived* from the CI vectors already
  stored: two answers to "what is this wavefunction" is one too many.

A DMRG-CASSCF writes a second, sibling `*.network.h5` with the network state, rolling at
the end of each completed sweep under the same cadence knobs; losing it costs a cold first
solve, never correctness.

⚠ **Failure semantics are the opposite of a cache's, on purpose**: a write failure is a
`WARNING` and the run continues (losing a restart point is not losing the calculation); a
read failure on an explicitly requested restart is an `ERROR` that propagates (silently
starting over wastes the hours the file existed to protect). A `schema_version` mismatch
hard-refuses; a changed source fingerprint only warns. The version bump is never what
detects a numerically relevant change — the fingerprint is.

## The NEVPT2 checkpoint

Written by `NEVPT2(..., checkpoint=path)`, per `(state, class)` pair — a class result is a
handful of scalars, so the table is kilobytes whatever the active space, and the only
cadence knob is a minimum interval (rationing filesystem calls, not bytes). ⚠ **It stores
no reference**: the orbitals and CI vectors have one owner (the CASSCF checkpoint) and this
file keeps a *digest* — a restart against a different reference is refused, because inside
a degenerate manifold the CI basis is arbitrary and a barycentre computed across two bases
would belong to neither run ([NEVPT2](stages/NEVPT2.md#checkpointing)).

## Molden files

`Reference.write_molden` writes **spinor densities decomposed exactly into real
components**, not orbitals — a spinor has no isosurface, and the file's own header says
what each entry is. The full reading guide is on
[`Reference`](stages/Reference.md#write_moldenpath-columns-); the format follows the molden
specification [[176]](../references.md#r176)[[177]](../references.md#r177), with h
functions (outside the standard) dropped by default and the discarded weight recorded.

## The Slater-Condon file

`kuiva.extras.slater_condon_parameters(..., file=...)` writes a versioned plain-text file
of $`F^k`$, $`G^k`$, $`R^k`$ and $`\zeta_{nl}`$ ([api](api.md#extras-atomic-slater-condon-parameters)).
Every file states: the radial-phase convention that fixes the sign of the genuine cross
parameters $`R^k`$ ($`P_{nl}(r) \gt 0`$ as $`r \to \infty`$; files with `format_version` < 2 carry
eigensolver-arbitrary $`R^k`$ signs and may not be compared across calculations), the $`R^k`$
index-ordering convention, the frozen average-of-configuration caveat, and whether the
two-electron screening is inside $`\zeta`$.

## The mean-field cache

`~/.cache/kuiva/amf` (or `$KUIVA_AMF_CACHE`) holds the per-element atomic mean-field
corrections — an implementation cache, not a user product: keyed on the parsed basis
content, canonicalized configuration, interaction, nuclear model and speed of light, plus a
formula version bumped whenever the numerical content changes for an unchanged key. Unlike
a checkpoint, it **degrades to a miss on every failure** — a cache that can raise is a
cache that can fail a calculation that would otherwise have run. Deleting it is always
safe; the solves are re-paid, nothing else changes.
