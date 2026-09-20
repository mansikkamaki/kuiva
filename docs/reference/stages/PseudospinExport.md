# `PseudospinExport(casscf, path, ...)`

The second formatted product and the tensor-network property route — a sibling of
[`PropertyDump`](PropertyDump.md), which needs the conventional CI. At the converged CASSCF
orbitals, the active-space Hamiltonian and the three magnetic-moment operators are
compressed onto a **local-multiplet model space**
[[140]](../../references.md#r140)[[141]](../../references.md#r141)[[142]](../../references.md#r142),
the model is assigned pseudospin labels [[179]](../../references.md#r179)
[[180]](../../references.md#r180), and the file for the external OuluSpin code is written:
the effective Hamiltonian and moment operators on the model space, the ordered pseudospin
product basis, and the unitary mapping the ab initio states onto it.

Works from **either** CASSCF solver — the manifold loop re-solves the network from the
integrals either way (warm topology from a DMRG run when available). ⚠ **It takes a
`CASSCF`, not a `CASCI`**: what it consumes is the converged *orbitals*, not the states —
the model space is re-solved from the integrals those orbitals define — so it belongs on the
stage that optimized them, and a CASCI's orbitals are always some other stage's.

**After `.run()`:** `.model` (a `PseudospinModel`), `.g_values` (per site), `.path`.
`kuiva.PseudospinModel.from_file(path)` and `kuiva.read_pseudospin` read files back.
Validation goes through phase-invariant reductions **only**.

## Quick reference

| option | default | meaning |
|---|---|---|
| `path` | — | the output file; written atomically |
| `sites` | `None` | partition of the **active spinors** (by position in the active list) into local multiplet sites, e.g. `[tuple(range(10))]`; `None` discovers the structure from the converged state's entanglement |
| `rule` | `"gap"` | how each site's multiplet space is cut: `"gap"` (the largest group-complete spectral gap, within `min_dim`/`max_dim`), `"dimension"` (exactly `dims`), `"weight"` (smallest cut whose discarded weight is below tolerance) |
| `dims` | `None` | per-site dimensions (one integer or a sequence), consumed by `rule="dimension"` — `dims=2` is "the ground Kramers doublet per site" |
| `max_bond` | solver's | bond-dimension cap for the manifold solves |
| `axes`, `common_axis`, `rotate_frame` | `None`/`False` | the pseudospin quantization axes: per site, one shared, or rotated into the principal frame |
| `g_electron` | CODATA | override the free-electron g factor |
| `seed` | `0` | RNG seed for the manifold loop's starting states (recorded; replayable) |
| `manifold_options` | `{}` | the ensemble-loop knobs (`n_roots`, `max_roots`, `max_outer`, `outer_tol`, …) of `kuiva.dmrg.manifold.solve_manifold` |
| `title`, `comments` | `""`, `()` | header text |
| `report` | `True` | print the model's report |

## Rules and refusals

- ⚠ **Every site must sit in one particle-number sector**, because a pseudospin labels a
  multiplet: a single delocalized electron over several sites is refused. For a single
  centre, one site holding the whole active space is the right form.
- `sites` must partition the active spinor positions **exactly** — a spinor claimed twice or
  left out is refused — and a site boundary cannot cut through a network node: the site
  grouping and the tensor-network topology must agree, and the refusal names the node. Site
  identity is what fragment localization defines
  ([workflows](../../guide/workflows.md#which-centre-an-active-orbital-belongs-to)); a
  site-blocked active space is the natural input.
- Cuts through the site spectra keep **degenerate groups whole** in every `rule`; a cut that
  would split one is refused, not rounded.
- The inactive electrons' moment — exactly zero for a Kramers-paired inactive set, warned
  about otherwise — is added to the total operators after the contraction (a scalar cannot
  be attributed to one site).

## The hyperfine field

⚠ **The hyperfine field operators are written whenever the reference ingested them**
(`hyperfine=` on [`ScalarSCF`](ScalarSCF.md)), with a `[NUCLEI]` table, and there is no
second switch here: naming the nuclei at ingestion *is* the request. They are the three
Cartesian components of the isotope-independent field operator `T` of each treated nucleus,
in Eh per nuclear magneton, over the same pseudospin product basis as `mu` and in the same
frame — the same section names, the same table and the same units the
[property dump](PropertyDump.md) uses, so a consumer reads one vocabulary whichever file it
opened. On the electron–nuclear product space the interaction is
$`H_{hf} = \sum_k g_N(k) \sum_u T_{k,u} \otimes I_{k,u}`$, and everything in that except the
matrices of `T` is nuclear-spin algebra belonging to the external code.

⚠ **Kuiva never forms the electron–nuclear product space.** Storing the electronic matrices
and a nuclear table separately is what lets the isotope, or the subset of nuclei, be changed
without re-running the calculation, so the product dimension
$`D \times \prod_k (2I_k+1)`$ is **reported** — in the header, in the stage summary, and above
a stated size as a warning — and never refused: the allocation is the consumer's.

⚠ **There are no site-projected `T` matrices**, unlike the site-projected moments. OuluSpin
consumes none, and a nucleus feels every electronic site (transferred hyperfine), so a
per-site table would invite being read as "this site's coupling" when what the interaction is
made of is the sum. Skipping it also saves one model-space contraction per nucleus per site
and loses no information here: on the charge-pure site spaces this export requires, the
per-site parts of a one-electron operator sum back to the whole.

⚠ **The isotropic part is only as good as the active space.** The contact mechanism comes
from core-s spin polarization, which a valence active space does not carry; the front end
warns at the point of selection and this file's provenance carries the active space so a
reader can judge. For a 4f ion the effect is minor (the orbital mechanism dominates); for
s/d spin density, ligand nuclei and spin-only ions the isotropic part is qualitatively wrong.
See [limitations](../../limitations.md).

The report beside the file gives the phase-invariant reduction: the principal `|A|` values in
MHz at the isotope named in the table, and `A.g`, the normalized mixed invariant
$`\mathrm{Tr}_b(\mu \cdot T)`$, whose sign is the relative sign of `A` and `g`. ⚠ Both are
*reductions*, in the same sense the principal g values are — Kuiva fits no A tensor and writes
none.

## The file

The same dull shape as the property dump — versioned header, `[SECTION]` markers, atomic
write, the full Hamiltonian provenance (an empty provenance warns) — with ⚠ **one deliberate
difference, stated in the header: `H` is *not* diagonal.** `[ENERGIES]` lists its
eigenvalues and `[MATRIX U]` the diagonalizing unitary. The `M` convention and the storage
order are OuluSpin's, restated in every file, so the file needs no permutation on the way
in; phases are never canonicalized. Nuclear sites follow the electronic ones, in `[NUCLEI]`
order, each with `M_I = −I … +I` ascending — the same lexicographic order extended, not a
second convention. Spin operator matrices are deliberately not written — the format is
confirmed against what OuluSpin reads, and nothing else widens it.
