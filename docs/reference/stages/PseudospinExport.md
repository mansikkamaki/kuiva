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

## The file

The same dull shape as the property dump — versioned header, `[SECTION]` markers, atomic
write, the full Hamiltonian provenance (an empty provenance warns) — with ⚠ **one deliberate
difference, stated in the header: `H` is *not* diagonal.** `[ENERGIES]` lists its
eigenvalues and `[MATRIX U]` the diagonalizing unitary. The `M` convention and the storage
order are OuluSpin's, restated in every file, so the file needs no permutation on the way
in; phases are never canonicalized. Spin operator matrices are deliberately not written —
the format is confirmed against what OuluSpin reads, and nothing else widens it.
