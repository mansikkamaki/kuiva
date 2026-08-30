# `PropertyDump(source, path, ...)`

The deliverable: a plain-text, self-describing file with the effective Hamiltonian `H`, the
three magnetic-moment components `mu_x, mu_y, mu_z` (μ_B) and the three electric-dipole
components `d_x, d_y, d_z` (e·a₀), **in the basis of the spin–orbit eigenstates**
[[183]](../../references.md#r183) — the contract with an external ITO / crystal-field
analysis code. `source` is a finished `solver="ci"` [`CASSCF`](CASSCF.md) or
[`CASCI`](CASCI.md), or a finished [`NEVPT2`](NEVPT2.md) on either (the tensor-network route
to properties is [`PseudospinExport`](PseudospinExport.md)).

**After `.run()`:** `.matrices` (a `PropertyMatrices` — compare only through its
phase-invariant `.analyse()`), `.path`, and `.assign()` (the term-assignment offer, reusing
the moment matrices already built).

## Quick reference

| option | default | meaning |
|---|---|---|
| `path` | — | the output file; written atomically |
| `title` | `""` | free text in the header |
| `include_l_s` | `True` | also write the bare `L` and `S` matrices |
| `include_dipole` | `True` | write the electric-dipole block |
| `comments` | `()` | extra header comment lines |
| `inactive_tol` | module default | tolerance on the inactive contribution check |
| `report` | `True` | print the matrices' analysis to the log |

```python
kuiva.PropertyDump(cas, "ticl3.props", title="TiCl3 d1").run()
kuiva.PropertyDump(pt, "ticl3.props").run()      # NEVPT2-corrected H; hybrid protocol recorded
kuiva.PropertyDump(cas, "ticl3.props", include_dipole=False).run()
```

Passing a finished `NEVPT2` substitutes the corrected energies on the diagonal **and
records the hybrid protocol in the header** (`H` from perturbation theory, `mu` from the
CASSCF states). That substitution is available only through this argument, never as a flag,
so the file and its provenance cannot be separated.

## The file

A versioned `[HEADER]`, a `[PROVENANCE]` block of JSON (the full Hamiltonian provenance —
screening, decoupling, nuclear model, environment digest — plus the gauge origin and the
active space as a *physical* statement, never an index window), and one `i j Re Im` record
per matrix element. `format_version` is bumped when the *meaning* of a stored field changes,
so a consumer can refuse rather than misinterpret. `kuiva.read_dump` is a working parser;
`kuiva.PropertyMatrices.from_dump(path)` restores the comparison object.

- ⚠ **`H` is diagonal**, unlike OpenMolcas RASSI's: this CI is already two-component, so its
  roots *are* the spin–orbit eigenstates and there is no separate mixing step. The header
  says so, because a reader from a two-step workflow will expect otherwise.
- ⚠ **Phases are arbitrary and are not canonicalized** (see
  [notation](../../notation.md#degenerate-blocks-and-arbitrary-phases)). Element-by-element
  comparison of these matrices — against another program or another run of this one — is
  meaningless. Compare through the invariants: degeneracy patterns, relative energies,
  `Tr_block(mu_i mu_j)` with its principal g values [[179]](../../references.md#r179);
  `PropertyMatrices.analyse()` is that reduction, and free ions then have *analytic* targets
  (Landé g [[181]](../../references.md#r181)) independent of every program involved.
- **The principal magnetic axes come with the g values** (`Multiplet.g_axes`, `.easy_axis`,
  `.axiality`). ⚠ An axis is a *line*, and a direction at all only where its g value is
  non-degenerate: an easy-plane or isotropic block has no easy axis, and the vector printed
  for one is whatever the eigensolver picked inside the degenerate plane — the table says
  which on every row, and `easy_axis_is_defined()` is the test. The **sign of `det(g)`** is
  reported from the third-order invariant `2i·Tr(mu_x[mu_y, mu_z])/μ_B³` — defined for a
  Kramers doublet only, and reported `?`, never guessed, anywhere else.
- ⚠ **A block of one state reports no g values — `nan`, never `0`.** A single state has no
  first-order moment; "not defined here" is a different statement from "measured zero".
  That distinction is the whole of the non-Kramers case
  ([workflows](../../guide/workflows.md#non-kramers-ions-when-the-ground-doublet-is-two-singlets)):
  `analyse(pseudo_doublet_tol_cm=...)` groups tunnelling-split singlets on explicit request.
- **The electric dipole is the total dipole**: electronic plus, **on the diagonal only**,
  the nuclear `Σ_A Z_A (R_A − R_G)` — so a diagonal element is a state's dipole moment and
  an off-diagonal one a transition dipole; the nuclear vector is a header field so the two
  parts stay separable. ⚠ Its inactive-electron share is **not** zero (`r` is time even, so
  a Kramers pair contributes twice its expectation value where a time-odd operator cancels)
  and is written in `[INACTIVE]`. Its phase-invariant reductions are `Tr_block(d_i d_j)` and
  the block-to-block line strength `S_AB = Σ|d_k[I,J]|²`
  (`PropertyMatrices.line_strengths()`). ⚠ A line strength is **not** an oscillator
  strength or a radiative rate — turning one into an intensity belongs to the external code,
  as the crystal-field analysis does.
- ⚠ **For a charged molecule the dipole is origin dependent**: the diagonal obeys
  `d(R_G) = d(0) − q·R_G` and block-internal invariants move with it; transition elements
  between distinct states do not. Writing such a file warns and records
  `molecular_charge` / `dipole_origin_dependence` — it is not refused, because a charged
  lanthanide complex's transition dipoles are exactly what the operator exists for.
- ⚠ **No picture change is applied to `L`, `S` or `r` by default** — the bare
  non-relativistic operators, used unchanged in the two-component basis, matching OpenMolcas
  RASSI so cross-code comparison is like-for-like; the file warns and records which
  operators were used, every time. `property_picture_change=True` on the front end applies
  the correction [[27]](../../references.md#r27) — to **both** operators, with deliberately
  no way to correct one and not the other — and the header fields
  (`picture_change_on_properties`, `picture_change_on_dipole`) are what distinguish such a
  file, `format_version` deliberately staying put. ⚠ The picture-changed moment does not
  separate into an `L` part and an `S` part; a consumer takes it as one operator.
- **The inactive contribution is computed and checked, never assumed away**: a
  Kramers-paired inactive set contributes exactly zero to the magnetic moment, and a nonzero
  result is a statement about the *orbitals* and warns rather than being dropped.

## Reading files back

```python
before = kuiva.PropertyMatrices.from_dump("before.props").analyse()
after  = kuiva.PropertyMatrices.from_dump("after.props").analyse()
[(b.size, b.g_values) for b in after]              # what may be compared
```

⚠ What comes back is the **file, not the calculation**: the provenance, gauge origin, active
space and picture-change record are restored — those are what make the numbers
interpretable — and nothing else; the object cannot be handed back to a stage.
`kuiva.read_dump` returns the raw form with the header verbatim as a dictionary.
