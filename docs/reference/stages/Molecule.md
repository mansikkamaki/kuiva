# The input objects: `Molecule`, `Environment`, `CustomBasis`

Everything a calculation starts from. A `Molecule` is validated eagerly — a typo in a basis
name, a malformed charge field or an unknown nuclear model fails at construction, before any
SCF is paid for.

## `Molecule(atoms, basis, ...)`

```python
mol = kuiva.Molecule(
    atoms=[("Ti", (0.0, 0.0, 0.0)), ("Cl", (2.25, 0.0, 0.0)), ...],
    basis="x2c-SVPall-2c", charge=0, spin=1)
```

| field | default | meaning |
|---|---|---|
| `atoms` | — | list of `(symbol, (x, y, z))`; `("ghost-Cl", pos)` is a ghost atom (below) |
| `basis` | — | one registry family name for all atoms, or a per-atom mapping (below) |
| `charge` | `0` | total charge |
| `spin` | `0` | `2S`, the number of unpaired electrons (PySCF convention) |
| `unit` | `"Angstrom"` | geometry unit; `"Bohr"` to override |
| `point_group` | `None` | abelian double-group symmetry: `"auto"`, or a name of the D2h chain; `None` means no labels and the pre-symmetry behaviour everywhere |
| `classification` | `"auto"` | the non-abelian classification layer (needs `point_group`); labels converged states by the full double group's irreps, changes no number; `False` switches it off |
| `nuclear_model` | `"point"` | nuclear charge distribution: `"point"` or `"gaussian"` (the Visscher–Dyall finite nucleus [[24]](../../references.md#r24)) |
| `environment` | `None` | an `Environment` of point charges (below) |

Constructors from files: `Molecule.from_xyz_file(path, basis)` reads an XMol `.xyz` file
(count line, comment line, atoms, in Angstrom) — ⚠ the count is **checked, not trusted**,
because a file whose header disagrees with its contents is truncated or concatenated, and
reading its first *n* atoms would be a different molecule; a headerless file is accepted.
`Molecule.from_xyz_string(xyz, basis)` takes bare `"El x y z"` lines only.

### The basis assignment

`basis` takes one registry family name, or a mapping whose keys are element symbols
(`"O"`), atom labels (`"Ti2"`), or 1-based atom numbers (`3`) — most specific wins — with an
optional `"default"` entry filling every atom no other key covers. The same addressing is
used by `configuration=`, `gauge_origin=("atom", ...)` and every other per-atom feature, so
there is one way to name an atom in this program.

```python
basis={"default": "x2c-SVPall-2c", "O": "x2c-TZVPall-2c"}   # upgrade every oxygen
basis={"U": "cc-pVTZ-X2C", "default": "x2c-TZVPall-2c"}     # the actinide pattern
```

The registered families: the Karlsruhe `x2c-nZVPall` / `-2c` sets
[[33]](../../references.md#r33)[[34]](../../references.md#r34) (segmented, H–Rn; the `-2c`
recontraction is the default choice for two-component work), the Peterson `cc-pVnZ-X2C` /
`cc-pwCVnZ-X2C` sets [[35]](../../references.md#r35)[[36]](../../references.md#r36)[[37]](../../references.md#r37)
(alkali/alkaline-earth, lanthanides, actinides; CBS-extrapolable), the Dyall uncontracted
sets [[39]](../../references.md#r39) (benchmarking), and ANO-RCC
[[40]](../../references.md#r40)[[41]](../../references.md#r41)[[42]](../../references.md#r42)[[43]](../../references.md#r43).

- **Mixing families across atoms is allowed and checked**: every per-atom basis must target
  a compatible relativistic treatment. Two X2C recontractions pass silently; ANO-RCC (a
  Douglas–Kroll–Hess recontraction) beside an X2C set is allowed and **warns**.
- ⚠ **For an actinide the per-atom form is required, not a convenience**: no single
  X2C-recontracted family covers an actinide *and* an ordinary ligand atom — the Karlsruhe
  sets stop at Rn, and the Peterson sets that reach Ac–Lr begin at K and contain no H, C, N,
  O or F.
- **ECP bases are refused** (X2C needs all-electron), as are Cartesian bases wherever
  two-component dimensions are involved.

### Ghost atoms

An atom written `("ghost-Cl", pos)` carries chlorine's basis functions and nothing else: no
nucleus, no electrons, no mass — what a counterpoise correction is made of. **The label is
the address**: `basis={"ghost-Ar": ...}` reaches it and `basis={"Ar": ...}` does not. Two
ghosts of one element get decorated labels (`ghost-Ar1`, `ghost-Ar2`) and may carry
different bases. A ghost has no chemistry — no atomic mean field, no free-atom reference, no
oxidation state (stating a configuration for one is refused) — but it costs what its
functions cost: it enters the working basis, the integrals, the memory plan and the symmetry
detection like any other centre. `X-Cl` and `ghost:Cl` are accepted and normalized to
`ghost-Cl`.

### The nuclear model

One statement per molecule, inherited by **every** consumer — the molecular integrals, the
four-component atomic solves behind the two-electron spin–orbit screening, the free-atom
reference orbitals, and the isolated-fragment blocks of a DLU decoupling. There is
deliberately no per-atom form. ⚠ It is part of the Hamiltonian, so results are not
comparable across settings, and the atomic mean-field cache keys on it. The effect on
core-region operators grows steeply with Z (negligible at neon, ~3e-3 relative at mercury);
a valence property moves far less. Several four-component programs default to the Gaussian
nucleus where Kuiva defaults to the point one — the first thing to match in a cross-code
comparison.

## `Environment(point_charges, ...)`

What surrounds the molecule: a field of classical point charges (the crystal-embedding
input; see [workflows](../../guide/workflows.md#embedding-in-a-crystal-point-charges) for
how to build and converge one).

| field | default | meaning |
|---|---|---|
| `point_charges` | — | `[(q, (x, y, z)), ...]`, or `(charges, coords)` arrays for a whole lattice |
| `unit` | `""` | unit of the charge coordinates; empty means **the molecule's own** — the only default that cannot silently be wrong |
| `picture_change` | `False` | transform the embedding potential through the Hamiltonian's own X2C decoupling instead of adding the bare operator (measured ~1e-5 relative; costs an integral per charge) |
| `label` | `""` | free text carried into the provenance |

The field reaches exactly two things — the one-electron Hamiltonian and a separately
reported charge–nucleus energy — and moves neither the gauge origin nor the reported nuclear
repulsion; it does take part in symmetry detection. A charge closer than 0.5 bohr to a
nucleus is refused. The provenance records the field by count, net charge, extent and a
digest.

## `CustomBasis(data, relativistic_treatment, ...)`

A basis the registry does not have, per atom, mixable with registered names:

```python
custom = kuiva.CustomBasis(open("my-set.nwchem").read(),      # or {"Ce": parsed_shells}
                           relativistic_treatment="x2c-2c",
                           name="modified-SVP", notes="from Table 2 of the paper")
mol = kuiva.Molecule(atoms=[...], basis={"Ce": custom, "Cl": "x2c-SVPall-2c"})
```

| field | default | meaning |
|---|---|---|
| `data` | — | an NWChem-format string (what Basis Set Exchange [[9]](../../references.md#r9) emits), or `{element: shells}` |
| `relativistic_treatment` | ⚠ **required** | `"x2c-2c"`, `"x2c-1c"`, `"dkh"` or `"nonrel"` |
| `name` | `"custom"` | a label for output and provenance |
| `notes` | `""` | free text carried into the provenance |

⚠ `relativistic_treatment=` is required because it is the one property that cannot be
measured from a list of exponents, and an undeclared one is how a non-relativistic set ends
up under a relativistic Hamiltonian — every number stays plausible and the heavy-element
splittings are wrong. It takes part in the same cross-atom compatibility check registered
families go through. Contraction type is **measured** from the parsed shells; conditioning
is unknown and says so, which routes the two-electron integrals to Cholesky. The atomic
mean-field cache keys on the parsed content of the shells, never on the name.
