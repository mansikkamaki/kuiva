# The methods: overview

Kuiva computes relativistic multireference wavefunctions for strongly correlated,
strongly relativistic systems. The pipeline, and the page that documents each part of it:

```
PySCF scalar X2C SCF ──► ingestion ──► orthonormal working basis
   [scf-reference]      [soc, x2c]         [integrals]
                                               │
                                               ▼
                     scalar MOs ──► spinor expansion (Kramers pairs)   [soc]
                                               │
                                               ▼
              cheap CI ──► state-averaged CASSCF orbital optimizer ◄── RDMs
          [active-spaces]              [casscf]                          ▲
                                ┌──────────┴──────────┐                  │
                                ▼                     ▼                  │
                       conventional CI            DMRG/TTNS ─────────────┘
                            [ci]                    [dmrg]
                                └──────────┬──────────┘
                                           ▼
                                       SC-NEVPT2      [nevpt2]
                                           │
                                           ▼
                property dump / pseudospin export     [properties]
```

Symmetry ([symmetry](symmetry.md)) is a cross-cutting layer: opt-in orbital and state
labels, consumed by the CI, the optimizer and the network.

## The physical strategy

1. **Relativity is two-component X2C** [[13]](../references.md#r13)[[15]](../references.md#r15):
   the four-component problem is exactly decoupled once, at the one-electron level, and the
   calculation runs in the two-component (spinor) picture ([x2c](x2c.md)). There is no
   four-component method path — four-component machinery exists only as an *ingredient* of
   the two-electron picture change.
2. **Spin–orbit coupling enters at the CASSCF/CI level, not at the SCF.** The reference
   orbitals come from a *scalar* (spin-free) X2C SCF — real, easy to converge — and SOC is
   introduced when the two-component wavefunction is built and optimized
   ([soc](soc.md), [scf-reference](scf-reference.md)). The CI roots are then already the
   spin–orbit eigenstates [[106]](../references.md#r106)[[107]](../references.md#r107);
   there is no separate state-interaction step.
3. **The two-electron spin–orbit screening is on by default** (X2CAMF
   [[16]](../references.md#r16)) — without it, spin–orbit splittings come out 5–30% too
   large ([soc](soc.md)).
4. **The multireference ansatz is state-averaged CASSCF** [[93]](../references.md#r93),
   with the CI step solvable by conventional complex determinant CI ([ci](ci.md)) or by an
   in-house tree tensor network ([dmrg](dmrg.md)) behind one orbital optimizer
   ([casscf](casscf.md)).
5. **Dynamic correlation is strongly contracted NEVPT2**
   [[144]](../references.md#r144) ([nevpt2](nevpt2.md)).
6. **The deliverable is a file of operator matrices** — the effective Hamiltonian, magnetic
   moments and electric dipoles in the spin–orbit eigenstate basis — for an external
   crystal-field/ITO code ([properties](properties.md)). Kuiva writes operators and their
   phase-invariant invariants; intensities, radiative rates and crystal-field
   decompositions are deliberately out of scope.

## Two structural principles worth knowing before reading further

**The orthonormal working basis is the fundamental object.** Everything downstream of
ingestion — spinors, CI, DMRG, NEVPT2, properties — is built on an orthonormal basis from
which near-linear dependence has been explicitly removed ([integrals](integrals.md)). This
is what quarantines basis-set contraction complexity in the front end: once orbitals are
orthonormalized, contraction type is invisible to every correlated method.

**Degenerate groups are never cut.** Kramers pairs and free-ion multiplets are exactly the
physics this program exists to resolve, and an operation that truncates *through* an exact
degeneracy breaks it by the truncation threshold — producing Hermitian, plausible, wrong
results. The rule that any truncation keeps whole degenerate groups recurs in the working
basis, the Cholesky pivoting, the state-averaging gate, the network truncations and the
NEVPT2 contraction; each method page states its instance. The companion rule — quantities
are reported per degenerate block, compared through phase-invariant reductions — is in
[notation](../notation.md#degenerate-blocks-and-arbitrary-phases).

## Method summary

| ingredient | method | page |
|---|---|---|
| relativity | one-step exact two-component (X2C), spin-free at SCF, full at CI | [x2c](x2c.md) |
| two-electron SOC | atomic mean-field X2C (X2CAMF), default on | [soc](soc.md) |
| nuclear model | point (default) or Gaussian finite nucleus | [x2c](x2c.md) |
| reference | RHF / ROHF / UHF scalar X2C SCF; broken symmetry; AOC atoms | [scf-reference](scf-reference.md) |
| working basis | canonical orthogonalization, threshold 1e-7 | [integrals](integrals.md) |
| two-electron integrals | one-centre pivoted Cholesky (default) or density fitting | [integrals](integrals.md) |
| active spaces | character selection, ordinal windows, AVAS, SPADE localization | [active-spaces](active-spaces.md) |
| CI | complex determinant full CI, block Davidson, optional Kramers restriction | [ci](ci.md) |
| orbital optimization | two-step state-averaged MCSCF, complex rotations, quasi-Newton / augmented Hessian | [casscf](casscf.md) |
| large active spaces | tree tensor network states (TTNS), two-site sweeps | [dmrg](dmrg.md) |
| dynamic correlation | SC-NEVPT2 in spinor second quantization | [nevpt2](nevpt2.md) |
| symmetry | abelian double groups (labels); full double group (classification only) | [symmetry](symmetry.md) |
| output | operator matrices and phase-invariant reductions | [properties](properties.md) |

Every page shows the working equations as implemented, in the conventions of
[notation](../notation.md), and states explicitly where Kuiva departs from the cited
literature.
