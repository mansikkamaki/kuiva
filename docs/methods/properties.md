# Properties: moments, g values, populations, pseudospin

Everything upstream exists to produce **operator matrices in the basis of the spin–orbit
eigenstates** — the effective Hamiltonian, the magnetic moments, the electric dipoles —
which an external ITO/crystal-field code turns into what an experiment measures. Kuiva
writes the operators and their phase-invariant reductions; intensities, radiative rates,
Stevens decompositions and crystal-field analysis are deliberately out of scope. The file
formats and options are on [`PropertyDump`](../reference/stages/PropertyDump.md) and
[`PseudospinExport`](../reference/stages/PseudospinExport.md); this page is the physics.

## The operator matrices

The magnetic moment is $\boldsymbol{\mu} = -(\mathbf L + g_e \mathbf S)\,\mu_B$, evaluated
through one-particle transition density matrices
$\gamma^{IJ}_{tu} = \langle I | E_{tu} | J \rangle$ — the third consumer of the same
excitation map the sigma vector and RDMs are built from
([ci](ci.md#the-sigma-vector))[[183]](../references.md#r183):

$$
\mu^{IJ}_k = -\Big( \sum_{tu} (L_k + g_e S_k)_{tu}\, \gamma^{IJ}_{tu}
             + \delta_{IJ} \sum_{i \in \mathrm{inact}} (L_k + g_e S_k)_{ii} \Big)
             \quad [\mu_B],
$$

$$
d^{IJ}_k = -\Big( \sum_{tu} (r - R_G)_{k,tu}\, \gamma^{IJ}_{tu}
           + \delta_{IJ} \sum_{i \in \mathrm{inact}} (r - R_G)_{k,ii} \Big)
           + \delta_{IJ} \sum_A Z_A (R_A - R_G)_k \quad [e\,a_0].
$$

Everything is one-electron, so nothing beyond the one-particle transition densities is ever
needed, however large the CI; on the tensor-network route the same densities come from
network contractions. Three built-in decisions:

- ⚠ **The nuclear term is on the diagonal only** — off the diagonal it would invent a
  transition dipole proportional to the nuclear charge. A diagonal element of $d$ is a
  state's dipole moment; an off-diagonal one is a transition dipole.
- ⚠ **The inactive term is computed for every operator, never assumed away — and what the
  theorem says depends on time parity.** For any time-**odd** operator a Kramers pair
  contributes $\langle\psi|A|\psi\rangle + \langle T\psi|A|T\psi\rangle = 0$, so a
  Kramers-paired inactive set contributes exactly nothing to $\mathbf L$ or $\mathbf S$ —
  a theorem about the *orbitals*, measured and warned about when violated rather than
  skipped, because the failure it guards is a moment silently missing a core contribution.
  For the time-**even** $r$ the same sum is real and generally large (a pair contributes
  *twice* its expectation value), computed and used with the warning off.
- ⚠ **No picture change is applied to $\mathbf L$, $\mathbf S$ or $r$ by default** — the
  bare non-relativistic AO operators used unchanged in the two-component basis, which
  matches OpenMolcas RASSI and keeps cross-code comparison like-for-like. The
  approximation is *measured*: 1e-4 to 1e-3 relative on free-ion g factors over Z = 5–81,
  ~2e-4 on a 3d complex, splitting no degeneracy — an order of magnitude below the
  validation band, which is why the default stands. `property_picture_change=True` applies
  the correction [[27]](../references.md#r27) to **both** operators (there is deliberately
  no way to correct one and not the other); ⚠ the picture-changed magnetic operator does
  not separate into an $L$ part and an $S$ part — the four-component magnetic interaction
  is the odd operator $c\,\boldsymbol\alpha\cdot\mathbf A$, so the whole of $L + 2S$
  transforms and the $g_e - 2$ anomaly is a separate term — and a consumer takes it as one
  operator. The electric operator is *even*, so its four-component blocks are the mirror
  image of the magnetic one's, through the same $X$ and $R$.

## Phase-invariant analysis

Matrix phases are arbitrary and within a degenerate block the states are defined only up to
a unitary mixing ([notation](../notation.md#degenerate-blocks-and-arbitrary-phases)) —
element-by-element comparison of moment matrices is meaningless, against another program or
another run of this one. The quantities that *are* well defined, and the only ones any
validation of the output may use: for a degenerate block $b$ of dimension $d$,

$$
M_{ij} = \mathrm{Tr}_b\!\left( \mu_i\, \mu_j \right) \quad [\mu_B^2],
$$

invariant under any unitary on the block. $M$ is the block analogue of the g tensor
[[179]](../references.md#r179):

- **Kramers doublet** ($d = 2$): with the pseudospin convention
  $\mu_i = -\tfrac12 \mu_B \sum_k g_{ik}\sigma_k$ [[180]](../references.md#r180),
  $M = \tfrac12 \mu_B^2\, g g^{\mathsf T}$, so the principal g values are
  $\sqrt{\mathrm{eig}(2M/\mu_B^2)}$.
- **Free-ion multiplet** ($d = 2J{+}1$, $\boldsymbol\mu = -g_J \mu_B \mathbf J$):
  $M_{ij} = \delta_{ij}\, g_J^2\, J(J{+}1)(2J{+}1)/3$.

Both are one formula with $J = (d-1)/2$ — a Kramers doublet *is* the $J = 1/2$ case — which
gives every free-ion system an **analytic** target (the Landé factor
[[181]](../references.md#r181)) that no program's conventions can touch: the most valuable
checks in the suite. The principal axes come with the g values; ⚠ an axis is a *line*, and
a direction at all only where its g value is non-degenerate. $|{\det g}|$ follows from $M$;
its **sign** comes from the third-order invariant
$2i\,\mathrm{Tr}(\mu_x[\mu_y,\mu_z])/\mu_B^3$ — defined for a Kramers doublet only,
reported `?` anywhere else. The electric analogues are
$\mathrm{Tr}_b(d_i d_j)$ and the block-to-block **line strength**
$S_{AB} = \sum_k \sum_{I\in A, J\in B} |d_k[I,J]|^2$ — the only phase-invariant statement
about a transition; ⚠ a line strength is not an oscillator strength or a rate.

**Non-Kramers ions.** An integer-spin ion has no Kramers protection: its ground "doublet"
is two singlets split by a tunnelling gap $\Delta$, each arriving as a block of one state —
which carries **no** first-order moment, so it reports `nan`, never 0 ("not defined" and
"measured zero" must not print the same). Grouping the pair is an explicit request; for a
grouped pair the effective-spin description [[182]](../references.md#r182)
[[180]](../references.md#r180) carries $g_z$ and $\Delta$ with transverse components
vanishing **identically** — and the transverse residual is the check that can fail: a value
comparable to $g_z$ says the two states were not one doublet
([workflows](../guide/workflows.md#non-kramers-ions-when-the-ground-doublet-is-two-singlets)).

## The spin analysis and the term-assignment offer

$\langle S^2 \rangle$ is computed per state as
$\langle S^2 \rangle = \sum_k \| S_k |I\rangle \|^2$ — each $S_k$ is one-body, so no 2-RDM
is needed [[46]](../references.md#r46) — with the operator split by orbital space: the
active part, the inactive trace, and three cross-space families whose squares add by
orthogonality. ⚠ **The out-of-space contributions are computed and included, never assumed
away** — they vanish identically for Kramers-paired spin-separable spaces, which is exactly
the assumption worth measuring (dropping them on a constructed non-separable case loses 93%
of the value) — and what is *reported* beside the value is their share. Reported **per
degenerate block, never per state**. With coupling off, the block value is $S(S{+}1)$
exactly and $2S{+}1$ reads straight off; with coupling on, $S$ is not conserved and the
same number is a **purity** reading — do not expect an integral multiplicity, and read the
excess against the printed leakage, not against an integer.

The assignment is an **inference and prints as one** — its own report, evidence and fit
residual beside every label, never a column of the state table, never in a stored file.
Coupling off: $S$ from $\langle S^2\rangle$, then $2L{+}1 = d/(2S{+}1)$, both required
integral. Coupling on: $J = (d-1)/2$, $S$ from purity, and $L$ from the Landé formula
inverted,

$$
g_J = \tfrac32 + \frac{S(S{+}1) - L(L{+}1)}{2 J(J{+}1)}
\;\;\Rightarrow\;\;
L(L{+}1) = S(S{+}1) - (g_J - \tfrac32)\, 2 J(J{+}1),
$$

with the triangle condition checked. ⚠ The inversion is only as good as the free-ion
picture: a crystal-field level of a complex is not a $2J{+}1$ manifold, its inverted $L$
comes out non-integral, and `?` is the correct outcome — which is why the residual is
printed rather than hidden.

## Populations and charges

Löwdin reduced populations [[44]](../references.md#r44), generalized to two components: the
**charge** is the spin trace of the symmetrically orthogonalized density's diagonal, and
the **spin is a vector**, read off the spin blocks
($s_x = +\mathrm{Re}\,\tilde D^{\alpha\beta}$,
$s_y = -\mathrm{Im}\,\tilde D^{\alpha\beta}$,
$s_z = \tfrac12(\tilde D^{\alpha\alpha} - \tilde D^{\beta\beta})$, per orthogonalized AO).
Three readings to avoid: a state-averaged Kramers pair has $\mathbf s = 0$ *identically*
(time reversal, not a lost moment — look at a single state); the quantization axis is
arbitrary, so $|\mathbf s|$ is the default output; and spin is not the magnetic moment —
with spin–orbit coupling the orbital part is comparable, and moments live in the analysis
above. Reduced *orbital* populations are the robust half and the feature's purpose; ⚠ the
Löwdin **atomic charge is withdrawn from every printed report** (measured sign-wrong on
ionic textbook compounds, unrescued by basis [[174]](../references.md#r174)); the supported
charge is the **atomic-reference partition** — populations in occupation-weight
orthogonalized free-atom AOC orbitals, atomic virtuals kept behind them, so bonding density
is attributed by atomic *character* rather than by whose functions describe it
([Reference](../reference/stages/Reference.md#atomic_reference_charges)). Orbital pictures
are spinor-density decompositions, not orbitals
([Reference](../reference/stages/Reference.md#write_moldenpath-columns-)).

## The pseudospin export

The multi-site deliverable: the local-multiplet model of the tensor network
([dmrg](dmrg.md#densities-and-the-local-multiplet-model)) gives per magnetic site a
$d_k$-dimensional multiplet space, named a **pseudospin** $\tilde S_k = (d_k - 1)/2$; the
effective Hamiltonian and moment operators are written on the ordered pseudospin product
basis with the unitary mapping the ab initio states onto it. The $M$ labelling inside each
site space is self-consistent and stated: descending moment projection along a stated axis
— by default the site's principal magnetic axis, the largest-eigenvalue eigenvector of the
site's $M_{ij}$ [[179]](../references.md#r179), with the Abragam–Bleaney sign convention
$\boldsymbol\mu = -g\mu_B\tilde{\mathbf S}$ [[180]](../references.md#r180) — while
per-state **phases are never canonicalized** (a phase convention is the consumer's job).
Validation of the export goes through the phase-invariant reductions above, only; no test
may compare an element.
