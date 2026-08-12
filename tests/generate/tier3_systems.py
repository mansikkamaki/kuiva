"""Tier 3: polynuclear systems for tensor-network validation.

Tiers 1 and 2 answer *"does Kuiva reproduce a calculation someone else can also do?"*.
Tier 3 answers a question no external code can answer for us: **does the tensor network
have the right structure for a system with three or more coupled paramagnetic centres?**

Why there are no reference calculations here
--------------------------------------------
Every system below is deliberately beyond conventional CI. ``cas_determinants`` records the
size of the complex-determinant space its honest active space would span (the conventional-CI ceiling sets the
conventional-CI ceiling at ~12-14 spinors); the smallest entry here is ~10^7 and the largest
~10^11. PySCF, OpenMolcas and DIRAC cannot produce a reference for any of them, so **nothing
in this tier is validated against a stored number from another program.** Instead each system
is chosen so that rigorous, program-independent statements exist about it:

* **Clebsch-Gordan counting** fixes the spin-manifold decomposition exactly.
* **The Lieb-Mattis theorem** fixes the exact ground-state total spin of a *bipartite*
  antiferromagnet - no approximation, no calculation.
* **Kramers' theorem** fixes the minimum degeneracy from the parity of unpaired electrons.
* **Graph topology** fixes which tensor network can represent the state exactly: a tree is
  exactly representable by a TTNS, a graph with cycles is not.
* Several systems have an **experimentally established** ground-state spin, which is
  independent of every program involved - the most valuable kind of check in the suite,
  for the same reason the analytic Lande g values are in Tier 2.

The systems are chosen to span *connectivity classes*, because the connectivity - not the
chemistry - is what decides the tensor network:

===================  =========================  ==========================================
Key                  Topology                   Tensor network it exercises
===================  =========================  ==========================================
``mn3_linear``       path P3                    plain MPS; the baseline chain
``dy2_n2rad``        path P3, mixed local dim   MPS with inhomogeneous local dimensions
``fe3_oxo``          cycle C3                   minimal frustrated loop (non-bipartite)
``fe4_star``         star K(1,3)                TTNS - the minimal genuine tree
``mn4ca_oec``        triangle + pendant         hierarchical: a loop with a tree branch
``fe4s4``            complete graph K4          3D connectivity; PEPS-like, worst case
``cr8_ring``         cycle C8                   periodic MPS / long-range bond
``cr7ni_ring``       cycle C8 with one defect   same topology, one substituted site
===================  =========================  ==========================================

``cr8_ring``/``cr7ni_ring`` are deliberately a matched pair, exactly as ``ce3p``/``yb3p`` are
in Tier 2: identical topology, one site changed, and a *different* rigorous ground-state spin
(0 vs 1/2). A network that gets the topology right but the local quantum numbers wrong passes
the first and fails the second.

Chemical provenance
-------------------
Every entry is a real structural motif, truncated for cost where the truncation cannot change
the magnetic topology (bulky co-ligands replaced by formate/hydroxide/amide, cysteinate by
SH-). Truncation never removes or adds a paramagnetic centre or an exchange pathway. These
are *model* structures in the same sense as ``ti2cl6`` in Tier 2: well defined and
representative, never to be quoted as predictions.

References
----------
* Lieb-Mattis theorem (ordering of energy levels, bipartite antiferromagnets):
  E. Lieb, D. Mattis, J. Math. Phys. 3, 749 (1962).
* Kramers degeneracy: H. A. Kramers, Proc. Amsterdam Acad. 33, 959 (1930).
* ``fe4_star`` - [Fe4(OMe)6(dpm)6], S = 5 ground state: A.-L. Barra, A. Caneschi, A. Cornia,
  F. Fabrizi de Biani, D. Gatteschi, C. Sangregorio, R. Sessoli, L. Sorace, J. Am. Chem. Soc.
  121, 5302 (1999); A. Cornia et al., Angew. Chem. Int. Ed. 43, 1136 (2004).
* ``cr8_ring`` - [Cr8F8(O2CtBu)16], S = 0 ground state: J. van Slageren et al., Chem. Eur. J.
  8, 277 (2002).
* ``cr7ni_ring`` - [Cr7NiF8(O2CtBu)16], S = 1/2 ground state: S. Larsen et al., Phys. Rev.
  Lett. 91, 067201 (2003); G. A. Timco et al., Nat. Nanotechnol. 4, 173 (2009).
* ``fe3_oxo`` - basic iron(III) carboxylate [Fe3O(O2CR)6L3]+ and its spin frustration:
  R. D. Cannon, R. P. White, Prog. Inorg. Chem. 36, 195 (1988).
* ``fe4s4`` - biological [4Fe-4S] cluster: H. Beinert, R. H. Holm, E. Munck, Science 277, 653
  (1997). As a DMRG benchmark: S. Sharma, K. Sivalingam, F. Neese, G. K.-L. Chan, Nat. Chem.
  6, 927 (2014); Z. Li, S. Guo, Q. Sun, G. K.-L. Chan, Nat. Chem. 11, 1026 (2019).
* ``mn4ca_oec`` - photosystem II oxygen-evolving complex, 1.9 A structure: Y. Umena,
  K. Kawakami, J.-R. Shen, N. Kamiya, Nature 473, 55 (2011).
* ``dy2_n2rad`` - N2(3-) radical-bridged Dy2 single-molecule magnet: J. D. Rinehart, M. Fang,
  W. J. Evans, J. R. Long, J. Am. Chem. Soc. 133, 14236 (2011); Nat. Chem. 3, 538 (2011).
* ``mn3_linear`` - model linear trinuclear Mn(II) carboxylate motif, representative of the
  [Mn3(O2CR)6L2] structural family; used here as the clean bipartite chain baseline.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: Conventional-CI ceiling in spinors. Active spaces above this need DMRG, which is
#: the entire reason this tier exists.
CONVENTIONAL_CI_SPINOR_CEILING = 14


@dataclass(frozen=True)
class Site:
    """One paramagnetic centre.

    Spins are stored as ``twice_spin`` (integer) so half-integer spins stay exact - a float
    3/2 compared with ``==`` is a bug waiting to happen in degeneracy counting.

    For ``kind == "ion_soc"`` the label is a *total angular momentum* J, not a spin: for
    Dy(III) the 4f^9 ground multiplet is 6H15/2 and SOC is far larger than the exchange, so
    the correct local Hilbert space is the 16-fold J = 15/2 manifold, not a spin multiplet.
    Isotropic-exchange theorems (Lieb-Mattis) do not apply to such sites.
    """
    label: str
    ion: str
    twice_spin: int
    kind: str = "ion"                      # "ion" | "ion_soc" | "radical"

    @property
    def spin(self) -> float:
        return self.twice_spin / 2.0

    @property
    def local_dim(self) -> int:
        """Dimension of this centre's local Hilbert space (its multiplet dimension)."""
        return self.twice_spin + 1


@dataclass(frozen=True)
class Tier3System:
    """A polynuclear system, defined by its magnetic topology rather than its geometry.

    No coordinates: nothing in this tier is fed to an integral code, and the connectivity is
    the whole content. ``edges`` are the *exchange pathways* - pairs of centres with a
    non-negligible magnetic interaction - which is exactly the graph a tensor network must
    be matched to.
    """
    key: str
    label: str
    formula: str
    sites: Tuple[Site, ...]
    edges: Tuple[Tuple[int, int], ...]
    topology: str
    network_target: str
    cas_electrons: int
    cas_orbitals: int
    #: All exchange pathways antiferromagnetic? Required for Lieb-Mattis.
    all_antiferromagnetic: bool = True
    #: False when SOC dominates exchange, so an isotropic spin Hamiltonian is not valid.
    isotropic_exchange: bool = True
    #: Experimentally established ground-state total spin, as ``twice_spin``; None if unknown.
    experimental_twice_spin: Optional[int] = None
    truncation: str = ""
    physics_note: str = ""
    provenance: str = ""

    # --- basic derived quantities ---------------------------------------------------------
    @property
    def n_sites(self) -> int:
        return len(self.sites)

    @property
    def local_dims(self) -> Tuple[int, ...]:
        return tuple(s.local_dim for s in self.sites)

    @property
    def hilbert_dim(self) -> int:
        """Dimension of the coupled local-multiplet space, prod_i (2S_i + 1)."""
        d = 1
        for s in self.sites:
            d *= s.local_dim
        return d

    @property
    def cas_spinors(self) -> int:
        return 2 * self.cas_orbitals

    @property
    def cas_determinants(self) -> int:
        """Complex determinants spanned by CAS(nelec, norb) with SOC on.

        SOC breaks spin symmetry, so there is no alpha/beta factorisation: the count
        is simply "choose nelec spinors out of 2*norb".
        """
        return math.comb(self.cas_spinors, self.cas_electrons)

    @property
    def beyond_conventional_ci(self) -> bool:
        return self.cas_spinors > CONVENTIONAL_CI_SPINOR_CEILING

    @property
    def total_unpaired_electrons(self) -> int:
        """Sum of 2S over sites = number of unpaired electrons in the high-spin picture."""
        return sum(s.twice_spin for s in self.sites)

    @property
    def kramers_system(self) -> bool:
        """Odd number of unpaired electrons => every level is at least doubly degenerate."""
        return self.total_unpaired_electrons % 2 == 1


# --- graph theory on the exchange topology --------------------------------------------------
def adjacency(system: Tier3System) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(system.n_sites)]
    for i, j in system.edges:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def connected_components(system: Tier3System) -> int:
    adj = adjacency(system)
    seen = [False] * system.n_sites
    n = 0
    for start in range(system.n_sites):
        if seen[start]:
            continue
        n += 1
        q = deque([start])
        seen[start] = True
        while q:
            for nb in adj[q.popleft()]:
                if not seen[nb]:
                    seen[nb] = True
                    q.append(nb)
    return n


def bipartition(system: Tier3System) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Two-colour the exchange graph; return the sublattices, or None if odd cycles exist.

    Bipartite is exactly the condition for the Lieb-Mattis theorem, and non-bipartite is
    exactly what "spin frustration" means for an antiferromagnet.
    """
    adj = adjacency(system)
    colour: List[Optional[int]] = [None] * system.n_sites
    for start in range(system.n_sites):
        if colour[start] is not None:
            continue
        colour[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if colour[v] is None:
                    colour[v] = 1 - colour[u]
                    q.append(v)
                elif colour[v] == colour[u]:
                    return None
    a = tuple(i for i, c in enumerate(colour) if c == 0)
    b = tuple(i for i, c in enumerate(colour) if c == 1)
    return a, b


def cycle_rank(system: Tier3System) -> int:
    """Independent cycles: E - V + C. Zero for a tree, and *the* number that decides whether
    a tree tensor network can represent the state exactly."""
    return len(system.edges) - system.n_sites + connected_components(system)


def is_tree(system: Tier3System) -> bool:
    return cycle_rank(system) == 0 and connected_components(system) == 1


def degree_sequence(system: Tier3System) -> Tuple[int, ...]:
    adj = adjacency(system)
    return tuple(sorted((len(a) for a in adj), reverse=True))


def graph_diameter(system: Tier3System) -> int:
    """Longest shortest-path. Sets the minimum depth/bond distance a network must span."""
    adj = adjacency(system)
    best = 0
    for start in range(system.n_sites):
        dist = [-1] * system.n_sites
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        best = max(best, max(dist))
    return best


# --- exact spin algebra ---------------------------------------------------------------------
def couple_spins(twice_spins: Sequence[int]) -> Dict[int, int]:
    """Clebsch-Gordan decomposition of a product of spin multiplets.

    Returns ``{twice_S: number_of_multiplets}``. Exact integer arithmetic throughout, and
    the total dimension ``sum (2S+1) * n_S`` must equal ``prod (2S_i + 1)`` - which is
    asserted in the tests as an internal consistency check.
    """
    dist: Dict[int, int] = {0: 1}
    for two_s in twice_spins:
        nxt: Dict[int, int] = {}
        for two_tot, count in dist.items():
            lo, hi = abs(two_tot - two_s), two_tot + two_s
            for two_new in range(lo, hi + 1, 2):
                nxt[two_new] = nxt.get(two_new, 0) + count
        dist = nxt
    return dict(sorted(dist.items()))


def lieb_mattis_twice_spin(system: Tier3System) -> Optional[int]:
    """Exact ground-state total spin of a bipartite antiferromagnet, or None if inapplicable.

    Lieb and Mattis, J. Math. Phys. 3, 749 (1962): for a connected bipartite Heisenberg
    antiferromagnet the ground state has total spin exactly ``|S_A - S_B|``, the difference
    of the two sublattice spin sums. This is a theorem, not an approximation - which is what
    makes it usable as a reference when no calculation is possible.

    Returns None when a hypothesis fails: a non-bipartite (frustrated) graph, a ferromagnetic
    pathway, a disconnected graph, or a site whose local space is a SOC multiplet rather than
    a spin.
    """
    if not system.all_antiferromagnetic or not system.isotropic_exchange:
        return None
    if connected_components(system) != 1:
        return None
    if any(s.kind == "ion_soc" for s in system.sites):
        return None
    parts = bipartition(system)
    if parts is None:
        return None
    a, b = parts
    two_sa = sum(system.sites[i].twice_spin for i in a)
    two_sb = sum(system.sites[i].twice_spin for i in b)
    return abs(two_sa - two_sb)


def is_frustrated(system: Tier3System) -> bool:
    """Antiferromagnetic coupling on a non-bipartite graph = geometric spin frustration."""
    return system.all_antiferromagnetic and bipartition(system) is None


# --- exact diagonalisation of the effective spin model --------------------------------------
#: Above this coupled dimension the dense ED below is not worth doing in a test suite. The
#: eight-site rings sit above it (4^8 = 65536) and are covered by Lieb-Mattis alone.
ED_MAX_DIM = 2000


def _spin_matrices(twice_s: int):
    """``(Sz, S+, S-)`` for one site, in the |S, m> basis ordered m = S, S-1, ..., -S.

    Real matrices throughout: every operator needed below can be written with Sz and the
    ladder operators, so no complex arithmetic is required for an isotropic spin model.
    """
    import numpy as np
    dim = twice_s + 1
    m = np.array([(twice_s - 2 * k) / 2.0 for k in range(dim)])
    sz = np.diag(m)
    sp = np.zeros((dim, dim))
    s = twice_s / 2.0
    for k in range(1, dim):
        # S+ |S, m> = sqrt(S(S+1) - m(m+1)) |S, m+1>, raising from row k to row k-1.
        mm = m[k]
        sp[k - 1, k] = math.sqrt(s * (s + 1) - mm * (mm + 1))
    return sz, sp, sp.T


def _embed(op, site: int, dims: Sequence[int]):
    """Kronecker-embed a single-site operator into the full product space."""
    import numpy as np
    out = np.array([[1.0]])
    for k, d in enumerate(dims):
        out = np.kron(out, op if k == site else np.eye(d))
    return out


def heisenberg_ground_state(system: Tier3System, j_coupling: float = 1.0,
                            tol: float = 1e-9) -> Optional[Dict[str, object]]:
    """Exact ground state of ``H = J sum_<ij> S_i . S_j`` for the exchange graph.

    This exists to check the Lieb-Mattis prediction against actual quantum mechanics rather
    than trusting a theorem lookup: for the bipartite systems the two must agree, and if they
    ever disagree the bug is in :func:`lieb_mattis_twice_spin` or in the system definition,
    not in the physics.

    Returns None when the model does not apply (SOC-dominated sites) or the space is too
    large for dense ED. ``J > 0`` is antiferromagnetic in this convention.
    """
    import numpy as np
    if not system.isotropic_exchange or any(s.kind == "ion_soc" for s in system.sites):
        return None
    dims = list(system.local_dims)
    dim = system.hilbert_dim
    if dim > ED_MAX_DIM:
        return None

    ops = [_spin_matrices(s.twice_spin) for s in system.sites]
    sz = [_embed(o[0], k, dims) for k, o in enumerate(ops)]
    sp = [_embed(o[1], k, dims) for k, o in enumerate(ops)]
    sm = [_embed(o[2], k, dims) for k, o in enumerate(ops)]

    def dot(a: int, b: int):
        return sz[a] @ sz[b] + 0.5 * (sp[a] @ sm[b] + sm[a] @ sp[b])

    ham = np.zeros((dim, dim))
    for a, b in system.edges:
        ham += j_coupling * dot(a, b)

    # S_tot^2 = sum_i S_i(S_i+1) + 2 sum_{i<j} S_i . S_j
    s2 = np.zeros((dim, dim))
    for site in system.sites:
        s = site.twice_spin / 2.0
        s2 += s * (s + 1) * np.eye(dim)
    for a in range(system.n_sites):
        for b in range(a + 1, system.n_sites):
            s2 += 2.0 * dot(a, b)

    evals, evecs = np.linalg.eigh(ham)
    e0 = float(evals[0])
    ground = np.where(evals - e0 <= tol * max(1.0, abs(e0)))[0]
    # <S^2> in the ground manifold; S(S+1) -> S = (-1 + sqrt(1+4<S^2>))/2.
    s2_vals = [float(evecs[:, i] @ s2 @ evecs[:, i]) for i in ground]
    spread = max(s2_vals) - min(s2_vals)
    s_val = 0.5 * (-1.0 + math.sqrt(1.0 + 4.0 * (sum(s2_vals) / len(s2_vals))))
    twice_s = int(round(2 * s_val))
    return {
        "ground_energy": e0,
        "degeneracy": int(len(ground)),
        "twice_total_spin": twice_s,
        "s2_spread": spread,
        "first_gap": float(evals[len(ground)] - e0) if len(ground) < dim else 0.0,
        "hilbert_dim": dim,
    }


# --- the suite -------------------------------------------------------------------------------
def _ions(ion: str, twice_spin: int, n: int, prefix: str, kind: str = "ion") -> List[Site]:
    return [Site(f"{prefix}{k + 1}", ion, twice_spin, kind) for k in range(n)]


def _path_edges(n: int) -> Tuple[Tuple[int, int], ...]:
    return tuple((k, k + 1) for k in range(n - 1))


def _ring_edges(n: int) -> Tuple[Tuple[int, int], ...]:
    return tuple((k, (k + 1) % n) for k in range(n))


SYSTEMS: Tuple[Tier3System, ...] = (
    Tier3System(
        key="mn3_linear",
        label="Mn(II)3 linear chain",
        formula="[Mn3(mu-OH)2(HCO2)4]",
        sites=tuple(_ions("Mn(II)", 5, 3, "Mn")),
        edges=_path_edges(3),
        topology="path P3",
        network_target="MPS",
        cas_electrons=15, cas_orbitals=15,
        experimental_twice_spin=None,
        truncation="model linear trinuclear Mn(II) carboxylate; bulky RCO2- replaced by "
                   "formate. Truncation touches no metal and no exchange pathway",
        physics_note="The baseline chain: three S = 5/2 centres, nearest-neighbour AF only. "
                     "Bipartite with sublattices {Mn1,Mn3} and {Mn2}, so Lieb-Mattis gives "
                     "S_ground = |5 - 5/2| = 5/2 exactly. An MPS in site order 1-2-3 matches "
                     "the exchange graph, so this is the case where a chain network should be "
                     "optimal and orbital ordering should recover the natural order",
        provenance="[Mn3(O2CR)6L2] structural family",
    ),
    Tier3System(
        key="dy2_n2rad",
        label="Dy2 N2(3-) radical-bridged SMM",
        formula="[{Dy(NH2)2}2(mu-N2)]-",
        sites=(Site("Dy1", "Dy(III) 6H15/2", 15, "ion_soc"),
               Site("N2rad", "N2(3-) pi* radical", 1, "radical"),
               Site("Dy2", "Dy(III) 6H15/2", 15, "ion_soc")),
        edges=_path_edges(3),
        topology="path P3 (mixed local dimensions)",
        network_target="MPS with inhomogeneous local dimensions",
        cas_electrons=19, cas_orbitals=15,
        isotropic_exchange=False,
        truncation="N(SiMe3)2- and THF replaced by NH2-; the N2(3-) bridge and both Dy(III) "
                   "centres - the entire magnetic core - are kept intact",
        physics_note="Ion-radical-ion, and the reason this tier is not spin-only: SOC on "
                     "Dy(III) far exceeds the exchange, so the local space is the 16-fold "
                     "J = 15/2 multiplet, not a spin multiplet. Local dimensions are "
                     "(16, 2, 16) - strongly inhomogeneous, which is what a real MPS bond "
                     "dimension has to cope with, and a case where a uniform-local-dimension "
                     "implementation silently does the wrong thing. Isotropic-exchange "
                     "theorems deliberately do NOT apply here",
        provenance="Rinehart, Fang, Evans, Long, J. Am. Chem. Soc. 133, 14236 (2011)",
    ),
    Tier3System(
        key="fe3_oxo",
        label="Fe(III)3 basic carboxylate triangle",
        formula="[Fe3(mu3-O)(HCO2)6(H2O)3]+",
        sites=tuple(_ions("Fe(III)", 5, 3, "Fe")),
        edges=_ring_edges(3),
        topology="cycle C3",
        network_target="frustrated loop (non-bipartite)",
        cas_electrons=15, cas_orbitals=15,
        truncation="pivalate/acetate replaced by formate; the mu3-oxo triangle is intact",
        physics_note="The minimal frustrated system, and the smallest topology no tree "
                     "network can represent exactly. Equilateral AF triangle => odd cycle => "
                     "NOT bipartite => Lieb-Mattis does not apply and there is no unique "
                     "classical ground configuration. The correct statement is the negative "
                     "one, and the test asserts exactly that rather than inventing a value",
        provenance="Cannon and White, Prog. Inorg. Chem. 36, 195 (1988)",
    ),
    Tier3System(
        key="fe4_star",
        label="Fe(III)4 star SMM",
        formula="[Fe4(mu-OMe)6(HCO2)6]",
        sites=(Site("Fe_c", "Fe(III) central", 5),
               Site("Fe1", "Fe(III) peripheral", 5),
               Site("Fe2", "Fe(III) peripheral", 5),
               Site("Fe3", "Fe(III) peripheral", 5)),
        edges=((0, 1), (0, 2), (0, 3)),
        topology="star K(1,3)",
        network_target="TTNS (minimal genuine tree)",
        cas_electrons=20, cas_orbitals=20,
        experimental_twice_spin=10,
        truncation="dpm- (dipivaloylmethanide) replaced by formate; the Fe4 core and all "
                   "six methoxide bridges are kept",
        physics_note="The minimal genuine tree: a central Fe(III) coupled to three peripheral "
                     "Fe(III), with no peripheral-peripheral pathway. An MPS must route two "
                     "of the three branches through a long-range bond, whereas a TTNS matches "
                     "the topology exactly - so this is the system that should show a "
                     "measurable MPS-vs-TTNS difference. Bipartite ({centre} | {3 leaves}), "
                     "so Lieb-Mattis gives S = |15/2 - 5/2| = 5 - which is the EXPERIMENTAL "
                     "ground-state spin of the Fe4 SMM family, an external check that depends "
                     "on no program at all",
        provenance="Barra, Caneschi, Cornia, Fabrizi de Biani, Gatteschi, Sangregorio, "
                   "Sessoli, Sorace, J. Am. Chem. Soc. 121, 5302 (1999)",
    ),
    Tier3System(
        key="mn4ca_oec",
        label="Photosystem II Mn4CaO5 cluster (S1 state)",
        formula="[Mn4CaO5(H2O)4]",
        sites=(Site("Mn1", "Mn(IV)", 3),
               Site("Mn2", "Mn(IV)", 3),
               Site("Mn3", "Mn(III)", 4),
               Site("Mn4", "Mn(III) dangler", 4)),
        edges=((0, 1), (1, 2), (0, 2), (2, 3)),
        topology="triangle + pendant vertex",
        network_target="hierarchical: loop with a tree branch",
        cas_electrons=14, cas_orbitals=20,
        truncation="protein ligands replaced by aqua/hydroxo; Ca(II) is closed-shell and is "
                   "not a network site, though it does mediate structure",
        physics_note="The hierarchical case, and the one closest to a real hard problem. The "
                     "Mn3Ca cubane face gives an odd cycle (frustrated), and the dangler Mn4 "
                     "hangs off it as a tree branch - so neither a pure loop network nor a "
                     "pure tree fits, and the network must handle both in one structure. "
                     "Cycle rank 1 with a pendant vertex: the smallest topology that forces "
                     "a mixed treatment. Oxidation states are the S1 Mn(III)2Mn(IV)2 "
                     "assignment; the point here is the topology, not the state assignment",
        provenance="Umena, Kawakami, Shen, Kamiya, Nature 473, 55 (2011)",
    ),
    Tier3System(
        key="fe4s4",
        label="[4Fe-4S] ferredoxin cubane (2+ state)",
        formula="[Fe4S4(SH)4]2-",
        sites=(Site("Fe1", "Fe(III)", 5),
               Site("Fe2", "Fe(III)", 5),
               Site("Fe3", "Fe(II)", 4),
               Site("Fe4", "Fe(II)", 4)),
        edges=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        topology="complete graph K4",
        network_target="PEPS-like; the worst case for any 1D ordering",
        cas_electrons=54, cas_orbitals=36,
        truncation="cysteinate replaced by SH-, the standard computational truncation of the "
                   "biological cluster; the Fe4S4 core is untouched",
        physics_note="Maximal connectivity on four sites: every centre couples to every other "
                     "through the mu3-sulfides, so the graph is K4 with cycle rank 3. There "
                     "is no ordering of the sites that makes the interactions short-ranged, "
                     "which is precisely why Fe-S clusters are the standard hard benchmark "
                     "for DMRG - any 1D network pays for it in bond dimension. Non-bipartite, "
                     "so frustrated, and the largest active space here: CAS(54,36) is ~10^11 "
                     "determinants and utterly beyond conventional CI",
        provenance="Beinert, Holm, Munck, Science 277, 653 (1997); DMRG benchmark: Sharma, "
                   "Sivalingam, Neese, Chan, Nat. Chem. 6, 927 (2014)",
    ),
    Tier3System(
        key="cr8_ring",
        label="Cr(III)8 antiferromagnetic ring",
        formula="[Cr8F8(HCO2)16]",
        sites=tuple(_ions("Cr(III)", 3, 8, "Cr")),
        edges=_ring_edges(8),
        topology="cycle C8",
        network_target="periodic MPS / one long-range bond",
        cas_electrons=24, cas_orbitals=40,
        experimental_twice_spin=0,
        truncation="pivalate replaced by formate; the Cr8F8 ring core is intact",
        physics_note="An even ring is bipartite despite being a cycle, so Lieb-Mattis still "
                     "applies: four Cr(III) per sublattice, S_A = S_B = 6, giving S = 0 "
                     "exactly - the experimentally established ground state. The topology "
                     "is still a loop, so an open MPS must carry one long-range bond closing "
                     "the ring; this is the system that shows whether that is handled or "
                     "quietly ignored",
        provenance="van Slageren et al., Chem. Eur. J. 8, 277 (2002)",
    ),
    Tier3System(
        key="cr7ni_ring",
        label="Cr(III)7Ni(II) ring (qubit candidate)",
        formula="[Cr7NiF8(HCO2)16]-",
        sites=tuple(_ions("Cr(III)", 3, 7, "Cr")) + (Site("Ni8", "Ni(II)", 2),),
        edges=_ring_edges(8),
        topology="cycle C8 with one substituted site",
        network_target="periodic MPS; identical topology, different local quantum numbers",
        cas_electrons=25, cas_orbitals=40,
        experimental_twice_spin=1,
        truncation="pivalate replaced by formate; the Cr7NiF8 ring core is intact",
        physics_note="The matched partner of cr8_ring, and the sharpest test in this tier. "
                     "Identical graph, one Cr(III) (S = 3/2) replaced by Ni(II) (S = 1), "
                     "which unbalances the sublattices: S_A = 6, S_B = 4 + 1 = 11/2, so "
                     "Lieb-Mattis gives S = 1/2 - the experimentally established ground "
                     "state and the reason this molecule is studied as a qubit. A network "
                     "that has the topology right but the local quantum numbers wrong "
                     "reproduces cr8_ring and fails here, which is exactly the bug this pair "
                     "is designed to catch (compare ce3p/yb3p in Tier 2)",
        provenance="Larsen et al., Phys. Rev. Lett. 91, 067201 (2003); Timco et al., "
                   "Nat. Nanotechnol. 4, 173 (2009)",
    ),
)

SYSTEMS_BY_KEY: Dict[str, Tier3System] = {s.key: s for s in SYSTEMS}


def get(key: str) -> Tier3System:
    if key not in SYSTEMS_BY_KEY:
        raise KeyError(f"unknown Tier-3 system {key!r}; known: {sorted(SYSTEMS_BY_KEY)}")
    return SYSTEMS_BY_KEY[key]


def by_network(target: str) -> Tuple[Tier3System, ...]:
    return tuple(s for s in SYSTEMS if s.network_target == target)


__all__ = [
    "CONVENTIONAL_CI_SPINOR_CEILING", "Site", "Tier3System", "SYSTEMS", "SYSTEMS_BY_KEY",
    "adjacency", "bipartition", "by_network", "connected_components", "couple_spins",
    "cycle_rank", "degree_sequence", "get", "graph_diameter", "heisenberg_ground_state",
    "is_frustrated", "is_tree", "ED_MAX_DIM",
    "lieb_mattis_twice_spin",
]
