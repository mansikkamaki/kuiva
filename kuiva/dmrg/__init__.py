"""In-house tensor-network (DMRG/TTNS) layer.

Tree-native from day one: the block-sparse tensor core (``block``) and the tree topology
owner (``graph``) exist for arbitrary trees, with the matrix-product chain as the path-graph
special case. The operator (TTNO), sweep, adaptive-reconnection and local-multiplet layers
are built on top of these.

**Deliberately still open** (recorded here because none of it constrains
code outside this package):

(a) **Ensemble-aware acceptance and site-cut criteria.** Absolute-entropy margins mis-signal
on state-averaged product families, where a product family carries *classical* label
correlation across an inter-site bond. Until this is settled, use ``entropy`` acceptance only
for uncapped structure discovery and ``weight`` for any capped run, every CASSCF included.

(b) **Global topology search**, deferred behind (a).

(c) **Energy-window and J-manifold multiplet rules**, and **non-uniform ensemble
re-weighting**. The pluggable slot exists (``manifold``); ``dimension``/``weight``/``gap`` are
realised.

(d) **A Kramers-restricted network** — the analogue of the second CI mode of the conventional-CI layer,
to be validated against the general one.

(e) **The multi-site ab initio manifold rung.** Whether a trimer-scale TTNO fits a given
memory limit is unmeasured, the compile being beyond the ten-minute ad-hoc budget.

(f) **A JW-relabeling move class** — the only way tree moves could subsume orbital ordering
for fermions, since the Jordan-Wigner order is the global ascending mode index and a leaf swap
is therefore not a reordering. Possible, costed, not planned.
"""
from .block import (BlockTensor, FuseRecord, QuantumNumber, Space, TruncationInfo,
                    SCHMIDT_DEGENERACY_RTOL, SCHMIDT_STABILITY_RTOL, fuse, qr, split, svd)
from .graph import NetworkGraph
from .sparse import SparseW, dot_sparse, sparse_w_gb
from .ttno import (FERMION_MODE, ModeBasis, ProductTerm, TTNO, TTNOTemplate,
                   compile_ttno, consolidate, fermion_term,
                   hamiltonian_product_terms, one_electron_product_terms,
                   ttno_from_cas_integrals)
from .sweep import (SweepResult, TTNState, random_state, solve_ttn, state_gb,
                    state_to_dense, BOUNDARY_GAP_WARN_CM)
from .guess import (TopologyGuess, expansion_to_ttn, topology_from_mutual_information,
                    DEFAULT_SITE_SPLIT)
from .reconnect import (AdaptiveResult, BondReport, Move, ReconnectionPolicy,
                        SiteReport, StructureReport, discovered_structure,
                        solve_adaptive)
from .manifold import (EffectiveModel, ManifoldResult, SiteSpace, UnderResolved,
                       effective_model, effective_operator, model_gb, site_spaces,
                       solve_manifold, MULTIPLET_GAP_RATIO_WARN,
                       DEFAULT_MULTIPLET_WEIGHT_TOL)
from .density import (annihilation_term, network_rdm, network_rdms,
                      node_environments)
from .extrapolate import BondSeriesResult, bond_series
from .checkpoint import (NETWORK_SCHEMA_VERSION, NetworkCheckpointError,
                         NetworkCheckpointPolicy, network_checkpoint_gb,
                         network_state_path, read_network_state, write_network_state)
from .solver import DMRGSolver, NetworkProposal

__all__ = ["QuantumNumber", "Space", "BlockTensor", "FuseRecord", "TruncationInfo",
           "fuse", "split", "qr", "svd", "NetworkGraph",
           "SparseW", "dot_sparse", "sparse_w_gb",
           "SCHMIDT_DEGENERACY_RTOL", "SCHMIDT_STABILITY_RTOL",
           "ModeBasis", "FERMION_MODE", "ProductTerm", "TTNO", "fermion_term",
           "consolidate", "hamiltonian_product_terms", "one_electron_product_terms",
           "compile_ttno", "ttno_from_cas_integrals",
           "TTNState", "SweepResult", "random_state", "solve_ttn", "state_gb",
           "state_to_dense", "BOUNDARY_GAP_WARN_CM",
           "TopologyGuess", "expansion_to_ttn", "topology_from_mutual_information",
           "DEFAULT_SITE_SPLIT",
           "AdaptiveResult", "BondReport", "Move", "ReconnectionPolicy", "SiteReport",
           "StructureReport", "discovered_structure", "solve_adaptive",
           "EffectiveModel", "ManifoldResult", "SiteSpace", "UnderResolved",
           "effective_model", "effective_operator", "model_gb", "site_spaces",
           "solve_manifold", "MULTIPLET_GAP_RATIO_WARN",
           "DEFAULT_MULTIPLET_WEIGHT_TOL",
           "TTNOTemplate", "annihilation_term", "network_rdm", "network_rdms",
           "node_environments", "DMRGSolver", "NetworkProposal",
           "NETWORK_SCHEMA_VERSION", "NetworkCheckpointError", "NetworkCheckpointPolicy",
           "network_checkpoint_gb", "network_state_path", "read_network_state",
           "write_network_state", "BondSeriesResult", "bond_series"]
