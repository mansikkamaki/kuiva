"""Tier 3: tensor-network structure tests for polynuclear systems.

The other tiers ask whether Kuiva reproduces a number another program also produces. This
one cannot: every system here is deliberately beyond conventional CI, so PySCF, OpenMolcas
and DIRAC have nothing to say about them (see ``tests/generate/tier3_systems.py``).

What is asserted instead is everything that is true *by theorem or by counting*:

1. **Reference integrity** - the stored invariants still match what the definitions imply,
   so an edit to a spin or an exchange pathway cannot pass unnoticed.
2. **Exact spin algebra** - Clebsch-Gordan decompositions, Kramers parity.
3. **Lieb-Mattis** - the exact ground-state total spin of the bipartite antiferromagnets,
   cross-checked against dense ED of the spin model *and* against experiment where known.
4. **Topology** - the graph invariants that decide which tensor network is appropriate,
   and the consistency of each system's declared ``network_target`` with them.
5. **Intractability** - that each system really is beyond the conventional-CI ceiling,
   which is the justification for this tier existing at all. If one of these ever became
   cheap enough for a Tier-1/2 reference, it belongs in that tier instead.
6. **Method placeholders** - what Kuiva's MPS/TTNS code must reproduce, skipped until
   ``kuiva.dmrg`` exists, so they light up on their own.

Nothing here needs an active space, an integral, or a basis set: these are statements about
the *structure* of the problem, which is exactly what a tensor network has to get right.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import pytest

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "tests/reference/tier3_invariants.json"

sys.path.insert(0, str(REPO / "tests/generate"))
import tier3_systems as t3  # noqa: E402

#: Ground-state energies of an isotropic spin model are exact rationals; ED reaches them to
#: machine precision, so this only has to absorb floating-point noise.
ED_TOL = 1e-9


@pytest.fixture(scope="module")
def ref() -> Dict:
    if not REF.is_file():
        pytest.skip(f"{REF.relative_to(REPO)} missing; run tests/generate/tier3_invariants.py")
    return json.loads(REF.read_text())


@pytest.fixture(scope="module")
def records(ref) -> Dict:
    return ref["records"]


def _rec(records: Dict, key: str) -> Dict:
    if key not in records:
        pytest.skip(f"no Tier-3 record for {key}")
    return records[key]


ALL_KEYS = [s.key for s in t3.SYSTEMS]
BIPARTITE_KEYS = [s.key for s in t3.SYSTEMS if t3.lieb_mattis_twice_spin(s) is not None]
EXPERIMENTAL_KEYS = [s.key for s in t3.SYSTEMS if s.experimental_twice_spin is not None]


# --- 1. reference integrity -----------------------------------------------------------------
def test_every_system_has_a_record(records):
    """No system may silently disappear from the reference file.

    Same guard as Tier 1: records are fetched through a helper that *skips* on a missing key,
    so without this a dropped system turns its tests green by omission.
    """
    missing = sorted(set(ALL_KEYS) - set(records))
    assert not missing, (
        f"{len(missing)} Tier-3 record(s) missing: {missing}\n"
        f"Regenerate with:  python tests/generate/tier3_invariants.py"
    )


@pytest.mark.parametrize("key", ALL_KEYS)
def test_stored_invariants_match_the_definitions(records, key):
    """The committed file must still agree with ``tier3_systems.py``.

    This is the whole point of committing a derived reference: changing an oxidation state or
    an exchange pathway changes the physics, and it should show up as a failing test plus a
    reviewable diff, not as a quietly different suite.
    """
    rec = _rec(records, key)
    system = t3.get(key)
    assert rec["n_sites"] == system.n_sites
    assert tuple(rec["local_dims"]) == system.local_dims
    assert rec["hilbert_dim"] == system.hilbert_dim
    assert rec["cycle_rank"] == t3.cycle_rank(system)
    assert rec["bipartite"] == (t3.bipartition(system) is not None)
    assert rec["lieb_mattis_twice_spin"] == t3.lieb_mattis_twice_spin(system)
    assert rec["cas_determinants"] == system.cas_determinants
    assert rec["kramers_system"] == system.kramers_system


# --- 2. exact spin algebra ------------------------------------------------------------------
@pytest.mark.parametrize("key", ALL_KEYS)
def test_spin_decomposition_is_complete(key):
    """sum_S (2S+1) * n_S == prod_i (2S_i + 1).

    Clebsch-Gordan decomposition must conserve the dimension of the product space. This is
    the cheapest possible check that the coupling routine is right, and it is exactly the
    bookkeeping a quantum-number-labelled tensor has to get right to be block-sparse
    without losing states.
    """
    system = t3.get(key)
    decomposition = t3.couple_spins([s.twice_spin for s in system.sites])
    total = sum((two_s + 1) * n for two_s, n in decomposition.items())
    assert total == system.hilbert_dim, \
        f"{key}: spin decomposition spans {total} states, product space has {system.hilbert_dim}"


@pytest.mark.parametrize("key", ALL_KEYS)
def test_kramers_parity(key):
    """Half-integer total spins occur exactly when the electron count is odd.

    Kramers' theorem: an odd number of unpaired electrons forces every level to be at least
    doubly degenerate, and the available total spins are then all half-integer. Getting this
    parity wrong in a quantum-number label is a classic time-reversal bookkeeping bug,
    and it is free to check.
    """
    system = t3.get(key)
    decomposition = t3.couple_spins([s.twice_spin for s in system.sites])
    odd_electrons = system.kramers_system
    for two_s in decomposition:
        assert (two_s % 2 == 1) == odd_electrons, (
            f"{key}: total 2S={two_s} has the wrong parity for "
            f"{system.total_unpaired_electrons} unpaired electrons"
        )


# --- 3. Lieb-Mattis, versus ED and versus experiment ----------------------------------------
@pytest.mark.parametrize("key", BIPARTITE_KEYS)
def test_lieb_mattis_ground_spin_is_available_and_in_range(key):
    """For a bipartite AF the ground spin is |S_A - S_B| exactly, and must be a spin the
    coupled space actually contains."""
    system = t3.get(key)
    two_s = t3.lieb_mattis_twice_spin(system)
    assert two_s is not None
    decomposition = t3.couple_spins([s.twice_spin for s in system.sites])
    assert two_s in decomposition, \
        f"{key}: Lieb-Mattis predicts 2S={two_s}, which is not in the coupled space"


@pytest.mark.parametrize("key", EXPERIMENTAL_KEYS)
def test_lieb_mattis_reproduces_the_experimental_ground_spin(records, key):
    """**The most valuable checks in this tier.**

    ``fe4_star`` (S = 5), ``cr8_ring`` (S = 0) and ``cr7ni_ring`` (S = 1/2) have
    experimentally established ground-state spins. Those numbers depend on no program at
    all - the same reason the analytic Lande g values anchor Tier 2. Agreement here
    means the system definition (spins, oxidation states, connectivity) describes the real
    molecule, which is the assumption every other Tier-3 assertion rests on.
    """
    system = t3.get(key)
    predicted = t3.lieb_mattis_twice_spin(system)
    assert predicted == system.experimental_twice_spin, (
        f"{key}: Lieb-Mattis gives 2S={predicted}, experiment gives "
        f"2S={system.experimental_twice_spin}"
    )
    assert _rec(records, key)["experimental_twice_spin"] == system.experimental_twice_spin


@pytest.mark.parametrize("key", ALL_KEYS)
def test_stored_ed_agrees_with_lieb_mattis(records, key):
    """Dense ED of the spin model against the theorem, for every system where both apply.

    A theorem is only as good as its implementation: this catches a wrong sublattice
    assignment or a mis-transcribed spin, which would otherwise produce a confident and
    wrong "exact" answer.
    """
    rec = _rec(records, key)
    ed, lm = rec["heisenberg_ed"], rec["lieb_mattis_twice_spin"]
    if ed is None or lm is None:
        pytest.skip(f"{key}: no ED and/or no Lieb-Mattis prediction (frustrated or SOC-dominated)")
    assert ed["twice_total_spin"] == lm, \
        f"{key}: ED ground state 2S={ed['twice_total_spin']} vs Lieb-Mattis 2S={lm}"
    # A non-degenerate total-spin ground state has exactly the 2S+1 magnetic sublevels.
    assert ed["degeneracy"] == lm + 1, \
        f"{key}: ground degeneracy {ed['degeneracy']} != 2S+1 = {lm + 1}"
    assert ed["s2_spread"] <= 1e-6, f"{key}: ground manifold is not a pure spin state"


def test_live_ed_reproduces_the_stored_chain_result(records):
    """Recompute the smallest ED here rather than only trusting the stored value.

    Cheap freshness check (216x216): if the spin-model code changes, the committed reference
    stops matching and this fails immediately instead of at the next full regeneration.
    """
    system = t3.get("mn3_linear")
    live = t3.heisenberg_ground_state(system)
    stored = _rec(records, "mn3_linear")["heisenberg_ed"]
    assert live is not None
    assert live["twice_total_spin"] == stored["twice_total_spin"]
    assert live["degeneracy"] == stored["degeneracy"]
    assert live["ground_energy"] == pytest.approx(stored["ground_energy"], abs=ED_TOL)


@pytest.mark.slow
@pytest.mark.parametrize("key", ALL_KEYS)
def test_live_ed_reproduces_every_stored_result(records, key):
    """The same freshness check for every ED-able system. ~2 s in total, so it is opt-in."""
    stored = _rec(records, key)["heisenberg_ed"]
    live = t3.heisenberg_ground_state(t3.get(key))
    if stored is None:
        assert live is None, f"{key}: reference has no ED but one is now computable"
        pytest.skip(f"{key}: ED not applicable")
    assert live is not None, f"{key}: reference has ED but it is no longer computable"
    assert live["twice_total_spin"] == stored["twice_total_spin"]
    assert live["degeneracy"] == stored["degeneracy"]
    assert live["ground_energy"] == pytest.approx(stored["ground_energy"], abs=ED_TOL)


def test_frustrated_systems_get_no_lieb_mattis_prediction():
    """Frustration must be reported, not papered over.

    Lieb-Mattis requires a bipartite graph. On an odd cycle there is no such partition and the
    theorem simply does not apply - the honest output is *no prediction*. A helper that
    returned a plausible number anyway would be worse than useless, because the value would
    look authoritative. This asserts the negative result explicitly.
    """
    frustrated = [s.key for s in t3.SYSTEMS if t3.is_frustrated(s)]
    assert frustrated, "no frustrated system in the suite - the loop topologies are missing"
    for key in frustrated:
        system = t3.get(key)
        assert t3.bipartition(system) is None, f"{key} is marked frustrated but is bipartite"
        assert t3.lieb_mattis_twice_spin(system) is None, \
            f"{key} is frustrated, so Lieb-Mattis must decline to predict"


def test_frustrated_triangle_has_the_known_degenerate_ground_state(records):
    """The equilateral AF triangle: S = 1/2 ground state, four-fold degenerate.

    Textbook result for three coupled half-integer spins on an odd cycle - the two-fold spin
    degeneracy is multiplied by a two-fold *chirality* degeneracy that frustration leaves
    unresolved. This is the structure a tensor network must not accidentally split, and it is
    exactly what a network with a wrongly-symmetrised loop gets wrong.
    """
    ed = _rec(records, "fe3_oxo")["heisenberg_ed"]
    assert ed["twice_total_spin"] == 1, "equilateral AF triangle must have an S = 1/2 ground state"
    assert ed["degeneracy"] == 4, "expected 2 (spin) x 2 (chirality) = 4-fold degeneracy"


def test_complete_graph_spectrum_is_exactly_analytic(records):
    """``fe4s4``: uniform J on K4 gives H = (J/2)(S_tot^2 - sum_i S_i(S_i+1)).

    Every centre couples to every other with the same J, so the Hamiltonian depends only on
    the *total* spin - the energy is an exact function of S_tot and the ground state is simply
    the lowest S it contains. The ground degeneracy is then the number of S = 0 multiplets in
    the Clebsch-Gordan decomposition, a pure counting statement. So the topologically worst
    case in this tier happens to have an exactly known spectrum, which makes it a very sharp
    test for a network that has to reach it the hard way.
    """
    system = t3.get("fe4s4")
    ed = _rec(records, "fe4s4")["heisenberg_ed"]
    decomposition = t3.couple_spins([s.twice_spin for s in system.sites])
    lowest_two_s = min(decomposition)
    assert ed["twice_total_spin"] == lowest_two_s, \
        "uniform J on a complete graph puts the lowest total spin lowest"
    assert ed["degeneracy"] == decomposition[lowest_two_s] * (lowest_two_s + 1), (
        "ground degeneracy must be (number of S_min multiplets) x (2 S_min + 1)"
    )


# --- 4. topology decides the network --------------------------------------------------------
@pytest.mark.parametrize("key", ALL_KEYS)
def test_graph_is_well_formed(key):
    """Connected, simple, and with every edge index in range."""
    system = t3.get(key)
    assert system.n_sites >= 3, "Tier 3 is for three or more paramagnetic centres"
    seen = set()
    for i, j in system.edges:
        assert i != j, f"{key}: self-loop on site {i}"
        assert 0 <= i < system.n_sites and 0 <= j < system.n_sites, f"{key}: edge out of range"
        pair = (min(i, j), max(i, j))
        assert pair not in seen, f"{key}: duplicate edge {pair}"
        seen.add(pair)
    assert t3.connected_components(system) == 1, \
        f"{key}: the exchange graph must be connected, or it is really two systems"


@pytest.mark.parametrize("key", ALL_KEYS)
def test_declared_network_target_matches_the_topology(key):
    """A system's declared target network must follow from its graph, not from a label.

    The rule this encodes: a tree (cycle rank 0) is exactly representable by a TTNS,
    of which an MPS is the special case where every vertex has degree <= 2. Cycles cannot be
    represented exactly by any tree network, so a system with cycle rank > 0 must not be
    declared as a plain MPS/TTNS target.
    """
    system = t3.get(key)
    rank, max_degree = t3.cycle_rank(system), max(t3.degree_sequence(system))
    target = system.network_target.lower()
    if rank == 0 and max_degree <= 2:
        assert "mps" in target, f"{key}: a path graph should be an MPS target"
    elif rank == 0:
        assert "ttns" in target or "tree" in target, \
            f"{key}: a branching tree should be a TTNS target"
    else:
        assert "mps" not in target or "periodic" in target or "long-range" in target, (
            f"{key}: cycle rank {rank} > 0, so a plain MPS target is wrong - it needs a "
            f"periodic/long-range bond or a loopy network"
        )


def test_the_suite_spans_the_intended_topology_classes():
    """The tier is only useful if the connectivity classes are actually distinct.

    Guards against the suite quietly collapsing to several copies of one topology, which is
    the failure mode when systems are added for chemical interest rather than for structure.
    """
    ranks = {t3.cycle_rank(s) for s in t3.SYSTEMS}
    assert {0, 1}.issubset(ranks), "need both tree-like and single-loop systems"
    assert max(ranks) >= 3, "need a densely connected (PEPS-like) case such as K4"

    assert any(t3.is_tree(s) and max(t3.degree_sequence(s)) == 2 for s in t3.SYSTEMS), \
        "need a plain chain (MPS baseline)"
    assert any(t3.is_tree(s) and max(t3.degree_sequence(s)) >= 3 for s in t3.SYSTEMS), \
        "need a branching tree (TTNS)"
    assert any(len(set(s.local_dims)) > 1 for s in t3.SYSTEMS), \
        "need inhomogeneous local dimensions (ion + radical)"
    assert any(s.kind == "ion_soc" for s in t3.SYSTEMS for s in s.sites), \
        "need a SOC-dominated centre, where a spin-only picture is invalid"


def test_the_matched_ring_pair_differs_only_by_one_site():
    """``cr8_ring`` vs ``cr7ni_ring``: same graph, one substitution, different ground spin.

    The Tier-3 analogue of ``ce3p``/``yb3p``. Identical topology means a network that
    is right about connectivity but wrong about local quantum numbers reproduces the first and
    fails the second, which is precisely the bug this pair exists to catch.
    """
    a, b = t3.get("cr8_ring"), t3.get("cr7ni_ring")
    assert a.edges == b.edges, "the matched pair must share a topology exactly"
    assert a.n_sites == b.n_sites
    differing = [k for k in range(a.n_sites) if a.sites[k].twice_spin != b.sites[k].twice_spin]
    assert len(differing) == 1, "exactly one site may differ"
    assert t3.lieb_mattis_twice_spin(a) != t3.lieb_mattis_twice_spin(b), \
        "the substitution must change the ground-state spin, or the pair proves nothing"


# --- 5. these systems really are out of reach of the other tiers ----------------------------
@pytest.mark.parametrize("key", ALL_KEYS)
def test_system_is_beyond_conventional_ci(records, key):
    """Justifies the tier's existence, and keeps it honest.

    The conventional-CI ceiling sits at ~12-14 spinors. Every system here is far past
    it, which is *why* no Tier-1/Tier-2 reference can exist. If a system ever fell below the
    ceiling it should be moved to Tier 1/2 and given a real reference calculation instead of
    being asserted about from theory.
    """
    system = t3.get(key)
    rec = _rec(records, key)
    assert system.beyond_conventional_ci, (
        f"{key}: CAS({system.cas_electrons},{system.cas_orbitals}) needs "
        f"{system.cas_spinors} spinors, within the conventional-CI ceiling of "
        f"{t3.CONVENTIONAL_CI_SPINOR_CEILING} - it belongs in Tier 1/2 with a real reference"
    )
    assert rec["cas_determinants"] > 10 ** 7, \
        f"{key}: {rec['cas_determinants']:.2e} determinants is not obviously intractable"


# --- 6. what Kuiva must eventually reproduce -------------------------------------------------
@pytest.mark.parametrize("key", BIPARTITE_KEYS)
def test_kuiva_dmrg_ground_spin(key):
    """Kuiva's DMRG must land on the Lieb-Mattis ground-state spin.

    When enabled: run the DMRG solver on the effective spin model for this topology and check
    the converged total spin against the theorem. This is a rigorous target that needs no
    reference calculation, which is what makes it usable at all for systems this size.
    """
    pytest.importorskip("kuiva.dmrg.sweep", reason="kuiva.dmrg not implemented yet ")
    pytest.skip("enable once dmrg/sweep.py can converge an effective spin model")


def test_kuiva_ttns_beats_mps_on_the_star():
    """``fe4_star`` is the system where a tree network should demonstrably win.

    A star cannot be ordered so that all three branches are nearest-neighbour, so an MPS must
    carry a long-range bond while a TTNS matches the topology exactly. When enabled, this
    should compare bond dimension at fixed accuracy - the concrete payoff of the NetworkGraph
    abstraction being genuinely abstract.
    """
    pytest.importorskip("kuiva.dmrg.graph", reason="kuiva.dmrg not implemented yet ")
    pytest.skip("enable once NetworkGraph supports both MPS and TTNS topologies")


def test_kuiva_orbital_ordering_recovers_the_exchange_graph():
    """Entanglement-driven ordering should rediscover the connectivity.

    For ``mn3_linear`` the mutual-information/Fiedler ordering should return the natural chain
    order, and for ``fe4_star`` it should place the central Fe between the peripheral ones.
    The exchange graph is the ground truth the ordering heuristic is trying to find, so this
    tests the heuristic against a known answer rather than against itself.
    """
    pytest.skip("orbital ordering is entanglement-driven and lives in rdm/entropy.py "
                "(Fiedler) plus dmrg/guess.py's clustering; a Tier-3-scale ordering check "
                "needs a network solve on a system beyond the conventional-CI ceiling, which is not a "
                "fast-suite test")


def test_kuiva_multisite_handles_inhomogeneous_local_dimensions():
    """``dy2_n2rad``: local dimensions (16, 2, 16), and SOC too large for a spin-only picture.

    A network that assumes a uniform local dimension, or that a site is a spin multiplet
    rather than a J multiplet, silently does the wrong thing here rather than failing loudly -
    which is why this system is in the suite (local multiplets).
    """
    # the machinery is `kuiva.dmrg.manifold`;
    # its inhomogeneous-local-dimension behaviour is covered Tier-0 by the fragment
    # oracles of tests/test_dmrg_manifold.py. This system is f-block and beyond the
    # fast suite by design (the cheapest system that shows the structure).
    pytest.importorskip("kuiva.dmrg.manifold")
    pytest.skip("needs an f-dimer network solve; Tier-0 coverage is test_dmrg_manifold.py")
