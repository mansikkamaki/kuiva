"""The class API (kuiva/interface/stages.py): the uniform stage contract, end to end.

Covers the three calculation shapes the layer was designed on (the class-API design scope),
on the cheapest system that exhibits each structure:

* conventional-CI CASSCF + property dump — B (2p^1), whose ground j = 1/2 doublet has the
  **analytic** Lande factor g = 1 - (g_e - 1)/3 = 0.66589, a target no convention of any
  code can move;
* CASSCF + SC-NEVPT2 + the corrected (hybrid-protocol) dump on the same reference;
* DMRG-CASSCF + the pseudospin export, whose g values must agree with the dump's
  phase-invariant reduction — two independent property routes to one invariant.

Tolerances: state degeneracies inside a Kramers doublet are asserted at 1e-10 Eh (the
general path's measured splitting is 1e-15..1e-13 Eh); CI-vs-DMRG energies at 1e-8 Eh
; g against Lande at 2e-3 on a term-complete average (see below);
dump-vs-pseudospin g at 1e-6 (same states, two contractions).

⚠ **A g value may only be asserted against Lande on an average over the WHOLE term**, which
is why ``cas_term`` exists beside ``cas``. Averaging the j = 1/2 doublet alone is a complete
manifold — the degeneracy gate and the boundary check both pass it — and it is still not an
ensemble the term's symmetry leaves invariant, so the residual anisotropy is whatever the
run's rounding noise made it: measured on this system as **6.6e-4 under one BLAS and 3.9e-2
under another**, from SCF energies agreeing to 1e-11. Over all six roots the same quantity is
6.5e-4 and 3.6e-5, an order inside the band. The two-root fixtures stay because the claims
they carry — the stage contract, solver agreement, restart, the two property routes agreeing
with each other — are indifferent to it; only the analytic-target claim is not.

⚠ **The slide needs a seed, and that is what decides which fixture may be a two-root one.**
Started from the canonical (spherical) orbitals there is no pairing defect to amplify and the
two-root runs are reproducible. Started from a pre-optimization the amplification is real:
2e-8 … 5e-8 Eh of Kramers splitting in about one run in eight, and once 1.0e-6 Eh, at which
point the state-averaging gate refuses the run outright — correctly. So everything downstream
of ``pre`` averages the whole term, and ``pre`` itself does too, because what it hands on is
orbitals.
"""
import numpy as np
import pytest

import kuiva
from kuiva.interface.stages import (CASSCF, CheapCI, NEVPT2, PropertyDump,
                                    PseudospinExport, Reference, ScalarSCF)

G_E = 2.00231930436256
G_LANDE = {2: 1.0 - (G_E - 1.0) / 3.0,              # p^1, j = 1/2, with the real g_e
           4: 1.0 + (G_E - 1.0) / 3.0}              # p^1, j = 3/2
E_TOL = 1e-8                                        # suite energy tolerance [Eh]
KRAMERS_TOL = 1e-10                                 # doublet degeneracy [Eh]
#: A free atom's j levels are degenerate as a matter of physics; 0.1 cm^-1 would already be
#: different physics, so that is the bar for a level, not the numerical KRAMERS_TOL.
LEVEL_TOL = 0.1 / 219474.63                         # [Eh]


def boron() -> kuiva.Molecule:
    return kuiva.Molecule([("B", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c", spin=1)


@pytest.fixture(scope="module")
def scf():
    # screening="none": the 2e-SOC picture change is pure cost on every assertion here
    # (pytest may not depend on a warm AMF cache); the 1e SOC stays on, which is
    # what the g values need.
    return ScalarSCF(boron(), memory_gb=4.0, screening="none").run()


@pytest.fixture(scope="module")
def ref(scf):
    return Reference(scf).run()


@pytest.fixture(scope="module")
def cas(ref):
    return CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=2,
                  report=False).run()


@pytest.fixture(scope="module")
def cas_term(ref):
    # every root of the 2p determinant space: the whole ^2P term, so the averaged density is
    # the one the term's symmetry leaves invariant and the g values are the analytic ones
    return CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=6,
                  report=False).run()


@pytest.fixture(scope="module")
def cas_dmrg(ref):
    return CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=2,
                  solver="dmrg", solver_options=dict(max_bond=16), report=False).run()


# --- the uniform contract --------------------------------------------------------------------

def test_unrun_upstream_is_refused():
    unrun = ScalarSCF(boron(), memory_gb=4.0, screening="none")
    with pytest.raises(ValueError, match="has not been run"):
        Reference(unrun)


def test_wrong_stage_type_is_refused(ref):
    with pytest.raises(TypeError, match="finished ScalarSCF"):
        Reference(ref)


def test_unknown_options_fail_at_construction(scf, ref):
    # eager validation is the contract: a typo fails before anything expensive runs
    with pytest.raises(TypeError, match="screning"):
        ScalarSCF(boron(), screning="none")
    with pytest.raises(TypeError, match="max_itr"):
        CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, max_itr=5)
    with pytest.raises(ValueError, match="reference"):
        ScalarSCF(boron(), reference="hf")


def test_results_exist_only_on_a_finished_stage(scf):
    stage = Reference(scf)
    with pytest.raises(RuntimeError, match="run"):
        stage.summary()
    with pytest.raises(RuntimeError, match="run"):
        stage.nspinor


def test_run_is_idempotent(scf):
    assert scf.run() is scf
    data = scf.data
    assert scf.run().data is data


def test_an_active_space_must_be_stated(ref):
    with pytest.raises(ValueError, match="active space"):
        CASSCF(ref, n_states=2)


def test_summary_is_text(scf, ref, cas):
    for stage in (scf, ref, cas):
        text = stage.summary()
        assert type(stage).__name__ in text and "\n" in text


# --- shape 1: conventional-CI CASSCF + property dump -----------------------------------------

def test_ci_casscf_ground_doublet(cas):
    assert cas.converged
    assert cas.energies.size == 2
    # the two states are one Kramers pair; their splitting is the general path's numerical
    # noise (1e-15..1e-13 Eh measured), asserted well above that but far below physics
    assert abs(cas.energies[1] - cas.energies[0]) < KRAMERS_TOL
    assert cas.boundary_initial is not None and cas.boundary is not None
    # the j = 1/2-only average leans on the spin-orbit structure (module docstring: its
    # residual anisotropy is whatever the rounding made it), and the converged boundary
    # report now says so — measured 0.64 here, 0 on the term-complete average
    assert cas.boundary.spin_noninvariance is not None
    assert cas.boundary.spin_noninvariance > 0.3 and cas.boundary.leaning is True


def test_property_dump_gives_the_lande_g(cas_term, tmp_path):
    # the term-complete average (see the module docstring): the 2p shell splits into
    # j = 1/2 and j = 3/2, each level degenerate by spherical symmetry, each carrying its
    # analytic Lande factor -- a target no convention of any code can move
    dump = PropertyDump(cas_term, tmp_path / "b.props", report=False).run()
    assert (tmp_path / "b.props").exists()
    levels = dump.matrices.analyse()
    assert [level.size for level in levels] == [2, 4]
    assert abs(cas_term.energies[1] - cas_term.energies[0]) < KRAMERS_TOL
    assert float(np.ptp(cas_term.energies[2:])) < LEVEL_TOL
    assert all(abs(g - G_LANDE[level.size]) < 2e-3
               for level in levels for g in level.g_values)
    # the flip side of the leaning assertion on `cas`: the term-complete ensemble is one the
    # symmetry leaves invariant, and the report measures that as an exact zero (1e-16 here)
    assert cas_term.boundary.spin_noninvariance < 1e-8
    assert cas_term.boundary.leaning is False


# --- shape 2: NEVPT2 on the converged reference ----------------------------------------------

def test_nevpt2_stage(cas, tmp_path):
    pt = NEVPT2(cas, report=False).run()
    assert pt.result.complete
    assert float(pt.e2[0]) < 0.0
    # the correction may not split the Kramers doublet (physical requirement, not a band)
    assert abs(pt.total_energies[1] - pt.total_energies[0]) < 1e-8
    dump = PropertyDump(pt, tmp_path / "b_pt.props", report=False).run()
    nev = dump.matrices.provenance["nevpt2"]
    assert nev["flavor"] == "SC-NEVPT2" and nev["complete"]
    assert np.allclose(dump.matrices.energies, np.asarray(pt.total_energies, dtype=float))


def test_nevpt2_on_a_network_reference_is_loudly_partial(cas, cas_dmrg, kuiva_caplog):
    """The network route reaches NEVPT2 through the provider seam — six of eight classes,
    and the partiality is visible everywhere it could mislead."""
    pt_ci = NEVPT2(cas, report=False).run()
    pt_net = NEVPT2(cas_dmrg, report=False).run()
    assert pt_ci.result.complete
    assert not pt_net.result.complete
    assert pt_net.result.missing == ("Sr", "Si")
    assert any("PARTIAL" in r.getMessage() for r in kuiva_caplog.records)
    # same molecule, independently converged orbitals: the six served classes agree to
    # how well the two CASSCFs agree, far tighter than any physical statement
    served = [n for n in pt_ci.class_energies if n not in ("Sr", "Si")
              and np.all(np.isfinite(pt_ci.class_energies[n]))]
    assert len(served) == 6
    for name in served:
        assert np.max(np.abs(pt_ci.class_energies[name]
                             - pt_net.class_energies[name])) < 5e-6, name
    ci_six = sum(pt_ci.class_energies[n] for n in served)
    assert np.max(np.abs(pt_net.e2 - ci_six)) < 5e-6


# --- shape 3: DMRG-CASSCF + pseudospin export ------------------------------------------------

def test_dmrg_casscf_matches_ci(cas, cas_dmrg):
    assert cas_dmrg.converged
    assert np.max(np.abs(cas.energies - cas_dmrg.energies)) < E_TOL
    assert cas_dmrg.solver.last.max_bond_dim <= 16


@pytest.fixture(scope="module")
def cas_dmrg_term(ref):
    # the whole ^2P term on the network route, for the analysis-layer parity tests: the
    # term-complete average is what makes the g evidence (and hence the labels) exact.
    # Three modes per node, deliberately: this six-root average spans the ENTIRE
    # CAS(1, 6) space, and on any finer bipartition some two-site window cannot hold all
    # six roots — which the solver rightly refuses rather than truncating the ensemble.
    from kuiva.dmrg import NetworkGraph
    graph = NetworkGraph(2, [(0, 1)], contents=[(0, 1, 2), (3, 4, 5)])
    return CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=6,
                  solver="dmrg", solver_options=dict(max_bond=16), graph=graph,
                  report=False).run()


def test_dmrg_spin_analysis_matches_ci(cas_term, cas_dmrg_term):
    """``<S^2>`` per block, network route against CI route — one implementation of the
    out-of-space correction, two implementations of the in-space square."""
    ci = cas_term.spin_analysis()
    net = cas_dmrg_term.spin_analysis()
    assert [n for _, n in net.blocks] == [n for _, n in ci.blocks] == [2, 4]
    # one electron: <S^2> = 3/4 exactly, with SOC on, on both routes
    assert np.allclose(net.block_s_squared, 0.75, atol=1e-6)
    assert np.max(np.abs(net.block_s_squared - ci.block_s_squared)) < 1e-5
    assert net.has_soc
    assert 0.0 < net.leakage < 1e-2                      # orbitals, not the CI method


def test_dmrg_assignment_offers_the_lande_labels(cas_dmrg_term):
    """The same three-way inference as the CI route (dimension, <S^2>, inverted Lande g),
    with every piece of evidence contracted through the network."""
    assignment = cas_dmrg_term.assign(report=False)
    assert assignment.has_soc
    assert assignment.labels() == ("^2P_1/2", "^2P_3/2")
    assert [t.size for t in assignment.terms] == [2, 4]
    assert [t.j for t in assignment.terms] == [0.5, 1.5]


def test_dmrg_needs_max_bond(ref):
    with pytest.raises(ValueError, match="max_bond"):
        CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, solver="dmrg")


def test_dmrg_checkpoint_and_restart(ref, cas_dmrg, tmp_path):
    """The interim DMRG checkpoint: the ordinary trajectory file, written and resumed.

    The trajectory file carries the orbitals, RDMs and optimizer state and deliberately
    **no** CI vectors and no state energies — a ``SweepResult`` has neither, and the
    checkpoint layer records nothing rather than storing a different quantity under
    those names. The network state goes to the sibling ``*.network.h5`` file, rolling,
    and the restart picks both up: the trajectory exactly, the network as a warm start.
    """
    from kuiva.dmrg.checkpoint import network_state_path, read_network_state
    from kuiva.io.checkpoint import read_checkpoint

    path = tmp_path / "b_dmrg.h5"
    stopped = CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=2,
                     solver="dmrg", solver_options=dict(max_bond=16), max_iter=2,
                     checkpoint=path, checkpoint_options=dict(min_interval=0.0),
                     report=False).run()
    assert path.exists()
    assert stopped.checkpoint_path == str(path)
    chk = read_checkpoint(path)
    assert chk.ci_vectors is None
    assert chk.state_energies.size == 0
    assert chk.space_key is not None and chk.space_key.startswith("dmrg:")
    network = network_state_path(path)
    assert stopped.network_checkpoint_path == str(network)
    assert network.is_file()
    _, meta = read_network_state(network)
    assert meta["space_key"] == chk.space_key
    # the restart takes its active space from the file (giving none is the point)
    resumed = CASSCF(ref, restart=path, n_states=2, solver="dmrg",
                     solver_options=dict(max_bond=16), max_iter=60, report=False).run()
    assert resumed.converged
    assert abs(resumed.energy - cas_dmrg.energy) < E_TOL


def test_dmrg_restart_with_a_different_state_average_is_refused(ref, tmp_path):
    path = tmp_path / "b_dmrg_sa.h5"
    CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=2,
           solver="dmrg", solver_options=dict(max_bond=16), max_iter=1,
           checkpoint=path, checkpoint_options=dict(min_interval=0.0),
           report=False).run()
    with pytest.raises(ValueError, match="state average"):
        CASSCF(ref, restart=path, n_states=3, solver="dmrg",
               solver_options=dict(max_bond=16), report=False).run()


def test_dmrg_restart_refuses_the_adaptive_driver(ref, tmp_path):
    path = tmp_path / "x.h5"
    path.touch()
    with pytest.raises(ValueError, match="frozen-chart"):
        CASSCF(ref, restart=path, n_states=2, solver="dmrg",
               solver_options=dict(max_bond=16, adaptive=True))
    with pytest.raises(ValueError, match="frozen-chart"):
        CASSCF(ref, restart=path, n_states=2, solver="dmrg",
               solver_options=dict(max_bond=16, bond_steps=[8, 16]))


def test_graph_needs_the_dmrg_solver(ref):
    with pytest.raises(ValueError, match="solver='dmrg'"):
        CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1,
               graph="mutual-information")


def test_property_dump_refuses_a_network_reference(cas_dmrg, tmp_path):
    with pytest.raises(ValueError, match="[Pp]seudospin"):
        PropertyDump(cas_dmrg, tmp_path / "x.props")


def test_pseudospin_export_agrees_with_the_dump(cas, cas_dmrg, tmp_path):
    from kuiva.props.pseudospin import read_pseudospin

    psd = PseudospinExport(cas_dmrg, tmp_path / "b.psd", rule="dimension", dims=2,
                           report=False).run()
    (site_g,) = psd.g_values
    # ⚠ The claim here is that two independent property routes (CI transition densities vs
    # network contraction) reduce the *same states* to the same phase-invariant quantity --
    # not that the quantity is Lande's. These fixtures average the j = 1/2 doublet alone, so
    # their anisotropy is machine-dependent (module docstring); the analytic target is
    # asserted on cas_term, where the ensemble makes it well defined.
    dump = PropertyDump(cas, tmp_path / "b.props", report=False).run()
    (doublet,) = dump.matrices.analyse()
    assert doublet.size == 2 and len(site_g) == 3
    assert max(abs(a - b) for a, b in zip(sorted(site_g), sorted(doublet.g_values))) < 1e-6

    back = read_pseudospin(psd.path)
    assert [tuple(row) for row in back["basis"]] == [(-1,), (1,)]
    assert "hamiltonian" in psd.model.provenance


# --- the CheapCI stage and what CASSCF inherits from it --------------------------------------

@pytest.fixture(scope="module")
def pre(ref):
    # ⚠ The whole term again, and here it buys something specific: what leaves this stage is
    # a set of *orbitals*, and orbitals optimized on a non-invariant two-root average are not
    # spherical. Everything built on them inherits that as a seed the next optimization can
    # amplify — measured as a Kramers splitting of 1e-6 Eh, large enough for the
    # state-averaging gate to refuse a downstream run outright. Averaging the term keeps the
    # orbitals spherical and the chains below reproducible.
    return CheapCI(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=6,
                   report=False).run()


def test_cheap_ci_suggests_the_occupied_doublet(pre):
    # occupation-based selection is a lower bound: it sees the fractionally occupied 2p
    # spinors and cannot see an empty orbital a better treatment would populate
    suggested = pre.suggested_active()
    assert 2 <= suggested.size <= 6
    assert pre.mutual_information.shape == (6, 6)


def test_casscf_inherits_space_and_orbitals_from_cheap_ci(pre, cas_term):
    # ⚠ 1e-5 Eh, not 1e-8: two SA-CASSCFs from different starting orbitals may converge to
    # slightly different stationary points (measured 1.2e-6 Eh here). The claims are that
    # the inherited run is the same calculation to far below any physical band, and that the
    # repaired preopt orbitals keep the Kramers doublet exact — the mechanism this chain
    # depends on (CheapCI restores pairing the truncated cheap CI is entitled to break).
    #
    # ⚠ Over the whole term, deliberately: starting orbitals that are not spherical seed a
    # pairing defect, and on a two-root (non-invariant) average the optimizer amplifies it
    # instead of damping it — measured splitting up to 5e-8 Eh, appearing in roughly one run
    # in eight. That is the mechanism this chain is here to exercise, not a tolerance to widen.
    inherited = CASSCF(pre, n_states=6, max_iter=150, report=False).run()
    assert inherited.converged
    assert inherited.active.description == pre.space.description
    assert abs(inherited.energies[1] - inherited.energies[0]) < KRAMERS_TOL
    assert np.max(np.abs(inherited.energies - cas_term.energies)) < 1e-5


def test_dmrg_casscf_on_the_entanglement_topology(pre, cas):
    # ⚠ Two roots, and here that is forced rather than chosen: a **one-electron** active space
    # cannot carry a six-root ensemble on a path network at any size, because the two-site
    # space on the edge bond is three-dimensional whatever the bond cap — one electron leaves
    # the one-spinor block with two states. The sweep refuses rather than truncate an ensemble
    # it cannot represent, which is right. What makes the two-root average safe here is that
    # `pre` averages the whole term (see its fixture), so the orbitals it hands over are still
    # spherical and there is no pairing defect for the non-invariant ensemble to amplify.
    seeded = CASSCF(pre, n_states=2, solver="dmrg", graph="mutual-information",
                    max_iter=150, solver_options=dict(max_bond=16), report=False).run()
    assert seeded.converged
    assert seeded.graph is not None
    # ⚠ The *physical* bar, not KRAMERS_TOL, and the difference is the solver rather than the
    # ensemble: 1e-15..1e-13 Eh is what the general CI path delivers, while a sweep converges
    # its two roots separately and leaves ~1e-9 Eh between them here — a statement about
    # iterative convergence, three orders below anything the state-averaging gate reacts to
    # and four below anything physical. Asserting the CI path's number here would be
    # asserting the wrong claim, and it fails about one full-suite run in ten.
    assert abs(seeded.energies[1] - seeded.energies[0]) < LEVEL_TOL
    assert np.max(np.abs(seeded.energies - cas.energies)) < 1e-5    # see the note above


# --- checkpoint / restart through the class layer --------------------------------------------

def test_restart_continues_the_calculation(ref, cas, tmp_path):
    path = tmp_path / "b_casscf.h5"
    stopped = CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=2,
                     max_iter=3, checkpoint=path,
                     checkpoint_options=dict(min_interval=0.0), report=False).run()
    assert not stopped.converged and path.exists()
    # the restart takes its active space from the file (giving none is the point), and
    # max_iter counts total macro-iterations across the interruption
    resumed = CASSCF(ref, restart=path, n_states=2, max_iter=60, report=False).run()
    assert resumed.converged
    assert abs(resumed.energy - cas.energy) < E_TOL


def test_restart_needs_an_existing_file(ref, tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        CASSCF(ref, restart=tmp_path / "nope.h5", n_states=2)


# --- multi-centre character selection ---------------------------------------------------------

def test_fragment_union_active_space(ref):
    from kuiva.interface.api import active_space_for

    space = active_space_for(ref.reference,
                             character=[("B", "s", 2), ("B", "p", 6)])
    assert space.spaces.n_active == 8
    assert space.fragments is not None
    assert tuple(len(f) for f in space.fragments) == (2, 6)
    assert "+" in space.description
    # 1s pair + 2p shell active on B leaves the 2s pair inactive: CAS(3, 8)
    assert space.n_elec == 3


def test_overlapping_fragments_are_refused(ref):
    from kuiva.interface.api import active_space_for

    with pytest.raises(ValueError, match="both claim"):
        active_space_for(ref.reference, character=[("B", "p", 6), ("B", "p", 6)])


def test_fragment_counts_must_divide(ref):
    from kuiva.interface.api import active_space_for

    with pytest.raises(ValueError, match="Kramers pairs"):
        active_space_for(ref.reference, character=[("B", "s"), ("B", "p")], n_active=6)


def test_top_level_exports():
    assert kuiva.CASSCF is CASSCF and kuiva.Reference is Reference
    assert set(kuiva.__all__) >= {"Molecule", "ScalarSCF", "Reference", "CheapCI",
                                  "CASSCF", "NEVPT2", "PropertyDump", "PseudospinExport"}


# --- the public surface: what a bare `import kuiva` reaches ---------------------------------

def test_the_top_level_namespace_pairs_every_writer_with_a_reader():
    """⚠ The namespace is deliberately thin, and the readers are in it on one argument: each
    is the *read* counterpart of something the same namespace writes.

    Reading a stored product back is not exotic — it is how two calculations are compared at
    all, because the phases in those files are arbitrary and only the phase-invariant
    reduction compares them soundly. Needing a module path for that made the documented
    comparison start with an import nobody guesses.
    """
    for writer, reader in [("PropertyDump", "read_dump"),
                           ("PseudospinExport", "read_pseudospin"),
                           ("PropertyDump", "PropertyMatrices"),
                           ("PseudospinExport", "PseudospinModel")]:
        assert hasattr(kuiva, writer) and hasattr(kuiva, reader)
    assert hasattr(kuiva, "read_checkpoint")           # CASSCF(checkpoint=) writes these

    # The names resolve to the real objects, not to placeholders.
    assert kuiva.PropertyMatrices is __import__(
        "kuiva.props.dump", fromlist=["x"]).PropertyMatrices
    assert kuiva.read_checkpoint is __import__(
        "kuiva.io.checkpoint", fromlist=["x"]).read_checkpoint
    # Every advertised name is reachable, and `__all__` advertises exactly the hook's keys.
    # (`dir(kuiva)` is not the comparison: importing any submodule anywhere in the process
    # binds it as an attribute here, so it grows with whatever else the suite has run.)
    assert set(kuiva.__all__) == set(kuiva._TOP_LEVEL) | {"__version__"}
    for name in kuiva._TOP_LEVEL:
        assert getattr(kuiva, name) is not None
    with pytest.raises(AttributeError):
        kuiva.no_such_name


def test_importing_kuiva_stays_side_effect_free():
    """⚠ The reason the top level is a PEP 562 hook and not a pile of imports. Adding five
    names to it must not drag PySCF, h5py or the integral machinery into every process that
    imports this package for something that never touches the front end."""
    import subprocess
    import sys

    probe = ("import sys, kuiva; "
             "print(int('pyscf' in sys.modules), int('h5py' in sys.modules), "
             "len([n for n in dir(kuiva) if not n.startswith('_')]))")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         check=True).stdout.split()
    assert out[0] == "0" and out[1] == "0"             # neither was imported
    assert int(out[2]) >= 13                           # ...and every name is still there


def test_a_molecule_can_be_read_from_an_xyz_file(tmp_path):
    """The two-line XMol header is the whole difference from ``from_xyz_string``, and the
    reason this exists: pointing the string form at a real file fails on the count line."""
    path = tmp_path / "water.xyz"
    path.write_text("3\nwater, B3LYP/6-31G*\n"
                    "O   0.000000  0.000000  0.117300\n"
                    "H   0.000000  0.757200 -0.469200\n"
                    "H   0.000000 -0.757200 -0.469200\n")

    mol = kuiva.Molecule.from_xyz_file(path, basis="x2c-SVPall-2c")
    assert [s for s, _ in mol.atoms] == ["O", "H", "H"]
    assert mol.atoms[1][1] == pytest.approx((0.0, 0.7572, -0.4692))
    assert mol.unit == "Angstrom"

    # A headerless file is accepted too -- some tools emit them. What is refused is a
    # *disagreement*, never the absence of a header.
    bare = tmp_path / "bare.xyz"
    bare.write_text("\n".join(path.read_text().splitlines()[2:]))
    assert kuiva.Molecule.from_xyz_file(bare, basis="x2c-SVPall-2c").atoms == mol.atoms


def test_a_miscounted_xyz_file_is_refused_not_truncated(tmp_path):
    """⚠ The count is checked, not trusted. A truncated or concatenated .xyz read as its
    first n atoms is a different molecule, and nothing downstream would say so."""
    path = tmp_path / "short.xyz"
    path.write_text("5\ntruncated\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24\n")
    with pytest.raises(ValueError, match="declares 5 atoms and the file carries 3"):
        kuiva.Molecule.from_xyz_file(path, basis="x2c-SVPall-2c")


def test_the_string_form_says_what_to_use_when_handed_a_file(tmp_path):
    """A confusing way to learn about a format is not learning about it."""
    with pytest.raises(ValueError, match="from_xyz_file"):
        kuiva.Molecule.from_xyz_string("3\nwater\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24",
                                       basis="x2c-SVPall-2c")
