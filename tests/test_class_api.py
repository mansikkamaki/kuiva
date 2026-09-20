"""The class API (kuiva/interface/stages.py): the uniform stage contract, end to end.

Covers the three calculation shapes the layer was designed on (the class-API design scope),
on the cheapest system that exhibits each structure:

* conventional-CI CASSCF + property dump — B (2p^1), whose ground j = 1/2 doublet has the
  **analytic** Lande factor g = 1 - (g_e - 1)/3 = 0.66589, a target no convention of any
  code can move;
* CASSCF + SC-NEVPT2 + the corrected (hybrid-protocol) dump on the same reference;
* DMRG-CASSCF + the pseudospin export, whose g values **and hyperfine field** must agree
  with the dump's phase-invariant reductions — two independent property routes to one set of
  invariants. The hyperfine half is the sharper half of that claim: it travels through the
  TTNO on one route and through CI transition densities on the other, and the ``|A|`` values
  and the mixed ``Tr_b(mu.T)`` invariant are the only comparable quantities.

Tolerances: state degeneracies inside a Kramers doublet are asserted at 1e-10 Eh (the
general path's measured splitting is 1e-15..1e-13 Eh); CI-vs-DMRG energies at 1e-8 Eh
; g against Lande at 2e-3 on a term-complete average (see below);
dump-vs-pseudospin g at 1e-6 and |A| at 1e-6 relative (same states, two contractions;
measured 2e-12).

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
import kuiva.util.deadline
from kuiva.interface.stages import (CASCI, CASSCF, CheapCI, NEVPT2, PropertyDump,
                                    PseudospinExport, Reference, ScalarSCF)
from kuiva.props.multiplet import block_collinearity

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
    #
    # hyperfine={"B": 11}: the two property routes have to agree on the hyperfine field as
    # well as on g, and this is the one fixture both of them are built from. It costs one
    # picture-changed AO operator on a 14-function basis -- no atomic solve is involved --
    # and it makes every file written below carry a [NUCLEI] table, which is the point:
    # naming the nuclei at ingestion IS the request, on either route.
    return ScalarSCF(boron(), memory_gb=4.0, screening="none",
                     hyperfine={"B": 11}).run()


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


# --- materializing finished stages from their checkpoints ---------------------------------
#
# ⚠ These are the end-to-end statement that a file is enough to *finish* a calculation rather
# than only to resume one: the dump and the correction below are computed from stored
# orbitals, and the run that produced them is gone.

def test_a_finished_casscf_is_materialized_from_its_checkpoint(ref, cas_term, tmp_path):
    """`from_checkpoint` reproduces the stage the file was written by, with no optimization.

    ⚠ The energies are compared **bitwise on the stored total** and to KRAMERS_TOL on the
    re-solved spectrum: the orbitals come back exactly, and the states are re-solved at them
    from the stored CI vectors, which starts Davidson at the answer.
    """
    path = tmp_path / "b_term.h5"
    written = CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=6,
                     checkpoint=path, checkpoint_options=dict(min_interval=0.0),
                     report=False).run()
    assert written.converged

    back = CASSCF.from_checkpoint(path, ref, report=False).run()

    assert back.converged and back.ran
    assert back.energy == written.energy               # the file's number, not a re-derived one
    assert back.n_states == written.n_states           # defaulted from the state-average record
    np.testing.assert_allclose(back.energies, written.energies, atol=KRAMERS_TOL)
    np.testing.assert_array_equal(back.coeff, written.coeff)
    # ⚠ The trajectory diagnostic belongs to the run that wrote the file and is not invented.
    assert back.boundary_initial is None
    assert back.boundary is not None


def test_a_materialized_casscf_feeds_the_property_dump(ref, tmp_path):
    """The gap this closes: a converged run whose dump was never requested. The g value here
    comes from orbitals that exist only in a file."""
    path = tmp_path / "b_props.h5"
    CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=6,
           checkpoint=path, checkpoint_options=dict(min_interval=0.0), report=False).run()

    back = CASSCF.from_checkpoint(path, ref, report=False).run()
    dump = PropertyDump(back, tmp_path / "props.out", report=False).run()

    levels = dump.matrices.analyse()
    assert [level.size for level in levels] == [2, 4]
    assert all(abs(g - G_LANDE[level.size]) < 2e-3
               for level in levels for g in level.g_values)


def test_materializing_an_unconverged_checkpoint_is_refused(ref, tmp_path):
    """⚠ Its orbitals are an iterate, not a result, and every number built on them inherits
    that. The override exists and says so."""
    path = tmp_path / "b_stopped.h5"
    CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=6, max_iter=1,
           checkpoint=path, checkpoint_options=dict(min_interval=0.0), report=False).run()

    with pytest.raises(ValueError, match="had NOT converged"):
        CASSCF.from_checkpoint(path, ref, report=False).run()
    CASSCF.from_checkpoint(path, ref, require_converged=False, report=False).run()


def test_materializing_against_a_different_system_is_refused(scf, ref, tmp_path):
    """⚠ The check that did not exist: an orthonormal set of the right dimension optimizes to
    a plausible number whatever molecule it came from."""
    path = tmp_path / "b_system.h5"
    CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=6,
           checkpoint=path, checkpoint_options=dict(min_interval=0.0), report=False).run()

    other = Reference(ScalarSCF(kuiva.Molecule([("C", (0.0, 0.0, 0.0))],
                                               basis="x2c-SVPall-2c", spin=2),
                                memory_gb=4.0, screening="none").run()).run()
    with pytest.raises(ValueError, match="different system"):
        CASSCF.from_checkpoint(path, other, report=False).run()


def test_a_finished_nevpt2_is_materialized_from_its_checkpoint(ref, cas_term, tmp_path):
    """The correction's own file, assembled back into a stage that a dump accepts.

    ⚠ Exact equality throughout: nothing is recomputed, so anything but a bitwise match would
    mean the table is not what the run produced.
    """
    path = tmp_path / "b_e2.h5"
    written = NEVPT2(cas_term, checkpoint=path,
                     checkpoint_options=dict(min_interval=0.0)).run()

    back = NEVPT2.from_checkpoint(path, cas_term, report=False).run()

    np.testing.assert_array_equal(written.e2, back.e2)
    np.testing.assert_array_equal(written.total_energies, back.total_energies)
    assert back.result.complete == written.result.complete
    assert back.result.n_frozen == written.result.n_frozen

    dump = PropertyDump(back, tmp_path / "corrected.out", report=False).run()
    assert "SC-NEVPT2" in (tmp_path / "corrected.out").read_text()
    assert dump.matrices.n_states == written.e2.size


def test_a_nevpt2_restart_finishes_an_interrupted_table(ref, cas_term, tmp_path):
    """End to end at the stage level: a table missing its later states, and a restart that
    reaches the same numbers. ⚠ The interruption is simulated by removing whole states, which
    is what a real one leaves — the driver finishes a state before it claims one.

    (The stop cause itself is driven at the function level, in ``test_nevpt2_checkpoint``,
    where a stub can stand in for a signal that a stage's eager ``signals=`` validation
    rightly refuses.)
    """
    from kuiva.pt.checkpoint import read_nevpt2_checkpoint, write_nevpt2_checkpoint

    path = tmp_path / "b_e2_stop.h5"
    whole = NEVPT2(cas_term, checkpoint=path,
                   checkpoint_options=dict(min_interval=0.0)).run()

    stored = read_nevpt2_checkpoint(path)
    assert stored.complete
    stored.entries = {key: value for key, value in stored.entries.items() if key[0] == 0}
    stored.e_casscf[1:] = np.nan
    write_nevpt2_checkpoint(path, stored)
    assert not read_nevpt2_checkpoint(path).complete

    resumed = NEVPT2(cas_term, restart=path, checkpoint=path,
                     checkpoint_options=dict(min_interval=0.0)).run()
    np.testing.assert_array_equal(whole.e2, resumed.e2)
    assert read_nevpt2_checkpoint(path).complete


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


def test_a_windowed_dmrg_restart_takes_its_count_from_the_file(ref, tmp_path):
    """⚠ Same rule as on the conventional-CI route, and it has to be: the count comes from
    the **file** and the window is what is compared, because the interrupted calculation ran
    at one count and re-resolving here could pick another. A changed cutoff is a different
    calculation and is refused."""
    from kuiva.dmrg import NetworkGraph
    from kuiva.io.checkpoint import (STATE_AVERAGE_KEY, read_checkpoint,
                                     state_average_window)

    graph = NetworkGraph(2, [(0, 1)], contents=[(0, 1, 2), (3, 4, 5)])
    window = kuiva.EnergyWindow(5.0, manifold_gap=1.0)
    options = dict(solver="dmrg", solver_options=dict(max_bond=16), graph=graph,
                   report=False)
    path = tmp_path / "b_dmrg_window.h5"
    stopped = CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1,
                     n_states=window, max_iter=2, checkpoint=path,
                     checkpoint_options=dict(min_interval=0.0), **options).run()
    assert not stopped.converged and stopped.n_states == 2
    stored = read_checkpoint(path)
    assert state_average_window(stored.metadata[STATE_AVERAGE_KEY]) == window

    resumed = CASSCF(ref, restart=path, n_states=window, max_iter=60, **options).run()
    assert resumed.converged and resumed.n_states == 2
    with pytest.raises(ValueError, match="different calculation"):
        CASSCF(ref, restart=path, n_states=kuiva.EnergyWindow(900.0), **options).run()


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

    # ⚠ **And the same two routes on the hyperfine field**, which is the part of this claim
    # that no g value can carry: the network route contracts T onto the model space through
    # the TTNO, the CI route contracts it against transition densities, and the only
    # comparable quantity is the phase-invariant reduction. |A| is quadratic and isotope
    # dependent, so it is taken at the isotope the [NUCLEI] table names -- the same table in
    # both files, written by the same code.
    assert psd.model.hyperfine_labels == ("B1",) == dump.matrices.hyperfine_labels
    assert psd.model.nucleus("B1")["label"] == dump.matrices.nucleus("B1")["label"] == "11B"
    # 2 electronic states x (2I+1 = 4) for 11B; reported, never refused -- Kuiva does not
    # form the electron-nuclear product space.
    assert psd.model.product_dim() == 8
    assert back["header"]["product_dim"] == "8"
    for k, axis in enumerate("xyz"):
        assert np.array_equal(back["matrices"]["T_B1_" + axis],
                              psd.model.hyperfine["B1"][k])

    g_n = float(psd.model.nucleus("B1")["g"])
    net = _principal_a(psd.model.analyse()[0], g_n)
    ci = _principal_a(doublet, g_n)
    assert min(ci) > 100.0                      # MHz: a real coupling, not a rounding artefact
    # the same band the g comparison above carries, and for the same reason: two contractions
    # of one set of states. Measured agreement is 2e-12 relative, five orders inside it.
    assert max(abs(a - b) for a, b in zip(net, ci)) < 1e-6 * max(ci)
    # The *mixed* invariant, which the magnitudes above cannot carry: it holds the relative
    # orientation and sign of T against mu, and it is what a permuted Cartesian component, an
    # unrotated frame or the conjugation trap breaks while every |A| stays right. ⚠ Compared
    # between the two routes and **not** against the analytic +-1 of Wigner-Eckart: these
    # fixtures average the j = 1/2 doublet alone, which is not the ensemble the term's symmetry
    # leaves invariant (module docstring), so +-1 is off by ~3e-7 here for the same reason the
    # g values are anisotropic. The analytic claim is asserted on a term-complete average in
    # tests/test_hyperfine_states.py, where it means something.
    net_coll = block_collinearity(psd.model.mu_in_eigenbasis(),
                                  psd.model.hyperfine_in_eigenbasis()["B1"], 0, 2)
    ci_coll = block_collinearity(_energy_sorted(dump.matrices, dump.matrices.mu),
                                 _energy_sorted(dump.matrices, dump.matrices.hyperfine["B1"]),
                                 doublet.start, doublet.size)
    assert abs(net_coll - ci_coll) < 1e-9       # measured 1e-14
    assert abs(abs(net_coll) - 1.0) < 1e-5


def _principal_a(block, g_nuclear):
    """The ``|A|`` principal values [MHz] of one degenerate block — the phase-invariant
    reduction, which is the only comparable hyperfine quantity between two routes."""
    from kuiva.props.multiplet import multiplet_hyperfine_values
    return multiplet_hyperfine_values(block.hyperfine["B1"], block.size, g_nuclear)


def _energy_sorted(matrices, operator):
    """An operator re-indexed into the energy-ordered basis the multiplets are indexed in."""
    order = np.argsort(np.asarray(matrices.energies, dtype=float))
    return np.asarray(operator)[:, order, :][:, :, order]


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


# --- the CASCI stage: a spectrum at fixed orbitals --------------------------------------------

def test_casci_reproduces_the_casscf_spectrum_at_its_own_orbitals(cas_term):
    """The identity that says the stage runs the calculation it claims to: the CASSCF's own
    last CI is a CASCI at its converged orbitals over its own state average, so asking for
    that again must return the same numbers — inheriting both the orbitals and the space,
    neither of which is restated here."""
    ci = CASCI(cas_term, n_states=6, report=False).run()
    assert ci.active is cas_term.active
    assert np.array_equal(ci.coeff, cas_term.coeff)
    assert np.max(np.abs(ci.energies - cas_term.energies)) < KRAMERS_TOL
    assert abs(ci.energy - cas_term.energy) < KRAMERS_TOL
    assert ci.solver_kind == "ci"


def test_casci_at_the_guess_orbitals_is_above_the_casscf(ref, cas):
    """The variational statement, and the reason the stage exists: the same functional over
    the same two states is *higher* at the reference's guess orbitals than at the optimized
    ones. Both numbers are state-averaged energies of one active space, so their order is a
    theorem rather than a measurement."""
    guess = CASCI(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=2,
                  report=False).run()
    assert guess.active.description == cas.active.description
    assert guess.energy > cas.energy
    assert guess.energy - cas.energy < 1e-2          # a guess, not a different calculation


def test_casci_inherits_from_a_cheap_ci(pre):
    """The third upstream: the pre-optimized orbitals and the space stated on that stage."""
    ci = CASCI(pre, n_states=6, report=False).run()
    assert ci.active is pre.space
    assert np.array_equal(ci.coeff, pre.orbitals)
    assert abs(ci.energies[1] - ci.energies[0]) < KRAMERS_TOL


def test_casci_takes_an_energy_window_and_resolves_it_to_the_j_manifold(cas_term):
    """The third form of ``n_states``: on boron's 2p^1 term the spin-orbit spectrum is a
    j = 1/2 doublet and a j = 3/2 quartet about 15 cm^-1 above it. A window that cuts between
    the two resolves to 2, and one that reaches past the quartet resolves to 6 — the whole
    space, so no witness is needed. Before ``run()`` the stage's count is ``None``; after it,
    it is the resolved count, which is what every downstream stage reads."""
    doublet = CASCI(cas_term, n_states=kuiva.EnergyWindow(5.0, manifold_gap=1.0),
                    report=False)
    assert doublet.n_states is None and doublet.window is None
    assert doublet.window_request == kuiva.EnergyWindow(5.0, manifold_gap=1.0)
    doublet.run()
    assert doublet.n_states == 2 and doublet.energies.size == 2
    assert doublet.window.complete and doublet.window.count == 2
    assert doublet.window.boundary_gap_cm > 5.0
    fixed = CASCI(cas_term, n_states=2, report=False).run()
    assert np.max(np.abs(doublet.energies - fixed.energies)) < KRAMERS_TOL
    assert "state window" in doublet.summary()

    term = CASCI(cas_term, n_states=kuiva.EnergyWindow(1000), report=False).run()
    assert term.n_states == 6 and term.window.verdict.spans_space
    assert np.max(np.abs(term.energies - cas_term.energies)) < KRAMERS_TOL


def test_a_window_refuses_weights_and_the_per_irrep_form_on_the_network(ref, cas_term):
    with pytest.raises(ValueError, match="weights= cannot be combined"):
        CASCI(cas_term, n_states=kuiva.EnergyWindow(1000), weights=[0.5, 0.5])
    with pytest.raises(ValueError, match="weights= cannot be combined"):
        CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1,
               n_states=kuiva.EnergyWindow(1000), weights=[0.5, 0.5])
    with pytest.raises(ValueError, match="weights= cannot be combined"):
        CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1,
               n_states=kuiva.EnergyWindow(1000), weights=[0.5, 0.5], solver="dmrg",
               solver_options=dict(max_bond=16))
    # a per-irrep window selects states per determinant sector, which a network solve —
    # targeting one sector — has nothing to do with
    with pytest.raises(ValueError, match="per-irrep energy window"):
        CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1,
               n_states={"1/2g": kuiva.EnergyWindow(1000)}, solver="dmrg",
               solver_options=dict(max_bond=16))


def test_a_windowed_casscf_is_the_fixed_count_run_it_resolves_to(ref, cas):
    """⚠ The statement the whole round design rests on: the count is resolved at fixed
    orbitals and **held for the whole optimization**, so the optimizer runs unchanged at a
    fixed count. On boron's 2p^1 the spin-orbit spectrum is a j = 1/2 doublet with the
    j = 3/2 quartet ~15 cm^-1 above, so a 5 cm^-1 window resolves to 2 at both ends of the
    trajectory — one round — and reproduces the ``n_states=2`` run **bitwise**: same solver,
    same warm start, same steps."""
    window = kuiva.EnergyWindow(5.0, manifold_gap=1.0)
    stage = CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=window,
                   report=False)
    assert stage.n_states is None and stage.window is None
    assert stage.window_request == window
    stage.run()
    assert stage.n_states == 2 and stage.energies.size == 2
    assert len(stage.rounds) == 1 and stage.window.count == 2
    assert stage.window.complete and not stage.window.ambiguous
    assert stage.energy == cas.energy                       # bitwise, not to a tolerance
    assert np.array_equal(stage.coeff, cas.coeff)
    # the boundary reports are the window's verdicts rather than a second Davidson solve
    assert stage.boundary.n_states == 2 and stage.boundary.gap_cm > 5.0
    assert stage.boundary_initial is not None
    assert stage.boundary_initial.where == "starting orbitals"
    # ... and every consumer of the count reads a number
    assert CASCI(stage, n_states=2, report=False).run().energies.size == 2


def test_a_windowed_casscf_reaching_past_the_term_averages_the_whole_space(ref, cas_term):
    """⚠ At the reference's *guess* orbitals boron's 2p shell is split by ~5300 cm^-1 — the
    ROHF solution is not spherical — and the CASSCF brings that down to the ~33 cm^-1 that is
    the actual spin-orbit splitting. A cutoff above the guess splitting therefore takes all
    six roots of the 2p determinant space at both ends: the average spans the space, so there
    is no witness root, and that is a **pass** rather than a gap of zero."""
    stage = CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1,
                   n_states=kuiva.EnergyWindow(10000), report=False).run()
    assert stage.n_states == 6 and stage.window.verdict.spans_space
    assert stage.boundary.spans_full_ci and stage.boundary.is_clean
    assert abs(stage.energy - cas_term.energy) < E_TOL


def test_a_count_that_changes_between_rounds_lands_on_the_fixed_count_run(ref):
    """⚠ The other branch of the round loop, end to end. On CAS(3, 8) the 4000 cm^-1 window
    resolves to **2** at the guess orbitals (the 2s-2p gap is ~5290 cm^-1 there), the state
    average over that doublet pulls the next pair down to ~2920 cm^-1, and the re-resolution
    reads **6** — so a second round runs at 6 and settles. What it converges to is the
    ``n_states=6`` CASSCF, to the driver's own tolerance: a window resolves to a count, and
    after that it is that calculation."""
    stage = CASSCF(ref, character=[("B", "s", 2), ("B", "p", 6)],
                   n_states=kuiva.EnergyWindow(4000.0), max_iter=60, report=False).run()
    assert [(r.n_states, r.verdict_count) for r in stage.rounds] == [(2, 6), (6, 6)]
    assert stage.n_states == 6 and stage.converged and not stage.window.ambiguous
    fixed = CASSCF(ref, character=[("B", "s", 2), ("B", "p", 6)], n_states=6, max_iter=60,
                   report=False).run()
    assert abs(stage.energy - fixed.energy) < 1e-8
    # the individual states agree to the *driver's* tolerance, not the suite's: two converged
    # runs of one functional stop wherever |g| first falls under conv_grad, which is not the
    # same point to the last digit
    assert np.max(np.abs(stage.energies - fixed.energies)) < 1e-6
    # ⚠ max_iter is the budget ACROSS rounds: the second round continues the count
    assert sum(r.n_iterations for r in stage.rounds) == stage.orbital.n_iterations


def test_a_windowed_restart_takes_its_count_from_the_file(ref, tmp_path):
    """⚠ A restart continues the calculation that was interrupted, and that calculation ran
    at one count: the count comes from the file and the **window** is what is compared. A
    different cutoff is a different calculation and is refused, exactly as a different count
    is."""
    window = kuiva.EnergyWindow(5.0, manifold_gap=1.0)
    path = tmp_path / "b_window.h5"
    stopped = CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=window,
                     max_iter=3, checkpoint=path, checkpoint_options=dict(min_interval=0.0),
                     report=False).run()
    assert not stopped.converged and path.exists() and stopped.n_states == 2

    from kuiva.io.checkpoint import (STATE_AVERAGE_KEY, read_checkpoint,
                                     state_average_window)
    stored = read_checkpoint(path)
    assert state_average_window(stored.metadata[STATE_AVERAGE_KEY]) == window

    resumed = CASSCF(ref, restart=path, n_states=window, max_iter=60, report=False).run()
    assert resumed.converged and resumed.n_states == 2
    # the resolution at the starting orbitals belongs to the run that wrote the file
    assert resumed.boundary_initial is None and resumed.boundary is not None

    with pytest.raises(ValueError, match="different calculation"):
        CASSCF(ref, restart=path, n_states=kuiva.EnergyWindow(900.0), report=False).run()
    with pytest.raises(ValueError, match="records no window"):
        CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=2,
               max_iter=2, checkpoint=tmp_path / "plain.h5",
               checkpoint_options=dict(min_interval=0.0), report=False).run() and None
        CASSCF(ref, restart=tmp_path / "plain.h5", n_states=window, report=False).run()


def test_a_windowed_checkpoint_materializes_at_its_recorded_count(ref, tmp_path):
    """``from_checkpoint`` configures nothing, so the count comes from the file and the
    window rides along as provenance — there is no trajectory left for a round to move it."""
    path = tmp_path / "b_window_done.h5"
    window = kuiva.EnergyWindow(5.0, manifold_gap=1.0)
    done = CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=window,
                  max_iter=60, checkpoint=path, report=False).run()
    assert done.converged
    back = CASSCF.from_checkpoint(path, ref, report=False).run()
    assert back.n_states == done.n_states == 2
    assert back.solver.window == window                     # provenance, not a request
    assert abs(back.energy - done.energy) < E_TOL
    assert back.boundary_initial is None


def test_a_cheap_ci_window_hands_the_casscf_its_first_rung(ref):
    """⚠ A rung, never a verdict. The cheap CI's spectrum is qualitative, so what the
    handoff carries is an estimate of how many roots lie inside the cutoff; the rule then
    runs on the full CI's own spectrum, and the output names where the rung came from."""
    window = kuiva.EnergyWindow(10000)
    pre = CheapCI(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=window,
                  report=False).run()
    assert pre.n_states == 6 and pre.window is not None
    assert pre.spectrum_cm.size == 6 and pre.spectrum_cm[0] == 0.0

    stage = CASSCF(pre, n_states=window, report=False)
    assert stage._window_first_rung() == 6
    stage.run()
    assert stage.n_states == 6
    # a Reference upstream has no spectrum to hand over, and the ladder says so
    assert CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1,
                  n_states=window, report=False)._window_first_rung() is None


def test_a_windowed_dmrg_casscf_is_the_fixed_count_run_it_resolves_to(ref):
    """The round design on the network route: the count is resolved at fixed orbitals by a
    ladder whose every rung is a sweep campaign, then **held** for a whole orbital
    optimization — so a window that resolves to 2 and stays there reproduces the
    ``n_states=2`` DMRG-CASSCF bitwise, the same statement the conventional-CI route carries.

    ⚠ Three modes per node, as in ``cas_dmrg_term`` and for a sharper version of the same
    reason: a window is resolved against a **converged witness root above the count**, so
    the tour's narrowest two-site window has to hold the count *and* a witness pair. On the
    default one-mode-per-node path of a one-electron active space it holds three roots
    whatever the bond dimension is, which the refusal below is about."""
    from kuiva.dmrg import NetworkGraph

    graph = NetworkGraph(2, [(0, 1)], contents=[(0, 1, 2), (3, 4, 5)])
    window = kuiva.EnergyWindow(5.0, manifold_gap=1.0)
    options = dict(character=("B", "p"), n_active=6, n_active_elec=1, solver="dmrg",
                   solver_options=dict(max_bond=16), graph=graph, report=False)
    stage = CASSCF(ref, n_states=window, **options)
    assert stage.n_states is None and stage.window is None
    stage.run()
    assert stage.n_states == 2 and stage.energies.size == 2
    assert len(stage.rounds) == 1 and stage.window.count == 2
    assert stage.window.complete and not stage.window.ambiguous
    # the witness is converged network roots, not the sweep's local one-sided gap
    assert stage.window.witness == "converged network roots"
    assert stage.boundary_gap_cm == stage.window.boundary_gap_cm > 50.0
    fixed = CASSCF(ref, n_states=2, **options).run()
    # ⚠ Not asserted bitwise, unlike the conventional-CI route's version of this test: the
    # windowed run's first sweep starts from the ladder's own converged ensemble and the fixed
    # one from a random state, and this layer's reductions are measurably order-sensitive. What
    # IS asserted is that the trajectory is the same length and ends in the same place — the
    # statement that the rounds wrapped the driver rather than modifying it.
    assert stage.orbital.n_iterations == fixed.orbital.n_iterations
    assert abs(stage.energy - fixed.energy) < E_TOL
    assert np.max(np.abs(stage.energies - fixed.energies)) < E_TOL
    assert "state window" in stage.summary()


def test_a_network_count_that_changes_between_rounds_lands_on_the_fixed_count_run(ref):
    """The other branch of the round loop, on the network route: on CAS(3, 8) the 4000 cm^-1
    window resolves to **2** at the guess orbitals, that average pulls the next pair down
    inside the cutoff, and the re-resolution reads **6** — so a second round runs at 6, on a
    sibling solver warm-started from the ladder's own ensemble and with the previous round's
    curvature discarded (a new count is a new energy functional). What it converges to is the
    ``n_states=6`` DMRG-CASSCF."""
    from kuiva.dmrg import NetworkGraph

    graph = NetworkGraph(2, [(0, 1)], contents=[(0, 1, 2, 3), (4, 5, 6, 7)])
    options = dict(character=[("B", "s", 2), ("B", "p", 6)], solver="dmrg",
                   solver_options=dict(max_bond=16), graph=graph, max_iter=60,
                   report=False)
    stage = CASSCF(ref, n_states=kuiva.EnergyWindow(4000.0), **options).run()
    assert [(r.n_states, r.verdict_count) for r in stage.rounds] == [(2, 6), (6, 6)]
    assert stage.n_states == 6 and stage.converged and not stage.window.ambiguous
    fixed = CASSCF(ref, n_states=6, **options).run()
    assert abs(stage.energy - fixed.energy) < E_TOL
    # ⚠ max_iter is the budget ACROSS rounds, here as on the conventional-CI route
    assert sum(r.n_iterations for r in stage.rounds) == stage.orbital.n_iterations


def test_a_windowed_dmrg_runs_on_the_event_gated_driver_too(ref):
    """A bond-dimension ladder routes the optimization through the event-gated driver, which
    is a *sibling* of the smooth one and has no ``start_iteration`` — so a window's rounds
    hand it what is left of the budget and shift its counter back, which is what keeps
    ``max_iter`` a budget across rounds on both drivers."""
    from kuiva.dmrg import NetworkGraph

    graph = NetworkGraph(2, [(0, 1)], contents=[(0, 1, 2), (3, 4, 5)])
    stage = CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1,
                   n_states=kuiva.EnergyWindow(5.0, manifold_gap=1.0), solver="dmrg",
                   solver_options=dict(max_bond=16, bond_steps=[8, 16]), graph=graph,
                   max_iter=30, report=False).run()
    assert stage.n_states == 2 and len(stage.rounds) == 1
    assert stage.orbital.n_iterations <= 30
    assert abs(stage.energies[1] - stage.energies[0]) < KRAMERS_TOL


def test_a_windowed_network_that_cannot_hold_its_witness_is_refused(ref):
    """⚠ Refused, never truncated to the roots the network happens to be able to hold: that
    truncation would be a cut the ensemble's dimension chose, which is exactly what a window
    exists to forbid. The message names the capacity, the sector and the fix."""
    with pytest.raises(ValueError, match="coarser node partition"):
        CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1,
               n_states=kuiva.EnergyWindow(5.0, manifold_gap=1.0), solver="dmrg",
               solver_options=dict(max_bond=16), report=False).run()


def test_a_cheap_ci_window_hands_the_network_its_first_rung(ref):
    """The handoff is the same one the conventional-CI route takes, and on this route it is
    what a pilot campaign costs: with an upstream estimate no pilot runs at all."""
    from kuiva.dmrg import NetworkGraph

    window = kuiva.EnergyWindow(5.0, manifold_gap=1.0)
    pre = CheapCI(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=window,
                  report=False).run()
    assert pre.n_states == 2
    stage = CASSCF(pre, n_states=window, solver="dmrg",
                   solver_options=dict(max_bond=16),
                   graph=NetworkGraph(2, [(0, 1)], contents=[(0, 1, 2), (3, 4, 5)]),
                   report=False).run()
    assert stage.n_states == 2
    assert stage.window.extra.get("pilot") is None          # no pilot anywhere in the run
    # ⚠ stage.window is the re-resolution at the CONVERGED orbitals, and its first rung comes
    # from what this calculation already resolved — the handoff is on the resolution at the
    # STARTING orbitals, which is what window_initial is for
    assert stage.window_initial.first_rung_source == "the upstream estimate"
    assert stage.window_initial.rungs[0].n_roots == 2
    assert stage.window.first_rung_source == ("the previous round's count plus a "
                                              "witness pair")
    assert stage.window.rungs[0].n_roots == 4


def test_casci_carries_the_solver_options(cas_term):
    """``solver_options`` reach the CI solver, and the Kramers-restricted mode is what it
    claims to be — the same six states from three time-reversal pairs."""
    general = CASCI(cas_term, n_states=6, report=False).run()
    restricted = CASCI(cas_term, n_states=6, report=False,
                       solver_options=dict(kramers="restricted")).run()
    assert np.max(np.abs(restricted.energies - general.energies)) < KRAMERS_TOL
    assert restricted.result.n_apply < general.result.n_apply


def test_casci_feeds_nevpt2_and_the_property_dump(cas_term, tmp_path):
    """Downstream is the whole point of it being a stage: the analytic Lande factors come
    back through the dump, and the perturbation runs on the CASCI reference."""
    ci = CASCI(cas_term, n_states=6, report=False).run()
    dump = PropertyDump(ci, tmp_path / "b_casci.props", report=False).run()
    levels = dump.matrices.analyse()
    assert [level.size for level in levels] == [2, 4]
    assert all(abs(g - G_LANDE[level.size]) < 2e-3
               for level in levels for g in level.g_values)
    assert dump.states_stage is ci
    pt = NEVPT2(ci, report=False).run()
    assert pt.result.complete
    assert abs(pt.total_energies[1] - pt.total_energies[0]) < 1e-8
    # the CASCI at the CASSCF's own orbitals over its own average IS that CASSCF's last CI,
    # so the correction is the one the CASSCF reference would have got
    assert np.max(np.abs(pt.e2 - NEVPT2(cas_term, report=False).run().e2)) < 1e-10


def test_a_statement_about_orbitals_belongs_to_those_orbitals(ref, cas, pre):
    """⚠ The rule the stage's refusals encode, twice. A character selection reads atomic
    populations off the reference's own SCF orbitals; the CI here runs at orbitals that have
    moved, so re-running it could legitimately return a different set and the spectrum would
    not be the one belonging to them — with nothing in the output saying so. ``coeff=`` is
    the same question asked from the other side."""
    with pytest.raises(ValueError, match="character="):
        CASCI(cas, character=("B", "p"), n_active=6, n_active_elec=1, n_states=2)
    with pytest.raises(ValueError, match="character="):
        CASCI(pre, character=("B", "p"), n_active=6, n_active_elec=1, n_states=2)
    with pytest.raises(ValueError, match="two answers"):
        CASCI(cas, coeff=cas.coeff, n_states=2)
    with pytest.raises(ValueError, match="coeff="):
        CASCI(ref, coeff=cas.coeff, character=("B", "p"), n_active=6, n_states=2)
    # a selection argument with no selection is not silently ignored either
    with pytest.raises(ValueError, match="n_active_elec="):
        CASCI(cas, n_active_elec=1, n_states=2)
    # ... and on a bare Reference there is nothing to inherit, so it must be stated
    with pytest.raises(ValueError, match="active space"):
        CASCI(ref, n_states=2)


def test_casci_runs_at_orbitals_of_its_own(ref, cas):
    """The one place ``coeff=`` is accepted: a Reference upstream, where there is nothing to
    inherit and the orbitals came from somewhere else. The space is then stated as indices,
    which is a statement about exactly those orbitals."""
    ci = CASCI(ref, active=cas.active.spaces.active.tolist(), n_active_elec=1,
               coeff=cas.coeff, n_states=2, report=False).run()
    assert np.max(np.abs(ci.energies - cas.energies)) < KRAMERS_TOL


def test_pseudospin_export_takes_the_stage_that_optimized_the_orbitals(cas):
    # ⚠ Not an oversight: the export re-solves its model space from the integrals the
    # orbitals define and never looks at the CI states, so it belongs on the stage that
    # produced the orbitals
    ci = CASCI(cas, n_states=2, report=False).run()
    with pytest.raises(TypeError, match="finished CASSCF"):
        PseudospinExport(ci, "unused.psd")


# --- the wall-clock deadline through the stage -----------------------------------------------

class _Countdown(kuiva.util.deadline.Deadline):
    """A deadline on a scripted clock: expires after ``stop_after`` macro-iterations.

    The mechanism itself — the queue probes, the reserve arithmetic, the forced write — is
    tested in ``test_deadline.py``; what these tests add is that the stage carries it, so
    they assert the *decision* and never the machine's speed.
    """

    def __init__(self, stop_after):
        super().__init__(duration=1e9, source="a scripted clock", safety=0.0)
        self.stop_after, self.seen = int(stop_after), 0

    def observe(self, seconds):
        super().observe(seconds)
        self.seen += 1

    def remaining(self):
        return 0.0 if self.seen >= self.stop_after else 1e9


def test_there_is_no_deadline_unless_one_is_asked_for(ref, cas):
    """⚠ The default that matters: a cluster with no queue limit is an ordinary place to
    run, so nothing is read, nothing is printed and nothing stops the run early."""
    assert cas.deadline is None
    # "auto" is the portable spelling — it must run unchanged where there is no queue
    stage = CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=2,
                   deadline="auto", report=False)
    assert stage.deadline is not None and stage.deadline.unlimited
    assert abs(stage.run().energy - cas.energy) < E_TOL


def test_a_queue_deadline_that_cannot_be_read_refuses_at_construction(ref, monkeypatch):
    """⚠ Eager, like every other option here, and a refusal rather than a silent absence:
    a run that asked for the allocation's limit and did not get one is killed at the wall
    with nothing written and nothing in the output saying the request failed."""
    for name in ("SLURM_JOB_ID", "SLURM_JOBID", "SLURM_JOB_END_TIME"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="could not be read"):
        CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, deadline="slurm")


def test_the_deadline_stops_the_stage_and_leaves_a_restart_point(ref, tmp_path,
                                                                 kuiva_caplog):
    """The cadence here suppresses every ordinary write (an hour of minimum interval), so
    the file existing proves the deadline forced the final one — and the summary says the
    result is an iterate rather than an answer."""
    path = tmp_path / "b_deadline.h5"
    stopped = CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=2,
                     checkpoint=path, checkpoint_options=dict(min_interval=3600.0),
                     deadline=_Countdown(stop_after=2), max_iter=50, report=False).run()

    assert not stopped.converged
    assert stopped.orbital.n_iterations == 2
    assert stopped.deadline.fired
    assert path.exists() and kuiva.read_checkpoint(path).iteration == 2
    assert "stopped by" in stopped.summary()
    assert any("deadline" in r.getMessage() for r in kuiva_caplog.records)

    # and it is a restart point, not just a file
    resumed = CASSCF(ref, restart=path, n_states=2, max_iter=60, report=False).run()
    assert resumed.converged


def test_a_signal_stops_the_stage_and_the_next_one_refuses_to_start(ref, tmp_path,
                                                                    kuiva_caplog):
    """⚠ The half the deadline cannot cover: a kill nobody announced. The run stops at the
    next macro-iteration boundary with its checkpoint forced past an hour of minimum
    interval — and the NEVPT2 after it refuses, which is what makes a signalled run *exit*
    rather than pause on its way to being killed anyway."""
    import os
    import signal as _signal

    from kuiva.util.signals import StopRequested, clear, pending

    path = tmp_path / "b_signal.h5"

    def kill_at_two(info):
        if info["iteration"] == 2:
            os.kill(os.getpid(), _signal.SIGUSR1)      # never SIGTERM in a test
        return None

    stopped = CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=2,
                     checkpoint=path, checkpoint_options=dict(min_interval=3600.0),
                     signals=("USR1",), callback=kill_at_two, max_iter=50,
                     report=False).run()

    assert not stopped.converged and stopped.orbital.n_iterations == 3
    assert stopped.signals.fired and pending() is not None
    assert path.exists() and kuiva.read_checkpoint(path).iteration == 3
    assert "stopped by" in stopped.summary() and "SIGUSR1" in stopped.summary()
    assert any("SIGUSR1" in r.getMessage() for r in kuiva_caplog.records)
    # ⚠ the handler is a loan: the stage gave it back
    assert _signal.getsignal(_signal.SIGUSR1) is _signal.SIG_DFL

    with pytest.raises(StopRequested, match="was not started"):
        NEVPT2(stopped).run()
    with pytest.raises(StopRequested, match="was not started"):
        CASSCF(ref, character=("B", "p"), n_active=6, n_active_elec=1, n_states=2,
               report=False).run()

    clear()                                            # the next job is a new process
    assert CASCI(stopped, n_states=2, report=False).run().energies.size == 2


def test_nothing_is_caught_unless_signals_are_asked_for(cas):
    """⚠ Never a default: a library that installs signal handlers behind your back breaks
    embedding, test runners and notebooks."""
    import signal as _signal

    assert cas.signals is None
    assert _signal.getsignal(_signal.SIGTERM) is _signal.SIG_DFL


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
                                  "CASSCF", "CASCI", "NEVPT2", "PropertyDump",
                                  "PseudospinExport"}


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
