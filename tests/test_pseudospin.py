"""Tests for the pseudospin assignment and the Ouluspin export.

Everything numerical is asserted through **phase-invariant reductions only**:
degeneracy patterns, relative energies, principal g values, and diagonal moment
expectation values — never a matrix element, because the dump fixes no phase convention.

The sharpest oracle is the sharp one: a ``p^1`` free ion with ``H = zeta L.S`` has
*analytic* Landé g factors — exactly 2/3 for the j = 1/2 doublet and 4/3 for the j = 3/2
quartet (with g_e taken as exactly 2) — independent of every convention involved. The
ground doublet is pushed through the **full route**: network solve, local-multiplet model,
pseudospin assignment, file write, file read.
"""
import numpy as np
import pytest

from kuiva.dmrg import (NetworkGraph, hamiltonian_product_terms,
                        one_electron_product_terms, solve_manifold)
from kuiva.props.multiplet import degeneracy_pattern
from kuiva.props.pseudospin import (FORMAT_VERSION, PseudospinModel, assign_pseudospin,
                                    pseudospin_from_model, read_pseudospin,
                                    write_pseudospin)

G_TOL = 1e-9


def p1_operators(zeta=0.02):
    """``H = zeta L.S`` and ``mu = -(L + 2 S)`` over the six p^1 spinors ``|m_l> (x) |m_s>``.

    ``g_e = 2`` exactly, so the Landé factors are the exact fractions 2/3 and 4/3.
    """
    lz = np.diag([-1.0, 0.0, 1.0]).astype(np.complex128)
    lp = np.zeros((3, 3), dtype=np.complex128)
    lp[1, 0] = lp[2, 1] = np.sqrt(2.0)
    lx = 0.5 * (lp + lp.conj().T)
    ly = -0.5j * (lp - lp.conj().T)
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
    i2, i3 = np.eye(2), np.eye(3)
    l_ops = np.stack([np.kron(a, i2) for a in (lx, ly, lz)])
    s_ops = np.stack([np.kron(i3, b) for b in (sx, sy, sz)])
    h = zeta * sum(l_ops[k] @ s_ops[k] for k in range(3))
    mu = -(l_ops + 2.0 * s_ops)
    return h, mu


def p1_manifold(n_roots, dims, seed=3):
    """The p^1 model through the network + manifold machinery.

    Two 3-mode nodes: the single two-site problem spans the whole 6-dimensional CI space,
    so even a 6-root average fits every local solve.
    """
    h, mu = p1_operators()
    n = 6
    eri = np.zeros((n, n, n, n), dtype=np.complex128)
    ops = {name: one_electron_product_terms(mu[k])
           for k, name in enumerate(("mu_x", "mu_y", "mu_z"))}
    graph = NetworkGraph(2, [(0, 1)], contents=[(0, 1, 2), (3, 4, 5)])
    return solve_manifold(hamiltonian_product_terms(h, eri), graph, 1,
                          sites=[(0, 1)], rule="dimension", dims=dims,
                          operators=ops, n_roots=n_roots, max_roots=n_roots,
                          rng=np.random.default_rng(seed))


# --- the full route: network -> model -> pseudospin -> file -> parse -----------------------

def test_ground_doublet_lande_g_through_the_full_route(tmp_path):
    result = p1_manifold(n_roots=2, dims=2)
    assert result.converged
    ps = pseudospin_from_model(result.model)
    site = ps.sites[0]
    assert site.twice_s == 1
    assert max(abs(g - 2.0 / 3.0) for g in site.g_values) < G_TOL   # analytic Lande
    assert site.n_electrons == 1
    # M convention: M = -S..+S ascending (OuluSpin order), so the diagonal is -g_J M
    # for M = (-1/2, +1/2) — diagonal expectation values are phase invariant
    diag = np.real(np.diag(np.tensordot(site.axis, site.moment, axes=(0, 0))))
    assert np.allclose(diag, [+1.0 / 3.0, -1.0 / 3.0], atol=1e-8)
    assert ps.unitarity_error() < 1e-12

    path = write_pseudospin(tmp_path / "p1.psd", ps, title="p1 ground doublet")
    back = read_pseudospin(path)
    assert int(back["header"]["format_version"]) == FORMAT_VERSION
    assert back["header"]["hamiltonian_is_diagonal"] == "no"
    assert back["header"]["frame"] == "input frame"
    assert np.allclose(back["frame_rotation"], np.eye(3))
    assert back["sites"][0]["twice_s"] == 1
    assert [tuple(row) for row in back["basis"]] == [(-1,), (1,)]
    for name in ("H", "mu_x", "mu_y", "mu_z", "U"):
        assert np.allclose(back["matrices"][name],
                           {"H": ps.h, "mu_x": ps.mu[0], "mu_y": ps.mu[1],
                            "mu_z": ps.mu[2], "U": ps.unitary}[name], atol=1e-14)
    assert np.allclose(back["site_matrices"][(0, "mu_z")], site.moment[2], atol=1e-14)


def test_full_p1_spectrum_gives_both_multiplets_their_lande_g():
    """SA over all six roots, no truncation: the model reproduces the 2 + 4 j pattern and
    both analytic Landé factors through the phase-invariant reduction."""
    from kuiva.props.multiplet import analyse_spectrum

    result = p1_manifold(n_roots=6, dims=6, seed=4)
    model = result.model
    mu = np.stack([model.operator_in_eigenbasis(n)
                   for n in ("mu_x", "mu_y", "mu_z")])
    multiplets = analyse_spectrum(model.spectrum(), mu)
    assert degeneracy_pattern(multiplets) == (2, 4)
    assert max(abs(g - 2.0 / 3.0) for g in multiplets[0].g_values) < G_TOL
    assert max(abs(g - 4.0 / 3.0) for g in multiplets[1].g_values) < G_TOL


# --- pure-linear-algebra units (no network) ------------------------------------------------

def test_j32_quartet_m_labelling_matches_lande():
    """Project onto the j = 3/2 quartet: 2S = 3, g = 4/3, and the M-ordered moment
    diagonal is exactly ``-g_J M`` for M = -3/2 .. +3/2 (diagonal expectation values are
    phase invariant, so this asserts the stated convention, not a phase)."""
    h, mu = p1_operators()
    evals, vecs = np.linalg.eigh(h)
    v = vecs[:, 2:]                                    # the quartet (upper 4 states)
    h_eff = v.conj().T @ h @ v
    mu_p = np.stack([v.conj().T @ m @ v for m in mu])
    ps = assign_pseudospin(h_eff, mu_p, [4], [mu_p], site_electrons=[1])
    site = ps.sites[0]
    assert site.twice_s == 3
    assert max(abs(g - 4.0 / 3.0) for g in site.g_values) < G_TOL
    g_j = 4.0 / 3.0
    diag = np.real(np.diag(np.tensordot(site.axis, site.moment, axes=(0, 0))))
    assert np.allclose(diag, [-g_j * m for m in (-1.5, -0.5, 0.5, 1.5)], atol=1e-8)
    assert site.twice_m == (-3, -1, 1, 3)


def spin1_ops():
    sz = np.diag([1.0, 0.0, -1.0]).astype(np.complex128)
    sp = np.zeros((3, 3), dtype=np.complex128)
    sp[0, 1] = sp[1, 2] = np.sqrt(2.0)
    sx = 0.5 * (sp + sp.conj().T)
    sy = -0.5j * (sp - sp.conj().T)
    return np.stack([sx, sy, sz])


def test_two_site_exchange_pattern_and_basis_listing(tmp_path):
    """S = 1 (x) S = 1 isotropic exchange: degeneracies 1 + 3 + 5, the Heisenberg ladder,
    a 9-state |1,M> (x) |1,M'> listing, and a byte-exact round trip."""
    s1 = spin1_ops()
    i3 = np.eye(3, dtype=np.complex128)
    sa = np.stack([np.kron(m, i3) for m in s1])
    sb = np.stack([np.kron(i3, m) for m in s1])
    j = 0.1
    h_eff = j * sum(sa[k] @ sb[k] for k in range(3))
    mu = -2.0 * (sa + sb)                              # spin-only, g = 2
    site_mu = [-2.0 * s1, -2.0 * s1]
    ps = assign_pseudospin(h_eff, mu, [3, 3], site_mu, site_electrons=[1, 1],
                           provenance={"model": "S=1 pair"})
    assert [s.twice_s for s in ps.sites] == [2, 2]
    labels = ps.basis_labels()
    assert labels[0] == (-2, -2) and labels[-1] == (2, 2) and len(labels) == 9

    rel = ps.energies - ps.energies.min()
    from kuiva.props.multiplet import degenerate_blocks
    sizes = [b for _, b in degenerate_blocks(rel / j, tol_cm=1e-9)]
    assert sizes == [1, 3, 5]
    e1, e2 = float(np.mean(rel[1:4])), float(np.mean(rel[4:9]))
    assert e2 / e1 == pytest.approx(3.0, abs=1e-9)     # exact for pure exchange

    path = write_pseudospin(tmp_path / "pair.psd", ps)
    back = read_pseudospin(path)
    assert back["provenance"] == {"model": "S=1 pair"}
    assert np.allclose(back["matrices"]["H"], ps.h, atol=1e-14)
    assert [tuple(row) for row in back["basis"]] == labels


def test_common_axis_and_frame_rotation():
    """``common_axis="ground-doublet"`` labels every site along one quantization axis,
    and ``rotate_frame`` re-expresses the components in its principal triad (z = axis) —
    checked through invariants: the g values are frame independent, the rotated site
    moment is diagonal along plain z, and the rotation round-trips through the file."""
    s1 = spin1_ops()
    i3 = np.eye(3, dtype=np.complex128)
    sa = np.stack([np.kron(m, i3) for m in s1])
    sb = np.stack([np.kron(i3, m) for m in s1])
    h_eff = 0.1 * sum(sa[k] @ sb[k] for k in range(3))
    mu = -2.0 * (sa + sb)
    site_mu = [-2.0 * s1, -2.0 * s1]

    ps = assign_pseudospin(h_eff, mu, [3, 3], site_mu,
                           common_axis="ground-doublet", rotate_frame=True)
    assert ps.frame.startswith("quantization-axis frame")
    r = ps.frame_rotation
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) > 0.0
    for s in ps.sites:
        assert s.axis_choice == "common (ground doublet)"
        assert np.allclose(s.axis, [0.0, 0.0, 1.0])
        # in the rotated frame the labelling axis IS z: mu_z diagonal, M ascending
        assert np.allclose(s.moment[2] - np.diag(np.diag(s.moment[2])), 0.0, atol=1e-10)
        assert max(abs(g - 2.0) for g in s.g_values) < 1e-9      # frame invariant

    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        path = write_pseudospin(pathlib.Path(d) / "f.psd", ps)
        back = read_pseudospin(path)
        assert back["header"]["frame"].startswith("quantization-axis")
        assert np.allclose(back["frame_rotation"], r, atol=1e-12)

    with pytest.raises(ValueError, match="common_axis"):
        assign_pseudospin(h_eff, mu, [3, 3], site_mu, rotate_frame=True)
    with pytest.raises(ValueError, match="not both"):
        assign_pseudospin(h_eff, mu, [3, 3], site_mu, common_axis=(0, 0, 1),
                          axes=[(0, 0, 1), (0, 0, 1)])


def test_labelling_axis_can_be_given_and_is_normalized():
    h, mu = p1_operators()
    evals, vecs = np.linalg.eigh(h)
    v = vecs[:, :2]
    h_eff = v.conj().T @ h @ v
    mu_p = np.stack([m2 @ np.eye(2) for m2 in
                     (v.conj().T @ mu[0] @ v, v.conj().T @ mu[1] @ v,
                      v.conj().T @ mu[2] @ v)])
    ps = assign_pseudospin(h_eff, mu_p, [2], [mu_p], axes=[(0.0, 0.0, 2.0)])
    assert ps.sites[0].axis_choice == "given"
    assert np.allclose(ps.sites[0].axis, [0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="nonzero 3-vector"):
        assign_pseudospin(h_eff, mu_p, [2], [mu_p], axes=[(0.0, 0.0, 0.0)])


# --- refusals -------------------------------------------------------------------------------

def test_charge_mixed_site_space_is_refused():
    """|S, M> labels a multiplet; a site space mixing particle-number sectors is not one,
    and the composite-spin manifold of the Heisenberg pair is exactly such a space."""
    from test_dmrg_manifold import _pair_model
    from kuiva.dmrg.manifold import effective_model
    from kuiva.dmrg.ttno import ProductTerm

    ttno, state, sweep = _pair_model(0.02, seed=21)
    sz = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.complex128)
    ops = {name: [ProductTerm(-2.0, (m,), (sz,)) for m in range(4)]
           for name in ("mu_x", "mu_y", "mu_z")}
    model = effective_model(ttno, state, [(0, 1), (2, 3)], weights=sweep.weights,
                            rule="dimension", dims=3, operators=ops, report=False)
    assert all(sp.n_electrons is None for sp in model.sites)
    with pytest.raises(ValueError, match="particle-number sectors"):
        pseudospin_from_model(model)


def test_missing_moment_operators_are_refused():
    result = p1_manifold(n_roots=2, dims=2, seed=5)
    model = result.model
    model.operators.pop("mu_y")
    with pytest.raises(ValueError, match="mu_y"):
        pseudospin_from_model(model)


def test_reader_refuses_an_unknown_format_version(tmp_path):
    h, mu = p1_operators()
    evals, vecs = np.linalg.eigh(h)
    v = vecs[:, :2]
    ps = assign_pseudospin(v.conj().T @ h @ v,
                           np.stack([v.conj().T @ m @ v for m in mu]), [2],
                           [np.stack([v.conj().T @ m @ v for m in mu])])
    path = write_pseudospin(tmp_path / "v.psd", ps)
    text = path.read_text().replace("format_version                   1",
                                    "format_version                   999")
    path.write_text(text)
    with pytest.raises(ValueError, match="format_version"):
        read_pseudospin(path)


def test_empty_provenance_warns(kuiva_caplog, tmp_path):
    """The provenance obligation transfers here: a file with no Hamiltonian provenance says so."""
    h, mu = p1_operators()
    evals, vecs = np.linalg.eigh(h)
    v = vecs[:, :2]
    mu_p = np.stack([v.conj().T @ m @ v for m in mu])
    ps = assign_pseudospin(v.conj().T @ h @ v, mu_p, [2], [mu_p])
    write_pseudospin(tmp_path / "bare.psd", ps)
    assert any("provenance" in r.message for r in kuiva_caplog.records)


def test_a_pseudospin_export_round_trips_through_from_file(tmp_path):
    """The sibling of ``PropertyMatrices.from_dump``, and there for the same reason: the
    phases in the file are arbitrary, so two stored exports can only be compared through
    ``analyse()`` — which needs the model object, not ``read_pseudospin``'s dictionary."""
    s1 = spin1_ops()
    i3 = np.eye(3, dtype=np.complex128)
    sa = np.stack([np.kron(m, i3) for m in s1])
    sb = np.stack([np.kron(i3, m) for m in s1])
    h_eff = 0.1 * sum(sa[k] @ sb[k] for k in range(3))
    ps = assign_pseudospin(h_eff, -2.0 * (sa + sb), [3, 3], [-2.0 * s1, -2.0 * s1],
                           site_electrons=[1, 1], orbitals=[(0, 1, 2), (3, 4, 5)],
                           provenance={"model": "S=1 pair"})
    path = write_pseudospin(tmp_path / "rt.psd", ps)

    back = PseudospinModel.from_file(path)
    assert np.array_equal(back.h, ps.h) and np.array_equal(back.mu, ps.mu)
    assert np.array_equal(back.unitary, ps.unitary)
    assert np.array_equal(back.energies, ps.energies)
    assert back.dims == ps.dims and back.frame == ps.frame
    assert back.energy_shift == ps.energy_shift
    assert back.provenance == {"model": "S=1 pair"}
    assert np.allclose(back.frame_rotation, ps.frame_rotation)
    assert back.basis_labels() == ps.basis_labels()

    for a, b in zip(back.sites, ps.sites):
        assert (a.index, a.twice_s, a.n_electrons, a.orbitals) == \
               (b.index, b.twice_s, b.n_electrons, b.orbitals)
        assert np.allclose(a.axis, b.axis) and a.axis_choice == b.axis_choice
        assert np.array_equal(a.moment, b.moment)
        # ⚠ g values are RECOMPUTED from the moments rather than read: a stored reduction is a
        # second thing that can disagree with the matrices it came from. They must therefore
        # agree to the arithmetic, not merely to the file's printed precision.
        assert a.g_values == pytest.approx(b.g_values, rel=1e-12)

    assert [m.size for m in back.analyse()] == [m.size for m in ps.analyse()]
