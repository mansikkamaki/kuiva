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
from kuiva.props.multiplet import block_collinearity, degeneracy_pattern
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


# --- the hyperfine field on the model space --------------------------------------------------
#
# ⚠ **What these can fail on.** The operator is built as ``T = a L + b S`` with ``a != b``, so
# it is genuinely a *different* vector operator from ``mu = -(L + 2 S)`` — one whose principal
# values and whose spatial content differ — and yet Wigner-Eckart forces it to be proportional
# to ``mu`` **inside each j manifold**. That is the sharp oracle: ``Tr_b(mu.T)^2 =
# Tr_b(mu.mu) Tr_b(T.T)`` exactly, and it breaks on a permuted Cartesian component, on a frame
# left unrotated, and on the conjugation trap of the transformation law — none of which any
# hermiticity, degeneracy or magnitude check here can see. The ratio ``A(1/2)/A(3/2)``, which
# the CI route asserts against the analytic 5, is *not* a claim about this synthetic operator:
# it is a claim about the real hyperfine integrals and belongs where they are.

def p1_hyperfine(a=1.0, b=0.25):
    """``T = a L + b S`` over the six ``p^1`` spinors — a vector operator that is NOT ``mu``.

    The shape that matters is the only one that matters: a Hermitian, time-odd vector operator
    with different orbital and spin weights from the moment's. Everything asserted about it is
    a consequence of it being *a* vector operator, which is exactly the content of the
    Wigner-Eckart checks the real hyperfine field has to pass.
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
    l_ops = np.stack([np.kron(m, i2) for m in (lx, ly, lz)])
    s_ops = np.stack([np.kron(i3, m) for m in (sx, sy, sz)])
    return a * l_ops + b * s_ops


NUCLEUS = {"atom": 1, "atom_label": "X1", "element": "X", "atomic_number": 5,
           "label": "11X", "twice_spin": 3, "g": 1.7924326, "quadrupole_barn": 0.040591,
           "position_bohr": [0.0, 0.0, 0.0], "source": "synthetic (test)"}
RECORD = {"unit": "Eh/mu_N", "decoupling": "1e", "nuclear_model": "point",
          "picture_change": "always applied", "g_electron": 2.0,
          "operator": "H_hf = sum_k g_N(k) sum_u T_k_u (x) I_k_u",
          "x2c_response": "not included (unperturbed X and R)",
          "two_electron_picture_change": "not applied", "nuclear_magneton_au": 2.7e-4}


def p1_projected(states=slice(None)):
    """``(h_eff, mu, T)`` of the p^1 model projected onto a set of its eigenstates."""
    h, mu = p1_operators()
    t = p1_hyperfine()
    _, vecs = np.linalg.eigh(h)
    v = vecs[:, states]
    return (v.conj().T @ h @ v,
            np.stack([v.conj().T @ m @ v for m in mu]),
            np.stack([v.conj().T @ m @ v for m in t]))


def test_the_hyperfine_field_is_collinear_with_the_moment_on_a_j_manifold():
    """The Wigner-Eckart oracle, on an operator deliberately unlike ``mu``.

    Both invariants are isotropic inside one ``2J+1`` block and Cauchy-Schwarz is saturated,
    so ``A.g`` is exactly ``+-1`` — a statement no convention in this program can move, and
    the one that fails on a permuted component or a conjugation error.
    """
    from kuiva.props.multiplet import block_cross_tensor, block_hyperfine_tensor

    for states, dim in ((slice(0, 2), 2), (slice(2, 6), 4)):
        h_eff, mu_p, t_p = p1_projected(states)
        ps = assign_pseudospin(h_eff, mu_p, [dim], [mu_p], site_electrons=[1],
                               hyperfine={"X1": t_p}, hyperfine_nuclei=[NUCLEUS],
                               hyperfine_record=RECORD)
        t_eig = ps.hyperfine_in_eigenbasis()["X1"]
        mu_eig = ps.mu_in_eigenbasis()
        tensor = block_hyperfine_tensor(t_eig, 0, dim)
        cross = block_cross_tensor(mu_eig, t_eig, 0, dim)
        for x in (tensor, cross):
            assert np.abs(x - np.diag(np.diag(x))).max() < 1e-12 * np.abs(x).max()
            assert np.diag(x) == pytest.approx([np.diag(x)[0]] * 3, rel=1e-12)
        assert abs(abs(block_collinearity(mu_eig, t_eig, 0, dim)) - 1.0) < 1e-12
        # ...and T really is a different operator from mu, so the agreement is not trivial.
        assert abs(np.trace(cross) / np.trace(block_hyperfine_tensor(mu_eig, 0, dim))
                   - 1.0) > 0.1


def test_the_hyperfine_field_rides_the_frame_rotation_with_the_moment():
    """⚠ The file states **one** frame for every operator in it.

    A ``T`` left behind in the input frame stays Hermitian, keeps every principal value it
    had, and sits at an arbitrary angle to the ``mu`` a consumer pairs it with — invisible in
    each operator alone and fatal to the pair. The invariant that sees it is the *mixed* one:
    the collinearity is frame independent only if both operators were rotated together.
    """
    h_eff, mu_p, t_p = p1_projected(slice(0, 2))
    axis = np.array([0.3, -0.5, 0.81])
    plain = assign_pseudospin(h_eff, mu_p, [2], [mu_p], common_axis=axis,
                              hyperfine={"X1": t_p}, hyperfine_nuclei=[NUCLEUS],
                              hyperfine_record=RECORD)
    rotated = assign_pseudospin(h_eff, mu_p, [2], [mu_p], common_axis=axis,
                                rotate_frame=True, hyperfine={"X1": t_p},
                                hyperfine_nuclei=[NUCLEUS], hyperfine_record=RECORD)
    assert rotated.frame != plain.frame
    assert not np.allclose(rotated.frame_rotation, np.eye(3))
    # the operator moved with the frame...
    assert not np.allclose(rotated.hyperfine["X1"], plain.hyperfine["X1"], atol=1e-8)
    # ...and the mixed invariant, which is what would have caught it had it not.
    before = block_collinearity(plain.mu_in_eigenbasis(),
                                plain.hyperfine_in_eigenbasis()["X1"], 0, 2)
    after = block_collinearity(rotated.mu_in_eigenbasis(),
                               rotated.hyperfine_in_eigenbasis()["X1"], 0, 2)
    assert after == pytest.approx(before, abs=1e-12)
    # a rotation is orthogonal, so the quadratic invariant is unchanged too
    from kuiva.props.multiplet import block_hyperfine_tensor
    assert np.trace(block_hyperfine_tensor(rotated.hyperfine_in_eigenbasis()["X1"], 0, 2)) \
        == pytest.approx(np.trace(block_hyperfine_tensor(
            plain.hyperfine_in_eigenbasis()["X1"], 0, 2)), rel=1e-12)


def test_the_two_halves_of_the_hyperfine_contract_cannot_come_apart():
    """Matrices without a table name no nucleus and state no ``I``; a table without matrices
    describes a coupling the file cannot supply. Both are refused, in both directions,
    because a ``T`` matched to the wrong nucleus is Hermitian, plausible and wrong."""
    h_eff, mu_p, t_p = p1_projected(slice(0, 2))
    with pytest.raises(ValueError, match="same nuclei"):
        assign_pseudospin(h_eff, mu_p, [2], [mu_p], hyperfine={"X1": t_p},
                          hyperfine_nuclei=[dict(NUCLEUS, atom_label="Y1")])
    with pytest.raises(ValueError, match="no hyperfine matrices"):
        assign_pseudospin(h_eff, mu_p, [2], [mu_p], hyperfine_nuclei=[NUCLEUS])
    with pytest.raises(ValueError, match=r"must be \(3, 2, 2\)"):
        assign_pseudospin(h_eff, mu_p, [2], [mu_p], hyperfine={"X1": t_p[:, :1, :1]},
                          hyperfine_nuclei=[NUCLEUS])
    # ⚠ absent means None, never {}: "no nuclei were selected" and "the coupling is small"
    # are different statements and a whole active space can be the difference.
    bare = assign_pseudospin(h_eff, mu_p, [2], [mu_p])
    assert bare.hyperfine is None and not bare.has_hyperfine
    assert bare.hyperfine_labels == () and bare.hyperfine_nuclei == ()


def test_the_hyperfine_half_round_trips_through_the_file(tmp_path):
    """Written and read back element for element, table included — and rebuilt from the
    file's own two sections rather than the provenance JSON, since a consumer that parses no
    JSON must still get the nuclei."""
    h_eff, mu_p, t_p = p1_projected(slice(0, 2))
    ps = assign_pseudospin(h_eff, mu_p, [2], [mu_p], site_electrons=[1],
                           hyperfine={"X1": t_p}, hyperfine_nuclei=[NUCLEUS],
                           hyperfine_record=RECORD, provenance={"model": "p1"})
    path = write_pseudospin(tmp_path / "hf.psd", ps)

    raw = read_pseudospin(path)
    assert len(raw["nuclei"]) == 1
    row = raw["nuclei"][0]
    assert row["atom_label"] == "X1" and row["label"] == "11X"
    assert row["twice_spin"] == 3 and row["g"] == pytest.approx(NUCLEUS["g"])
    assert row["quadrupole_barn"] == pytest.approx(NUCLEUS["quadrupole_barn"])
    for k, a in enumerate("xyz"):
        assert np.array_equal(raw["matrices"]["T_X1_" + a], ps.hyperfine["X1"][k])
    header = raw["header"]
    assert header["hyperfine_unit"] == "Eh/mu_N"
    assert header["hyperfine_decoupling"] == "1e"
    assert header["hyperfine_nuclear_model"] == "point"
    assert "unperturbed" in header["hyperfine_x2c_response"]
    assert "not applied" in header["hyperfine_2e_picture_change"]
    assert header["n_hyperfine_nuclei"] == "1"
    # ⚠ reported, never refused: 2 electronic states x (2I+1 = 4) nuclear
    assert header["product_dim"] == "8" and ps.product_dim() == 8
    assert "M_I = -I .. +I ascending" in header["nuclear_site_order"]

    back = PseudospinModel.from_file(path)
    assert back.hyperfine_labels == ("X1",)
    assert np.array_equal(back.hyperfine["X1"], ps.hyperfine["X1"])
    assert back.nucleus("X1")["label"] == "11X"
    assert back.hyperfine_record["nuclear_model"] == "point"
    with pytest.raises(KeyError, match="no nucleus labelled"):
        back.nucleus("X2")
    for a, b in zip(back.analyse(), ps.analyse()):
        assert np.allclose(a.hyperfine["X1"], b.hyperfine["X1"], rtol=0, atol=0)
        assert np.allclose(a.hyperfine_cross["X1"], b.hyperfine_cross["X1"], rtol=0, atol=0)


def test_a_file_without_hyperfine_says_nothing_about_it(tmp_path):
    h_eff, mu_p, _ = p1_projected(slice(0, 2))
    ps = assign_pseudospin(h_eff, mu_p, [2], [mu_p], provenance={"model": "p1"})
    path = write_pseudospin(tmp_path / "plain.psd", ps)
    text = path.read_text()
    assert "[NUCLEI]" not in text and "hyperfine_unit" not in text
    assert "product_dim" not in text
    back = PseudospinModel.from_file(path)
    assert back.hyperfine is None and back.hyperfine_nuclei == ()
    assert read_pseudospin(path)["nuclei"] == []


def test_a_nuclear_table_without_its_operators_is_refused(tmp_path):
    h_eff, mu_p, t_p = p1_projected(slice(0, 2))
    ps = assign_pseudospin(h_eff, mu_p, [2], [mu_p], hyperfine={"X1": t_p},
                           hyperfine_nuclei=[NUCLEUS], hyperfine_record=RECORD,
                           provenance={"model": "p1"})
    path = write_pseudospin(tmp_path / "cut.psd", ps)
    text = path.read_text()
    # ⚠ searched from `start`, not from the top: the file's own header comment mentions
    # "[MATRIX U]", and cutting to that would duplicate half the file instead of trimming it.
    start = text.index("[MATRIX T_X1_x]")
    path.write_text(text[:start] + text[text.index("[MATRIX U]", start):])
    with pytest.raises(ValueError, match="no complete set of"):
        PseudospinModel.from_file(path)


def test_a_header_count_that_disagrees_with_the_table_is_refused(tmp_path):
    h_eff, mu_p, t_p = p1_projected(slice(0, 2))
    ps = assign_pseudospin(h_eff, mu_p, [2], [mu_p], hyperfine={"X1": t_p},
                           hyperfine_nuclei=[NUCLEUS], hyperfine_record=RECORD,
                           provenance={"model": "p1"})
    path = write_pseudospin(tmp_path / "count.psd", ps)
    path.write_text(path.read_text().replace("n_hyperfine_nuclei               1",
                                             "n_hyperfine_nuclei               2"))
    with pytest.raises(ValueError, match="hyperfine nuclei"):
        read_pseudospin(path)


def test_writing_warns_about_what_no_number_in_the_file_shows(tmp_path, kuiva_caplog):
    """The standing obligation, discharged every time the file is written: the treatment of
    the operators and the core-polarization limitation the active space cannot repair."""
    h_eff, mu_p, t_p = p1_projected(slice(0, 2))
    ps = assign_pseudospin(h_eff, mu_p, [2], [mu_p], hyperfine={"X1": t_p},
                           hyperfine_nuclei=[NUCLEUS], hyperfine_record=RECORD,
                           provenance={"model": "p1"})
    path = write_pseudospin(tmp_path / "warn.psd", ps)
    assert any("HYPERFINE FIELD operators" in r.message and "core-s spin polarization"
               in r.message for r in kuiva_caplog.records)
    text = path.read_text()
    assert "H_hf = sum_k g_N(k) sum_u T_<k>_u (x) I_<k>_u" in text
    assert "ISOTOPE-INDEPENDENT" in text


def test_a_large_product_space_is_reported_and_never_refused(tmp_path, kuiva_caplog):
    """⚠ Kuiva does not form the electron-nuclear product space — that is the whole point of
    storing electronic matrices and a nuclear table separately — so a big one is *said*, not
    refused. Refusing on a consumer's behalf teaches a user to raise a limit blindly."""
    from kuiva.props.pseudospin import PRODUCT_DIM_WARN

    s1 = spin1_ops()
    i3 = np.eye(3, dtype=np.complex128)
    sa = np.stack([np.kron(m, i3) for m in s1])
    sb = np.stack([np.kron(i3, m) for m in s1])
    # nine electronic states and two fat nuclei: 9 x 512 x 512, well past the threshold
    nuclei = [dict(NUCLEUS, atom_label="X1", twice_spin=511),
              dict(NUCLEUS, atom_label="X2", twice_spin=511)]
    t = 0.3 * sa - 0.1 * sb
    ps = assign_pseudospin(0.1 * sum(sa[k] @ sb[k] for k in range(3)), -2.0 * (sa + sb),
                           [3, 3], [-2.0 * s1, -2.0 * s1],
                           hyperfine={"X1": t, "X2": t}, hyperfine_nuclei=nuclei,
                           hyperfine_record=RECORD, provenance={"model": "S=1 pair"})
    assert ps.product_dim() == 9 * 512 * 512 > PRODUCT_DIM_WARN
    path = write_pseudospin(tmp_path / "big.psd", ps)         # written, not refused
    assert any("electron-nuclear product space" in r.message
               for r in kuiva_caplog.records)
    assert read_pseudospin(path)["header"]["product_dim"] == str(9 * 512 * 512)


def test_the_report_names_the_reduction_and_stays_ascii(kuiva_caplog):
    """A reader has to be able to tell an ``|A|`` reduction from a fitted tensor, because this
    program writes no tensor — and the output stream is ASCII only."""
    h_eff, mu_p, t_p = p1_projected(slice(0, 2))
    ps = assign_pseudospin(h_eff, mu_p, [2], [mu_p], hyperfine={"X1": t_p},
                           hyperfine_nuclei=[NUCLEUS], hyperfine_record=RECORD)
    ps.report()
    text = kuiva_caplog.text
    assert "hyperfine field operators" in text
    assert "|A_1| [MHz]" in text and "A.g" in text
    assert "X1" in text and "11X" in text
    assert "electron-nuclear product dimension" in text
    for record in kuiva_caplog.records:
        record.getMessage().encode("ascii")


def test_the_model_route_pulls_the_named_operators_and_refuses_a_missing_one():
    """``pseudospin_from_model`` takes the three operator *names* per nucleus, so the naming
    convention stays with the caller that chose it — and a name the model does not carry is
    refused rather than silently dropped."""
    result = p1_manifold(n_roots=2, dims=2)
    with pytest.raises(ValueError, match="T_X1_x"):
        pseudospin_from_model(result.model, hyperfine={"X1": ("T_X1_x", "T_X1_y", "T_X1_z")},
                              hyperfine_nuclei=[NUCLEUS], hyperfine_record=RECORD)
