"""The shared X2C decoupling layer and the molecular four-component blocks (shared decoupling primitives).

Three things are established here, in increasing order of how much they can fail on:

1. **The move did not fork anything.** :mod:`kuiva.x2c.decouple` now defines the block
   container, the metric threshold and the picture change that :mod:`kuiva.amf` used to own.
   Both must be the *same objects*, not equal copies — a second definition of the block
   convention would satisfy every numerical test in the suite while making a future
   four-component backend match the letter of the contract and not its content.

2. **Kuiva's spin conventions reproduce PySCF's private helpers bitwise.** The
   conventions live in :mod:`kuiva.spinor.expand` and nowhere else, so the bridge assembles
   ``sigma . W`` with :func:`~kuiva.spinor.expand.two_component_operator` rather than with
   ``pyscf.x2c.x2c._sigma_dot``. That is only safe if the two agree, and "agree" here means
   **exactly zero difference**, not a tolerance: they are two spellings of one index
   permutation, so any nonzero residual is a convention error, not rounding.

3. **The exact path through the new blocks reproduces PySCF's X2C Hamiltonian**, and where it
   does not, the difference is attributable rather than mysterious. This is the part worth
   reading: agreement is at 1e-13 relative when the metric projection drops nothing, and
   degrades to 6.7e-09 (TiCl3) and 2.4e-07 (TlH) when it drops directions — deliberately,
   because Kuiva applies :data:`~kuiva.x2c.decouple.METRIC_LINDEP_THRESHOLD` to the
   four-component metric and PySCF applies nothing. ⚠ **Neither number is an error bar on the
   physics.** The projected operator is the better-conditioned one, and the assertion that
   says so is on ``max |X|``, not on the difference: decontracted TlH sits at 13.8 with the
   projection and **988** without it, which is the same failure mode that made a Ce correction
   96% time-reversal odd (see :func:`kuiva.x2c.decouple.canonical_orth`). That evidence was
   atomic only; this is its molecular counterpart.
"""
import numpy as np
import pytest

from kuiva.interface.pyscf_bridge import four_component_one_electron
from kuiva.spinor.expand import (decompose_two_component, spin_block_diagonal,
                                 time_reversal_residual, two_component_operator)
from kuiva.x2c.decouple import (METRIC_LINDEP_THRESHOLD, canonical_orth, decoupling_matrices,
                                picture_change)


def _mol(atom, **kw):
    from pyscf import gto
    return gto.M(atom=atom, basis=kw.pop("basis", "x2c-SVPall-2c"), verbose=0, **kw)


@pytest.fixture(scope="module")
def systems():
    """(label, Mole) for the systems used here. Light ones are the exactness checks; TiCl3 and
    TlH are the two that exercise the metric projection, at 1.8 s and 1.4 s respectively."""
    return [
        ("Ne", _mol("Ne 0 0 0")),
        ("H2O", _mol("O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24")),
        ("TiCl3", _mol("Ti 0 0 0; Cl 0 0 2.2; Cl 1.905 0 -1.1; Cl -1.905 0 -1.1", spin=1)),
        ("TlH", _mol("Tl 0 0 0; H 0 0 1.872")),
    ]


def _pyscf_hcore(mol):
    from pyscf.x2c import x2c
    return np.asarray(x2c.SpinOrbitalX2CHelper(mol).get_hcore())


def _exact_x2c(mol):
    """The exact molecular X2C-1e Hamiltonian, assembled through the new blocks."""
    fc = four_component_one_electron(mol)
    x, r = decoupling_matrices(fc.hcore, fc.overlap, fc.light_speed)
    return fc, x, fc.contract(picture_change(fc.hcore, x, r))


# --- 1. The move did not create a second definition ---------------------------------------

def test_moved_names_are_the_same_objects_not_copies():
    """⚠ Identity, not equality. Two modules each defining ``FourComponentBlocks`` would pass
    every numerical test in the suite and still be two incompatible conventions."""
    from kuiva.amf import backend as amf_backend
    from kuiva.amf import decouple as amf_decouple
    from kuiva.x2c import decouple as x2c_decouple

    for name in ("FourComponentBlocks", "LIGHT_SPEED", "METRIC_LINDEP_THRESHOLD",
                 "blocks_memory_gb"):
        assert getattr(amf_backend, name) is getattr(x2c_decouple, name), name
    for name in ("picture_change", "two_component_density", "renormalization"):
        assert getattr(amf_decouple, name) is getattr(x2c_decouple, name), name

    # The AMF module must not have kept a private copy of what it handed over.
    assert not hasattr(amf_decouple, "_canonical_orth")
    assert not hasattr(amf_decouple, "_METRIC_TOL")


def test_the_dependency_runs_one_way():
    """``kuiva.x2c`` may not import ``kuiva.amf``. The whole point of the package is that the
    decoupling is not an atomic-mean-field concept; a back-edge would make it one again."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "kuiva" / "x2c"
    offenders = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("amf"):
                offenders.append((path.name, node.module))
            if isinstance(node, ast.ImportFrom) and node.level and "amf" in (node.module or ""):
                offenders.append((path.name, node.module))
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("kuiva.amf"):
                        offenders.append((path.name, alias.name))
    assert offenders == [], "kuiva.x2c must not depend on kuiva.amf: {}".format(offenders)


# --- 2. The spin conventions reproduce PySCF's helpers, bitwise ---------------------------

def test_kuiva_spin_conventions_reproduce_pyscf_helpers_exactly(systems):
    """⚠ Exact zero, not a tolerance. ``int1e_spnucsp`` returns ``(4, nao, nao)``: the first
    three are the Kuiva-convention ``w_k`` (``W_k = i w_k``) and the fourth is the spin-free
    part, so ``sigma . W`` assembles with Kuiva's own spin function. If this ever fails, the bridge
    is silently using a different spin convention from the rest of the program."""
    from pyscf.x2c import x2c

    for label, mol in systems[:2]:
        raw = mol.intor("int1e_spnucsp")
        assert raw.shape == (4, mol.nao, mol.nao)
        assert np.array_equal(two_component_operator(raw[3], raw[0:3]), x2c._sigma_dot(raw)), \
            label
        s = mol.intor_symmetric("int1e_ovlp")
        assert np.array_equal(spin_block_diagonal(s), x2c._block_diag(s)), label


def test_light_speed_constant_is_not_pyscfs(systems):
    """⚠ Pins a real discrepancy so nobody quietly "fixes" one constant to match the other.

    Kuiva's :data:`kuiva.x2c.decouple.LIGHT_SPEED` is CODATA 2018; PySCF ships an older
    determination. The blocks must be built at **PySCF's**, because PySCF produced the
    integrals — using Kuiva's instead degrades agreement with PySCF's own X2C Hamiltonian from
    6e-13 to 1e-11 relative, a discrepancy with no physical cause whatsoever.
    """
    from pyscf import lib

    from kuiva.x2c.decouple import LIGHT_SPEED

    assert LIGHT_SPEED != lib.param.LIGHT_SPEED
    # Bounded on both sides: they must differ (or this test is vacuous) and the difference
    # must stay a last-digits one (or the two are describing different physics).
    relative = abs(LIGHT_SPEED - lib.param.LIGHT_SPEED) / LIGHT_SPEED
    assert 1e-10 < relative < 1e-8
    fc = four_component_one_electron(systems[0][1])
    assert fc.light_speed == lib.param.LIGHT_SPEED


# --- 3. The exact path, and what the metric projection does to it -------------------------

@pytest.mark.parametrize("label", ["Ne", "H2O"])
def test_exact_decoupling_reproduces_pyscf_where_nothing_is_dropped(systems, label):
    """The formula, the block conventions and the contraction, checked together.

    Tolerance 1e-11 relative against a measured 4e-13: these systems are well conditioned, the
    metric projection removes nothing, and the two implementations are then doing the same
    arithmetic in a different order. Physically meaningful would be 1e-8 Eh; this is
    three orders tighter because it is a *consistency* check, not an accuracy one.
    """
    mol = dict(systems)[label]
    fc, x, h = _exact_x2c(mol)
    ref = _pyscf_hcore(mol)

    assert h.shape == ref.shape == (2 * mol.nao, 2 * mol.nao)
    assert np.max(np.abs(h - ref)) / np.max(np.abs(ref)) < 1e-11
    # Nothing dropped: the canonical orthogonalization keeps the full 4c space.
    n4c = 2 * fc.hcore.n2c
    assert canonical_orth(_metric(fc)).shape[1] == n4c


def _metric(fc):
    n = fc.hcore.n2c
    m = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    m[:n, :n], m[n:, n:] = fc.overlap.ll, fc.overlap.ss
    return m


@pytest.mark.parametrize("label,max_dropped,max_x", [("TiCl3", 20, 20.0), ("TlH", 100, 50.0)])
def test_heavy_systems_differ_by_the_metric_projection_and_stay_conditioned(
        systems, label, max_dropped, max_x):
    """⚠ The assertion is on ``max |X|``, not on the difference from PySCF.

    Where the projection drops directions Kuiva and PySCF *must* differ, and the difference is
    not an error: PySCF removes no linear dependence, so its ``X`` is the contaminated one.
    Asserting a tight agreement here would be asserting the bug. What is asserted instead is
    that the projection did its job — directions were dropped and ``X`` stayed of order 10
    rather than 1e3, which is the diagnostic that the one-electron check provably cannot see
    (:func:`kuiva.x2c.decouple.canonical_orth`).
    """
    mol = dict(systems)[label]
    fc, x, h = _exact_x2c(mol)

    n4c = 2 * fc.hcore.n2c
    dropped = n4c - canonical_orth(_metric(fc)).shape[1]
    assert 0 < dropped <= max_dropped, "expected the projection to act on {}".format(label)
    assert float(np.max(np.abs(x))) < max_x

    # It is still the same operator, to a bounded and recorded degree.
    ref = _pyscf_hcore(mol)
    assert np.max(np.abs(h - ref)) / np.max(np.abs(ref)) < 1e-6


def test_projection_is_what_separates_kuiva_from_pyscf(systems):
    """The attribution itself, on the system where it is cleanest.

    TiCl3 drops 6 of 1008 directions at the shared threshold and agrees to 6.7e-09; loosen the
    threshold so that nothing is dropped and agreement snaps back to 1e-13. That is the whole
    explanation of the discrepancy, demonstrated rather than asserted in prose.
    """
    import scipy.linalg

    mol = dict(systems)["TiCl3"]
    fc = four_component_one_electron(mol)
    ref = _pyscf_hcore(mol)
    m = _metric(fc)
    n = fc.hcore.n2c

    def rel_error_at(threshold):
        d = np.real(np.diag(m))
        norm = 1.0 / np.sqrt(np.where(d > 0.0, d, 1.0))
        val, vec = np.linalg.eigh(norm[:, None] * m * norm[None, :])
        keep = val >= threshold
        xo = norm[:, None] * (vec[:, keep] / np.sqrt(val[keep]))
        e, a = np.linalg.eigh(xo.conj().T @ fc.hcore.assemble() @ xo)
        a = xo @ a
        pos = e > -fc.light_speed**2
        cl, cs = a[:n, pos], a[n:, pos]
        xm = scipy.linalg.lstsq(cl.conj().T, cs.conj().T)[0].conj().T
        from kuiva.x2c.decouple import renormalization
        r = renormalization(fc.overlap.ll, fc.overlap.ll + xm.conj().T @ fc.overlap.ss @ xm)
        h = fc.contract(picture_change(fc.hcore, xm, r))
        return int(val.size - keep.sum()), np.max(np.abs(h - ref)) / np.max(np.abs(ref))

    dropped_shared, err_shared = rel_error_at(METRIC_LINDEP_THRESHOLD)
    dropped_loose, err_loose = rel_error_at(1e-9)
    assert dropped_shared > 0 and dropped_loose == 0
    assert err_loose < 1e-11 < err_shared


# --- The container's own contract ---------------------------------------------------------

def test_blocks_are_time_reversal_even_and_hermitian(systems):
    """Structural, and cheap. A four-component one-electron Hamiltonian is Hermitian, and its
    two-component picture change must be time-reversal even — the property the ingestion relies on when
    it projects and reports the odd residual."""
    for label, mol in systems[:2]:
        fc, _, h = _exact_x2c(mol)
        assert fc.hcore.hermiticity() < 1e-10, label
        assert fc.overlap.hermiticity() < 1e-10, label
        residual, relative = time_reversal_residual(h)
        assert relative < 1e-10, (label, residual, relative)
        # And it decomposes into spin-free + spin-orbit parts without loss.
        a_sf, w = decompose_two_component(h)
        assert np.max(np.abs(two_component_operator(a_sf, w) - h)) / np.max(np.abs(h)) < 1e-10


def test_atom_ranges_partition_the_working_basis(systems):
    """The ranges DLU will slice with. They must tile ``[0, nao)`` exactly — an overlap or a
    gap would put part of an atom in two blocks or none, and neither shows up as a shape
    error."""
    for label, mol in systems:
        fc = four_component_one_electron(mol)
        assert len(fc.atom_ranges) == mol.natm, label
        covered = []
        for p0, p1 in fc.atom_ranges:
            assert p0 < p1
            covered.extend(range(p0, p1))
        assert covered == list(range(fc.nao)), label
        # an atom owns two row ranges, not one contiguous one.
        idx = fc.spin_blocked_indices(0)
        p0, p1 = fc.atom_ranges[0]
        assert idx.size == 2 * (p1 - p0)
        assert np.array_equal(idx[:p1 - p0], np.arange(p0, p1))
        assert np.array_equal(idx[p1 - p0:], np.arange(fc.nao + p0, fc.nao + p1))


def test_uncontract_false_works_in_the_molecular_basis(systems):
    """``uncontract=False`` decouples in the contracted basis: no contraction matrix, and the
    working basis already is the target. Slightly different numbers, same shape — the point is
    that the flag is honoured, not that the two agree."""
    mol = dict(systems)["Ne"]
    fc = four_component_one_electron(mol, uncontract=False)
    assert fc.contraction is None and not fc.decontracted
    assert fc.nao == fc.nao_target == mol.nao
    x, r = decoupling_matrices(fc.hcore, fc.overlap, fc.light_speed)
    h = fc.contract(picture_change(fc.hcore, x, r))
    assert h.shape == (2 * mol.nao, 2 * mol.nao)
    assert np.array_equal(fc.contract(h), h)


def test_ecp_and_cartesian_bases_are_refused():
    """Both refusals are structural: X2C has no meaning with a pseudopotential core, and
    a Cartesian basis would express the result over different functions from the ones it is
    added to. ⚠ The Cartesian refusal must not depend on which angular momenta are present —
    for l <= 1 the two bases coincide and a shape check would pass."""
    from pyscf import gto

    ecp = gto.M(atom="Xe 0 0 0", basis="lanl2dz", ecp="lanl2dz", verbose=0)
    with pytest.raises(NotImplementedError, match="all-electron"):
        four_component_one_electron(ecp)

    cart = gto.M(atom="Ne 0 0 0", basis="x2c-SVPall-2c", cart=True, verbose=0)
    with pytest.raises(NotImplementedError, match="spherical"):
        four_component_one_electron(cart)
