"""The DLU (local, atom-blocked) X2C decoupling — Tier 0.

DLU is unusual among approximations in having **exact limits that are cheap to test**, and
those limits are what this file is built on. There is no external reference for a DLU
Hamiltonian and none is needed:

* **A single fragment.** One block covering the whole basis makes the local transformation the
  global one, so DLU must reproduce exact X2C-1e *bitwise*. Measured: **0.0**, both sources.
  This is not a tautology — it exercises the partition machinery, the sub-block slicing and
  the assembly, and any of them getting an index wrong breaks it.
* **Separated fragments.** The exact ``U`` becomes block-diagonal as the off-diagonal
  four-component blocks vanish, so the DLU error must fall with distance. That makes DLU's
  error a *bonding-region* error, and it is why the informative molecular measurement is a
  geometry scan rather than a single point.
* **The non-relativistic limit.** ``X -> 1`` and ``R -> 1`` block by block in this
  normalization, so DLU reduces to ``T + V`` exactly, as exact X2C does.
* **Structure.** Block-diagonal ``X`` and ``R`` preserve Hermiticity and time-reversal
  evenness, because each block does.

⚠ **The measured DLU error is small — 4e-07 (TiCl3) to 2e-06 (HF) relative on the one-electron
Hamiltonian — and that is not yet a statement about anything a user cares about.** These are
matrix-element norms, not state energies or splittings. What DLU costs in physics is a stage-4
question and is deliberately not answered here.
"""
import numpy as np
import pytest

from kuiva.interface.pyscf_bridge import (four_component_one_electron,
                                          isolated_fragment_blocks, molecular_partition)
from kuiva.spinor.expand import decompose_two_component, time_reversal_residual
from kuiva.x2c.decouple import decoupling_matrices, picture_change
from kuiva.x2c.local import (Partition, check_local_blocks, local_block_scales,
                             local_decoupling_matrices, off_block_weight, sub_blocks)


def _mol(atom, **kw):
    from pyscf import gto
    return gto.M(atom=atom, basis=kw.pop("basis", "x2c-SVPall-2c"), spin=kw.pop("spin", None),
                 verbose=0, **kw)


def _exact(fc):
    x, r = decoupling_matrices(fc.hcore, fc.overlap, fc.light_speed)
    return fc.contract(picture_change(fc.hcore, x, r))


def _dlu(mol, fc, source="diagonal", local=None):
    partition = molecular_partition(mol, fc)
    x, r = local_decoupling_matrices(fc.hcore, fc.overlap, fc.light_speed, partition,
                                     source=source, local=local)
    return fc.contract(picture_change(fc.hcore, x, r)), x, partition


def _rel(a, b):
    return float(np.max(np.abs(a - b))) / float(np.max(np.abs(b)))


# --- The exact limits ---------------------------------------------------------------------

@pytest.mark.parametrize("source", ["diagonal", "isolated"])
def test_a_single_atom_is_exact_bitwise(source):
    """⚠ ``==``, not ``approx``. For one fragment DLU *is* the exact transformation — the same
    matrices through the same code path — so anything but an exact match means the partition
    machinery perturbed a calculation it should have left alone."""
    mol = _mol("Ne 0 0 0")
    fc = four_component_one_electron(mol)
    local = isolated_fragment_blocks(mol) if source == "isolated" else None
    h_dlu, x, _ = _dlu(mol, fc, source, local)
    assert np.array_equal(h_dlu, _exact(fc))


def test_the_trivial_partition_is_exact_on_a_molecule():
    """The other half of the same statement: it is the *partition*, not the atom count, that
    makes DLU an approximation. One block over a polyatomic must also be exact."""
    mol = _mol("O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24")
    fc = four_component_one_electron(mol)
    whole = Partition.single(fc.hcore.n2c)
    x, r = local_decoupling_matrices(fc.hcore, fc.overlap, fc.light_speed, whole)
    assert np.array_equal(fc.contract(picture_change(fc.hcore, x, r)), _exact(fc))


def test_the_error_vanishes_as_fragments_separate():
    """DLU's error is a bonding-region error. Both it and the off-diagonal weight of the
    four-component Hamiltonian must collapse together as the atoms are pulled apart — the
    second being the cheap predictor of the first."""
    errors, weights = [], []
    for distance in (1.6, 4.0, 12.0):
        mol = _mol("Ne 0 0 0; Ne 0 0 {}".format(distance))
        fc = four_component_one_electron(mol)
        h_dlu, _, partition = _dlu(mol, fc)
        errors.append(_rel(h_dlu, _exact(fc)))
        weights.append(off_block_weight(fc.hcore.ll, partition))

    assert errors[0] > errors[1] > errors[2]
    assert weights[0] > weights[1] > weights[2]
    # By 12 bohr the two atoms barely see each other and DLU is essentially exact.
    assert errors[-1] < 1e-12
    assert errors[0] > 100 * errors[-1]


def test_non_relativistic_limit_is_the_schrodinger_hamiltonian():
    """As ``c -> inf`` the picture change disappears: ``X -> 1``, ``R -> 1``, and the
    transformation collapses to ``T + V``. DLU has the same limit as exact X2C because each
    block does, so this fails loudly on a mis-assembled block-diagonal ``R``."""
    from kuiva.spinor.expand import spin_block_diagonal

    mol = _mol("O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24")
    fc = four_component_one_electron(mol, light_speed=137.035999084 * 1e4)
    h_dlu, _, _ = _dlu(mol, fc)

    h_nr = spin_block_diagonal(mol.intor_symmetric("int1e_kin")
                               + mol.intor_symmetric("int1e_nuc"))
    assert _rel(h_dlu, h_nr) < 1e-8
    assert _rel(h_dlu, _exact(fc)) < 1e-10


def test_structure_survives_the_approximation():
    """Hermiticity and time-reversal evenness are not "close" under DLU — they are exact,
    because block-diagonal ``X`` and ``R`` cannot mix a Kramers pair across fragments."""
    mol = _mol("Ti 0 0 0; Cl 0 0 2.2; Cl 1.905 0 -1.1; Cl -1.905 0 -1.1")
    fc = four_component_one_electron(mol)
    h_dlu, _, _ = _dlu(mol, fc)

    assert np.max(np.abs(h_dlu - h_dlu.conj().T)) / np.max(np.abs(h_dlu)) < 1e-14
    _, relative = time_reversal_residual(h_dlu)
    assert relative < 1e-12
    a_sf, w = decompose_two_component(h_dlu)
    assert np.max(np.abs(a_sf - a_sf.T)) / np.max(np.abs(a_sf)) < 1e-14
    assert np.max(np.abs(w + np.transpose(w, (0, 2, 1)))) / np.max(np.abs(w)) < 1e-14


# --- The two sources are two approximations, not two spellings ----------------------------

def test_the_two_sources_genuinely_differ():
    """⚠ If this ever passes trivially the parameter is decoration.

    The diagonal block of the molecular Hamiltonian carries every *other* nucleus's attraction
    — 13.1 Eh of 8310 on TiCl3's titanium — so the local decouplings really are different
    problems, and ``X`` differs by 1.4e-03 on a scale of 5.9. That they then give a similar
    *accuracy* is a result, not a tautology.
    """
    mol = _mol("Ti 0 0 0; Cl 0 0 2.2; Cl 1.905 0 -1.1; Cl -1.905 0 -1.1")
    fc = four_component_one_electron(mol)
    local = isolated_fragment_blocks(mol)
    partition = molecular_partition(mol, fc)

    x_diag, _ = local_decoupling_matrices(fc.hcore, fc.overlap, fc.light_speed, partition)
    x_iso, _ = local_decoupling_matrices(fc.hcore, fc.overlap, fc.light_speed, partition,
                                         source="isolated", local=local)
    assert np.max(np.abs(x_diag - x_iso)) > 1e-5

    # ...but the underlying *overlap* must be identical: it depends on the basis alone.
    idx = partition.indices[0]
    assert np.array_equal(sub_blocks(fc.overlap, idx).ll, local[partition.labels[0]][1].ll)


def test_isolated_blocks_are_checked_against_the_molecular_overlap():
    """⚠ The guard is on the overlap, and it has to be: a permuted block stays Hermitian, keeps
    the right magnitude, and is wrong. A shape check cannot see it."""
    mol = _mol("Ti 0 0 0; Cl 0 0 2.2; Cl 1.905 0 -1.1; Cl -1.905 0 -1.1")
    fc = four_component_one_electron(mol)
    partition = molecular_partition(mol, fc)
    local = isolated_fragment_blocks(mol)

    check_local_blocks(fc.overlap, partition, local)          # the real ones pass

    label = partition.labels[0]
    hcore_a, overlap_a = local[label]
    permutation = np.arange(overlap_a.ll.shape[0])
    permutation[[0, 1]] = permutation[[1, 0]]
    grid = np.ix_(permutation, permutation)
    permuted = type(overlap_a)(ll=overlap_a.ll[grid], ls=overlap_a.ls[grid],
                               sl=overlap_a.sl[grid], ss=overlap_a.ss[grid])
    with pytest.raises(ValueError, match="ordering"):
        check_local_blocks(fc.overlap, partition, dict(local, **{label: (hcore_a, permuted)}))

    with pytest.raises(KeyError, match="no isolated-fragment blocks"):
        check_local_blocks(fc.overlap, partition,
                           {k: v for k, v in local.items() if k != label})


def test_isolated_source_refuses_to_run_without_blocks():
    mol = _mol("Ne 0 0 0")
    fc = four_component_one_electron(mol)
    partition = molecular_partition(mol, fc)
    with pytest.raises(ValueError, match="isolated-fragment"):
        local_decoupling_matrices(fc.hcore, fc.overlap, fc.light_speed, partition,
                                  source="isolated")
    with pytest.raises(ValueError, match="unknown local decoupling source"):
        local_decoupling_matrices(fc.hcore, fc.overlap, fc.light_speed, partition,
                                  source="dlh")


# --- The partition contract ---------------------------------------------------------------

def test_a_partition_must_be_an_exact_cover():
    """⚠ The one error here that nothing downstream can see. A repeated index overwrites one
    fragment's decoupling with its neighbour's; a missing one leaves ``X`` and ``R`` zero
    there, making the transformation singular on that direction — and both leave a Hermitian,
    plausible, wrong Hamiltonian."""
    with pytest.raises(ValueError, match="exact cover"):
        Partition(indices=(np.arange(4),), labels=("a",)).validate(6)          # too few
    with pytest.raises(ValueError, match="exact cover"):
        Partition(indices=(np.arange(4), np.arange(2, 6)),
                  labels=("a", "b")).validate(6)                              # too many
    # ⚠ The dangerous one: the right *number* of indices, and still not a cover. A size check
    # passes it, so the duplicate/missing detection is what has to catch it.
    with pytest.raises(ValueError, match="more than once"):
        Partition(indices=(np.arange(4), np.array([2, 3])),
                  labels=("a", "b")).validate(6)
    with pytest.raises(ValueError, match="one label per fragment"):
        Partition(indices=(np.arange(6),), labels=("a", "b"))
    with pytest.raises(ValueError, match="at least one fragment"):
        Partition(indices=(), labels=())
    Partition.single(6).validate(6)                                            # the valid case


def test_partition_matches_the_spin_blocked_layout():
    """A fragment owns one row range in the alpha half and one in the beta half. A
    partition built as if the layout were interleaved would still be an exact cover, so the
    cover check cannot catch this — the block content is what does."""
    mol = _mol("O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24")
    fc = four_component_one_electron(mol)
    partition = molecular_partition(mol, fc)

    partition.validate(fc.hcore.n2c)
    assert len(partition) == mol.natm
    assert partition.dimension == fc.hcore.n2c == 2 * fc.nao
    for ia, idx in enumerate(partition.indices):
        p0, p1 = fc.atom_ranges[ia]
        assert np.array_equal(idx, np.concatenate([np.arange(p0, p1),
                                                   np.arange(fc.nao + p0, fc.nao + p1)]))
    # The overlap sub-block of an atom is that atom's own overlap: block-diagonal in spin,
    # which is only true if the two halves were picked consistently.
    s_a = sub_blocks(fc.overlap, partition.indices[0]).ll
    half = s_a.shape[0] // 2
    assert np.max(np.abs(s_a[:half, half:])) == 0.0
    assert np.array_equal(s_a[:half, :half], s_a[half:, half:])


def test_sizing_functions_are_exact_and_bounded_on_both_sides():
    """A sizing function is **exact** and never pads, so it is pinned against a
    real array's ``nbytes`` from above *and* below — a safety factor that crept in would fail
    this rather than quietly shrinking every budget."""
    from kuiva.util import resources as res
    from kuiva.x2c.decouple import decoupling_memory_gb, exact_decoupling_workspace_gb

    nao = 40
    x = np.zeros((2 * nao, 2 * nao), dtype=np.complex128)
    predicted = decoupling_memory_gb(nao)
    assert predicted == pytest.approx(2 * x.nbytes / 1024**3, rel=1e-12)

    workspace = np.zeros((4 * nao, 4 * nao), dtype=np.complex128)
    assert exact_decoupling_workspace_gb(nao) == pytest.approx(
        4 * workspace.nbytes / 1024**3, rel=1e-12)

    # The claim that makes DLU worth having: the exact workspace dwarfs what the local path
    # pays per fragment. Eight atoms of 40 functions each against one problem of 320.
    assert exact_decoupling_workspace_gb(8 * nao) > 50 * exact_decoupling_workspace_gb(nao)
    assert res.array_gb((2, 2), np.complex128) > 0.0


def test_local_block_scales_are_reported_per_fragment(systems_scale=20.0):
    """⚠ Per fragment, not globally: ``max |X|`` is the conditioning diagnostic that catches a
    near-singular decontracted heavy element, and averaging it over a molecule would let one
    bad heavy atom hide behind a dozen healthy ligands."""
    mol = _mol("Ti 0 0 0; Cl 0 0 2.2; Cl 1.905 0 -1.1; Cl -1.905 0 -1.1")
    fc = four_component_one_electron(mol)
    _, x, partition = _dlu(mol, fc)
    scales = local_block_scales(x, partition)

    assert set(scales) == set(partition.labels)
    assert all(0.0 < value < systems_scale for value in scales.values()), scales
    # Titanium is the heavy centre, so it carries the largest decoupling.
    assert max(scales, key=scales.get).startswith("Ti")
