"""The integer-spin case, end to end: a non-Kramers doublet on a real molecule.

⚠ **Every other committed reference in this project is odd-electron** — Ce(3+), Yb(3+),
Dy(3+), Ti(3+), B, Bi, Tl — and that is not a symmetric half of the target space. An
integer-spin ion has no Kramers protection: its ground "doublet" is degenerate only if some
spatial symmetry says so, its transverse moment is zero rather than merely small, and when the
symmetry is broken the pair splits by a tunnelling gap, which is the quantity a Tb or Ho
single-molecule magnet is *about*. The machinery for all of that landed in v0.18.0 and until
now was exercised only against models built in a test file.

The system is ``fecl2`` from ``tests/generate/systems.py``: linear FeCl2, Fe(2+) d^6, a
5-Delta ground term whose spin-orbit ladder is |Omega| = 4, 3, 2, 1, 0. The lowest member is
the non-Kramers doublet, and it has an **analytic** g value — 2 (Lambda + g_e Sigma) = 12.009
for a pure |+-2, +-2> — which is what makes this a validation system rather than a
demonstration. Nothing external is needed for the target.

⚠ Marked slow: the run needs a four-component atomic solve for Fe and for Cl, and the default
suite may not depend on a warm cache.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "generate"))
import systems as sysdef                                              # noqa: E402

import kuiva                                                          # noqa: E402
from kuiva.interface import Molecule                                  # noqa: E402

pytestmark = pytest.mark.slow

#: g_z of a pure |Lambda = 2, Sigma = 2> non-Kramers doublet: ``2 (Lambda + g_e Sigma)``.
#: ⚠ Theory, not a previous run of this code — which is the whole point of choosing a system
#: whose ground state has a closed form. Covalency and the ligand field mix in the other
#: |Omega| = 4 components, so the calculated value sits slightly below it.
G_Z_FREE_ION = 2.0 * (2.0 + 2.0023193 * 2.0)


@pytest.fixture(scope="module")
def spectrum(tmp_path_factory):
    """The SOC spectrum and its phase-invariant reductions, computed once."""
    system = sysdef.get("fecl2")
    mol = Molecule(system.atoms, basis=system.basis, charge=system.charge, spin=system.spin)
    scf = kuiva.ScalarSCF(mol, memory_gb=8.0).run()
    ref = kuiva.Reference(scf).run()
    cas = kuiva.CASSCF(ref, character=("Fe", "d"), n_active=2 * system.ncas,
                       n_active_elec=system.nelecas, n_states=system.soc_states,
                       mode="second-order", report=False).run()
    dump = kuiva.PropertyDump(cas, str(tmp_path_factory.mktemp("nk") / "fecl2.out"),
                              report=False).run()
    return dump.matrices


def test_the_ground_block_is_a_doublet_with_the_analytic_axial_g(spectrum):
    """⚠ The one number this system exists for. A ligand field and covalency can only reduce
    ``g_z`` from the free-ion limit, so the band is one-sided and stated as such: agreement to
    better than 1% of an analytic value, from a calculation that knows nothing about it."""
    blocks = spectrum.analyse()
    ground = blocks[0]
    assert ground.size == 2
    g = np.asarray(ground.g_values)
    assert g[2] == pytest.approx(G_Z_FREE_ION, rel=0.01)
    assert g[2] < G_Z_FREE_ION + 1e-6          # covalency reduces it; it cannot exceed it


def test_the_transverse_g_is_zero_because_the_doublet_is_not_a_kramers_one(spectrum):
    """⚠ **The statement no odd-electron system in this suite can make.** A Kramers doublet
    always carries a transverse moment; this pair cannot, because ``mu_x`` connects
    |Omega| = 4 to |Omega| = 3 and there is no such matrix element inside the block. Zero here
    is a symmetry statement, so it is asserted at 1e-6 rather than at a physical tolerance."""
    ground = spectrum.analyse()[0]
    gx, gy = np.asarray(ground.g_values)[:2]
    assert abs(gx) < 1e-6 and abs(gy) < 1e-6


def test_the_spin_orbit_ladder_is_the_one_a_5_delta_term_has(spectrum):
    """The whole ground term, as a pattern rather than as numbers: five doublets of the
    |Omega| ladder below the next term, with the g values falling 12, 8, 4 as |Omega| does.
    A pattern check is what survives the basis and the ligand field being approximate."""
    blocks = spectrum.analyse()
    ladder = [b for b in blocks if b.energy_cm < 2000.0]
    assert [b.size for b in ladder] == [2, 2, 2, 2, 2]
    g_z = [np.asarray(b.g_values)[2] for b in ladder]
    np.testing.assert_allclose(g_z[:3], [G_Z_FREE_ION, 8.0, 4.0], rtol=0.02)
    assert blocks[len(ladder)].energy_cm > 2000.0          # and the term is complete


def test_every_block_of_an_even_electron_system_may_be_a_singlet(spectrum):
    """⚠ The absence of Kramers protection, asserted where it is visible: an odd-electron
    spectrum is doubly degenerate throughout, and this one is not — the term above the ladder
    contains size-1 blocks, which is the case ``multiplet_g_values`` returns ``()`` for
    rather than a silent ``(0, 0, 0)``."""
    blocks = spectrum.analyse()
    assert any(b.size == 1 for b in blocks)
    for b in blocks:
        if b.size == 1:
            assert b.g_values == ()                        # no moment, and it says so


def test_breaking_the_axis_splits_the_pair_into_a_tunnelling_gap(tmp_path):
    """⚠ **The case an integer-spin SMM is about**, and the only end-to-end exercise of the
    pseudo-doublet path: the ground pair's degeneracy is the axial symmetry's doing, not time
    reversal's, so bending the molecule turns it into two singlets separated by a tunnelling
    gap. Each singlet alone carries no moment; the pair carries ``g_z``, and grouping them is
    an explicit request because whether two near-singlets are one doublet is physics their
    energies cannot settle.

    ⚠ **Bent to 90 degrees, and the angle is the measurement rather than a taste.** For an
    |Omega| = 4 ground state the transverse field has to act at high order, so the gap is
    minuscule until the distortion is severe: measured 3.7e-05 cm^-1 at 150 degrees and
    9.0e-03 at 120, both inside or below the 1e-8..1e-6 Eh band this project reserves for
    *numerical* splitting, where no claim can be made about them. At 90 degrees it is
    0.32 cm^-1 — above that band and still 400 times smaller than the next block, which is
    exactly the separation of scales that makes a tunnelling gap interesting.
    """
    import math

    r, half = 2.151, math.radians(45.0)                    # 90 degrees, from 180
    bent = Molecule([("Fe", (0.0, 0.0, 0.0)),
                     ("Cl", (r * math.sin(half), 0.0, r * math.cos(half))),
                     ("Cl", (-r * math.sin(half), 0.0, r * math.cos(half)))],
                    basis="x2c-SVPall-2c", spin=4)
    scf = kuiva.ScalarSCF(bent, memory_gb=8.0).run()
    ref = kuiva.Reference(scf).run()
    cas = kuiva.CASSCF(ref, character=("Fe", "d"), n_active=10, n_active_elec=6, n_states=25,
                       mode="second-order", report=False).run()
    dump = kuiva.PropertyDump(cas, str(tmp_path / "bent.out"), report=False).run()

    # ⚠ tol_cm below the gap on purpose: at the default 1 cm^-1 these two ARE one degenerate
    # block, which is the honest reading of a pair this close together and not what is under
    # test here.
    singlets = dump.matrices.analyse(tol_cm=1e-4)
    assert singlets[0].size == 1 and singlets[1].size == 1
    gap = singlets[1].energy_cm - singlets[0].energy_cm
    assert 0.05 < gap < 5.0                                # above the numerical band, tiny
    assert singlets[2].energy_cm > 50.0 * gap              # and far below the ligand field
    assert singlets[0].g_values == () and singlets[1].g_values == ()

    paired = dump.matrices.analyse(tol_cm=1e-4, pseudo_doublet_tol_cm=10.0 * gap)[0]
    assert paired.non_kramers and paired.size == 2
    assert paired.tunnelling_gap_cm == pytest.approx(gap, rel=1e-6)
    assert paired.g_z > 8.0                                # still a strongly axial pair
    assert paired.g_transverse_residual < 1e-3             # and still non-Kramers
