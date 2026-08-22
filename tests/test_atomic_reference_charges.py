"""The atomic-reference charges: the robust partition that replaced the Loewdin charge.

Tier 0: exactness and invariance properties (sum rule, rotation invariance, scalar/spinor
consistency) plus the refusal and warning behaviours that make the feature safe to hand to a
user. The *robustness* claims (basis stability, ghost immunity, correlated densities) are
measurements, not unit properties — they live in the props validation record with the
generator `tests/generate/atomic_reference_charge_study.py`; what is pinned here is one
committed charge value, so a change in the scheme's numerics cannot land unnoticed.
"""
import numpy as np
import pytest

from kuiva.interface import Molecule
from kuiva.interface import api
from kuiva.interface.pyscf_bridge import run_scalar_x2c
from kuiva.props.population import atomic_reference_charges

TIF3 = [("Ti", (0.0, 0.0, 0.0)), ("F", (1.7796, 0.0, 0.0)),
        ("F", (-0.8898, 1.5412, 0.0)), ("F", (-0.8898, -1.5412, 0.0))]


@pytest.fixture(scope="module")
def tif3_ref():
    mol = Molecule(TIF3, basis="x2c-SVPall-2c", spin=1)
    return api.spinor_reference(mol, screening="none", memory_gb=4.0,
                                atomic_reference=True)


def test_sum_rule_and_signs(tif3_ref):
    q = tif3_ref.atomic_reference_charges(report=False)
    assert q.charge.sum() == pytest.approx(0.0, abs=1e-8)   # exact partition of N
    assert q.charge[0] > 1.5                                # Ti(III) clearly positive
    assert all(x < -0.4 for x in q.charge[1:])              # fluorides clearly negative
    # near-equal fluorides: the single ROHF d electron picks a direction, so the *density*
    # is only approximately D3h and ~5e-6 of genuine inequivalence remains. The partition
    # must not add to it.
    assert q.charge[1] == pytest.approx(q.charge[2], abs=1e-4)
    # the committed pin: the scheme's numerics moved if this does (tolerance is loose
    # against SCF/BLAS noise, tight against any change of partition)
    assert q.charge[0] == pytest.approx(2.2044, abs=2e-3)


def test_rotation_invariance(tif3_ref):
    """The partition must not depend on the molecule's orientation. The atomic reference
    fills shells equally over m (average of configuration), so the occupied tier is a
    rotation-invariant subspace per atom and the charges must reproduce to numerical
    precision — this is the degenerate-group discipline in its population instance."""
    c, s = np.cos(0.7), np.sin(0.7)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ \
        np.array([[1.0, 0.0, 0.0], [0.0, np.cos(0.4), -np.sin(0.4)],
                  [0.0, np.sin(0.4), np.cos(0.4)]])
    atoms = [(sym, tuple(rot @ np.asarray(xyz))) for sym, xyz in TIF3]
    ref2 = api.spinor_reference(Molecule(atoms, basis="x2c-SVPall-2c", spin=1),
                                screening="none", memory_gb=4.0, atomic_reference=True)
    q1 = tif3_ref.atomic_reference_charges(report=False).charge
    q2 = ref2.atomic_reference_charges(report=False).charge
    assert np.allclose(q1, q2, atol=5e-6)


def test_scalar_and_spinor_densities_agree(tif3_ref):
    """The spin-traced spinor-guess density and the scalar SCF density are the same
    physical density, so the charges must match through either entry."""
    data = tif3_ref.data
    (c_scalar,) = data.mo_sets()
    q_spinor = tif3_ref.atomic_reference_charges(report=False).charge
    q_scalar = atomic_reference_charges(c_scalar, data.s_ao, data.ao_layout,
                                        data.atomic_reference,
                                        occupation=data.mo_occ).charge
    assert np.allclose(q_spinor, q_scalar, atol=1e-8)


def test_missing_reference_names_the_knob():
    mol = Molecule([("Ne", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c")
    d = run_scalar_x2c(mol, screening="none", memory_gb=4.0)   # no atomic_reference
    with pytest.raises(ValueError, match="atomic_reference=True"):
        atomic_reference_charges(d.mo_coeff, d.s_ao, d.ao_layout, d.atomic_reference,
                                 occupation=d.mo_occ)


def test_non_default_configuration_warns(kuiva_caplog):
    """User decision: an overridden reference state is honoured, and the report warns that
    its charges are not comparable with default-reference ones."""
    mol = Molecule(TIF3, basis="x2c-SVPall-2c", spin=1)
    d = run_scalar_x2c(mol, screening="none", memory_gb=4.0, with_soc=False,
                       atomic_reference=True,
                       screening_options={"configuration": {"Ti": "+3"}})
    assert d.atomic_reference.any_non_default
    assert "3+" in d.atomic_reference["Ti"].configuration or \
           "+3" in d.atomic_reference["Ti"].configuration
    q = atomic_reference_charges(d.mo_coeff, d.s_ao, d.ao_layout, d.atomic_reference,
                                 occupation=d.mo_occ, report=True)
    assert any("not comparable" in r.getMessage().lower() or
               "NOT comparable" in r.getMessage() for r in kuiva_caplog.records)
    # the override changed the reference, so the charge moves relative to the default one
    assert q.charge[0] != pytest.approx(2.2044, abs=1e-4)
