"""Stage 5: compiling fermionic operators into circuits .

What has to be airtight here, and why the reference is ``expm``
----------------------------------------------------------------
:mod:`kuiva.qc.fermionic` is the layer every Stage-C ansatz and the SKQD circuit source are
built from, and its failure mode is the worst one: a wrong basis-change sign, a wrong
Jordan-Wigner string, a wrong conjugation order all produce a **unitary** circuit that
conserves particle number, normalizes, and gives a plausible energy. Structural checks cannot
see any of them.

So every claim below is checked against something that does not share the implementation:

* Pauli exponentials against ``scipy.linalg.expm`` of the dense Pauli matrix, including the
  global phase, for **every** string on three qubits;
* the orbital-rotation compilation against the **Slater-determinant minors**
  ``det(u[occ_out, occ_in])`` — a formula from a different branch of the theory entirely;
* the parameter-shift gradient against a central finite difference of the same objective.

⚠ **And one guard that exists to prove the others can fail** (a guard that cannot fail proves nothing): the ``Y``
basis-change sign is deliberately flipped and the comparison must then break by a wide margin.
Without it, "the exponentials agree" would be consistent with a test blind to the one piece of
bookkeeping here that cannot be argued from first principles.

⚠ **The amplitudes are complex on purpose**, and the difference is the Stage-C research
content rather than extra coverage: with a real amplitude an excitation generator's Pauli
strings all commute — which is why every published UCC circuit is a plain product — and with a
complex one, forced on Kuiva by spin-orbit coupling, they do not.
"""
import numpy as np
import pytest
from scipy.linalg import expm

from kuiva.ci.strings import CASSpace, popcount
from kuiva.qc import fermionic as fx
from kuiva.qc.circuits import CircuitSpec, apply_circuit
from kuiva.qc.fermionic import (CircuitBuilder, double_excitation_generator, excitation_circuit,
                                exponential_circuit, givens_decomposition, jastrow_generator,
                                one_body_generator, orbital_rotation_circuit,
                                parameter_shift_gradient, pauli_exponential_circuit,
                                single_excitation_generator, su2_mode_rotation, trotter_circuit)
from kuiva.qc.mapping import PauliSum, jordan_wigner

#: Machine precision. Both sides evaluate the same unitary, so anything above rounding is a
#: defect rather than a method difference.
EXACT = 1e-12


def _unitary(circuit: CircuitSpec) -> np.ndarray:
    """The circuit's matrix, column by column through the reference semantics."""
    dim = 1 << circuit.n_qubit
    eye = np.eye(dim, dtype=np.complex128)
    return np.column_stack([apply_circuit(circuit, eye[:, j]) for j in range(dim)])


def _realized(compiled) -> np.ndarray:
    """The operator a :class:`CompiledCircuit` was asked to realize, phase included."""
    return np.exp(1j * compiled.global_phase) * _unitary(compiled.circuit)


def _one_string(x: int, z: int, n: int) -> np.ndarray:
    return PauliSum(np.array([1.0]), np.array([x], np.uint64), np.array([z], np.uint64),
                    n).to_dense()


# --- the one primitive --------------------------------------------------------------------

@pytest.mark.parametrize("theta", [0.0, 0.7331, -2.4, np.pi])
def test_every_pauli_exponential_on_three_qubits_matches_expm(theta):
    """All 64 strings, including the identity, whose exponential is a pure global phase."""
    n = 3
    worst = 0.0
    for x in range(1 << n):
        for z in range(1 << n):
            compiled = pauli_exponential_circuit(x, z, theta, n)
            reference = expm(-0.5j * theta * _one_string(x, z, n))
            worst = max(worst, float(np.abs(reference - _realized(compiled)).max()))
    assert worst < EXACT


def test_the_identity_string_becomes_a_phase_and_no_gate():
    compiled = pauli_exponential_circuit(0, 0, 1.3, 2)
    assert compiled.circuit.n_gates == 0
    assert compiled.global_phase == pytest.approx(-0.65)


def test_the_y_basis_change_is_load_bearing(monkeypatch):
    """⚠ A guard that cannot fail proves nothing, so break it and watch it fail.

    ``Rx(pi/2) Z Rx(pi/2)^dag = -Y``, so the rotation taking ``Y`` into ``Z`` is ``rx(+pi/2)``
    first and ``rx(-pi/2)`` after. Flipping that sign leaves a unitary circuit of the right
    depth acting on the right qubits, and every ``Y``-carrying term of every ansatz gets the
    wrong sign.
    """
    n, theta = 2, 0.9
    y_strings = [(x, z) for x in range(1 << n) for z in range(1 << n) if x & z]
    original = fx.pauli_exponential

    def flipped(builder, x, z, angle, **kwargs):
        # The same construction with the Y rotation reversed — the plausible wrong version.
        support = [k for k in range(builder.n_qubit) if (x >> k) & 1 or (z >> k) & 1]
        if not support:
            return original(builder, x, z, angle, **kwargs)
        for k in support:
            if (x >> k) & 1 and (z >> k) & 1:
                builder.add("rx", (k,), (-0.5 * np.pi,))
            elif (x >> k) & 1:
                builder.add("h", (k,))
        for a, b in zip(support[:-1], support[1:]):
            builder.add("cx", (a, b))
        builder.add("rz", (support[-1],), (angle,))
        for a, b in zip(support[-2::-1], support[:0:-1]):
            builder.add("cx", (a, b))
        for k in support:
            if (x >> k) & 1 and (z >> k) & 1:
                builder.add("rx", (k,), (0.5 * np.pi,))
            elif (x >> k) & 1:
                builder.add("h", (k,))
        return builder

    worst = 0.0
    for x, z in y_strings:
        broken = flipped(CircuitBuilder(n), x, z, theta).result()
        reference = expm(-0.5j * theta * _one_string(x, z, n))
        worst = max(worst, float(np.abs(reference - _realized(broken)).max()))
    assert worst > 1e-3, "flipping the Y basis change must break the comparison"


# --- exactness of an ansatz generator's exponential ------------------------------------------

@pytest.mark.parametrize("modes", [(0, 1), (0, 3), (2, 5)])
def test_a_real_single_excitation_generator_has_two_commuting_strings(modes):
    generator = single_excitation_generator(modes[0], modes[1], 1.0, 6)
    assert generator.n_terms == 2
    assert generator.all_commute()


@pytest.mark.parametrize("modes", [(0, 1, 2, 3), (0, 2, 3, 5)])
def test_a_real_double_excitation_generator_has_eight_commuting_strings(modes):
    generator = double_excitation_generator(*modes, 1.0, 6)
    assert generator.n_terms == 8
    assert generator.all_commute()


@pytest.mark.parametrize("modes", [(0, 1), (2, 5)])
def test_a_complex_single_excitation_generator_does_not_commute_with_itself(modes):
    """⚠ The Stage-C finding, pinned as a test rather than left in a docstring.

    A complex amplitude doubles the Pauli count and the two halves anticommute where they
    differ, so the plain product every published UCC implementation uses stops being the
    exponential it is called. The exact route is the phase conjugation of
    :func:`excitation_circuit`, checked below.
    """
    generator = single_excitation_generator(modes[0], modes[1], 0.3 + 0.4j, 6)
    assert generator.n_terms == 4
    assert not generator.all_commute()


def test_a_complex_double_excitation_generator_does_not_commute_with_itself():
    generator = double_excitation_generator(0, 1, 2, 3, 0.3 + 0.4j, 6)
    assert generator.n_terms == 16
    assert not generator.all_commute()


def test_exponential_circuit_refuses_to_call_a_trotter_product_exact():
    generator = single_excitation_generator(0, 1, 0.3 + 0.4j, 4)
    with pytest.raises(ValueError, match="do not all commute"):
        exponential_circuit(CircuitBuilder(4), generator, 1.0, require_exact=True)


@pytest.mark.parametrize("magnitude,phase", [(0.4, 0.0), (0.4, 1.1), (-0.9, -2.3)])
@pytest.mark.parametrize("creations,annihilations",
                         [((0,), (3,)), ((0, 1), (2, 4)), ((1, 3), (0, 4))])
def test_a_complex_excitation_is_compiled_exactly(magnitude, phase, creations, annihilations):
    """The whole point of the phase-gate conjugation: exact, complex amplitude included."""
    n = 5
    amplitude = magnitude * np.exp(1j * phase)
    if len(creations) == 1:
        generator = single_excitation_generator(creations[0], annihilations[0], amplitude, n)
    else:
        generator = double_excitation_generator(creations[0], creations[1], annihilations[0],
                                                annihilations[1], amplitude, n)
    compiled = excitation_circuit(CircuitBuilder(n), creations, annihilations,
                                  magnitude, phase).result()
    assert np.abs(expm(-1j * generator.to_dense()) - _realized(compiled)).max() < 1e-11


def test_an_excitation_that_creates_only_what_it_annihilates_is_refused():
    with pytest.raises(ValueError, match="number operator in disguise"):
        excitation_circuit(CircuitBuilder(4), (0, 1), (1, 0), 0.3)


def test_a_jastrow_generator_is_diagonal_and_exact():
    coupling = np.zeros((4, 4))
    coupling[0, 1] = coupling[1, 0] = 0.3
    coupling[2, 2] = -0.7
    generator = jastrow_generator(coupling)
    assert generator.all_commute()
    assert not np.any(generator.x_masks)               # diagonal: Z factors only
    compiled = exponential_circuit(CircuitBuilder(4), generator, 1.0,
                                   require_exact=True).result()
    assert np.abs(expm(-1j * generator.to_dense()) - _realized(compiled)).max() < EXACT


def test_a_non_antihermitian_one_body_generator_is_refused():
    with pytest.raises(ValueError, match="anti-Hermitian"):
        one_body_generator(np.eye(3, dtype=complex))


# --- exact compilation of an orbital rotation -------------------------------------------------

def _random_unitary(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kappa = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return expm(kappa - kappa.conj().T)


@pytest.mark.parametrize("n", [2, 4, 5])
def test_the_givens_decomposition_reconstructs_the_unitary(n):
    u = _random_unitary(n, seed=n)
    rotations, phases = givens_decomposition(u)
    rebuilt = np.diag(np.exp(1j * phases))
    for rotation in reversed(rotations):
        block = np.eye(n, dtype=np.complex128)
        block[rotation.mode:rotation.mode + 2, rotation.mode:rotation.mode + 2] = \
            rotation.matrix()
        rebuilt = block.conj().T @ rebuilt
    assert np.abs(rebuilt - u).max() < 1e-12


def test_a_non_unitary_mode_matrix_is_refused():
    with pytest.raises(ValueError, match="not unitary"):
        givens_decomposition(np.array([[1.0, 2.0], [0.0, 1.0]], dtype=complex))


@pytest.mark.parametrize("n,n_elec", [(4, 1), (4, 2), (5, 3)])
def test_the_orbital_rotation_circuit_reproduces_the_slater_determinant_minors(n, n_elec):
    """⚠ The reference is a formula from a different branch of the theory, not another circuit.

    ``U(u) a_p^dag U^dag = sum_q u_qp a_q^dag`` means the amplitude of ``U(u)|ref>`` on ``|I>``
    is ``det(u[occ(I), occ(ref)])``. Nothing about that derivation passes through a Givens
    network, a Jordan-Wigner string or a gate matrix, so the comparison can fail.
    """
    u = _random_unitary(n, seed=10 + n)
    compiled = orbital_rotation_circuit(u)
    reference = (1 << n_elec) - 1
    occupied_in = list(range(n_elec))
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[reference] = 1.0
    out = np.exp(1j * compiled.global_phase) * apply_circuit(compiled.circuit, psi)
    dets = CASSpace(n, n_elec).determinants()
    for mask in dets.masks.tolist():
        occupied_out = [k for k in range(n) if (int(mask) >> k) & 1]
        minor = np.linalg.det(u[np.ix_(occupied_out, occupied_in)])
        assert out[int(mask)] == pytest.approx(minor, abs=1e-12)
    # And nothing at all outside the particle-number sector.
    leaked = np.abs(out[popcount(np.arange(1 << n, dtype=np.uint64)) != n_elec]).max()
    assert leaked < EXACT


@pytest.mark.parametrize("n", [3, 4])
def test_the_recorded_global_phase_is_half_the_argument_of_the_determinant(n):
    """A cross-check on the phase bookkeeping that needs no circuit at all: every Givens factor
    is in ``SU(2)``, so the whole determinant sits in the final diagonal."""
    u = _random_unitary(n, seed=20 + n)
    compiled = orbital_rotation_circuit(u)
    assert np.exp(2j * compiled.global_phase) == pytest.approx(np.linalg.det(u), abs=1e-12)


def test_su2_mode_rotation_conserves_particle_number_exactly():
    n = 4
    compiled = su2_mode_rotation(CircuitBuilder(n), 1, 0.7, 0.3, -0.2).result()
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0b0110] = 1.0
    out = apply_circuit(compiled.circuit, psi)
    assert np.abs(out[popcount(np.arange(1 << n, dtype=np.uint64)) != 2]).max() < EXACT


# --- Trotter ----------------------------------------------------------------------------------

def test_trotter_error_falls_as_one_over_steps():
    """First order, so halving the slice halves the error. Stated, measured, not tuned away."""
    import sys
    sys.path.insert(0, "tests")
    from test_ci_strings import random_spinor_integrals

    n, time = 4, 0.2
    h, eri = random_spinor_integrals(n, seed=3)
    operator = jordan_wigner(h, eri)
    reference = expm(-1j * operator.to_dense() * time)
    errors = []
    for steps in (1, 2, 4, 8):
        compiled = trotter_circuit(operator, time, steps=steps)
        errors.append(float(np.abs(_realized(compiled) - reference).max()))
    assert errors[0] > errors[-1]
    # Each doubling of the slice count cuts the error by close to two.
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert 1.6 < coarse / fine < 2.6


def test_trotter_with_a_reference_prepares_the_occupation_mask():
    operator = jordan_wigner(np.zeros((4, 4), dtype=complex),
                             np.zeros((4,) * 4, dtype=complex))
    compiled = trotter_circuit(operator, 0.5, reference=0b0101)
    psi = np.zeros(16, dtype=np.complex128)
    psi[0] = 1.0
    assert np.abs(apply_circuit(compiled.circuit, psi)[0b0101]) == pytest.approx(1.0)


# --- gradients ----------------------------------------------------------------------------------

def test_the_parameter_shift_gradient_is_exact():
    """Against a central finite difference of the *same* objective — so the two sides differ
    only in how the derivative is taken, which is the thing under test."""
    n = 4
    operator = jordan_wigner(*_small_integrals(n))
    dense = operator.to_dense()

    def energy(circuit):
        # From |0...0>: the circuit prepares its own reference, exactly as an ansatz does.
        psi = np.zeros(1 << n, dtype=np.complex128)
        psi[0] = 1.0
        state = apply_circuit(circuit, psi)
        return float(np.real(np.vdot(state, dense @ state)))

    def build(params):
        builder = CircuitBuilder(n).prepare(0b0011)
        excitation_circuit(builder, (2,), (0,), params[0], params[1],
                           magnitude_index=0, phase_index=1)
        excitation_circuit(builder, (2, 3), (1, 0), params[2], params[3],
                           magnitude_index=2, phase_index=3)
        return builder.result(4)

    x0 = np.array([0.31, 0.7, -0.22, 1.4])
    analytic = parameter_shift_gradient(build(x0), energy)
    numeric = np.empty_like(analytic)
    for k in range(x0.size):
        step = np.zeros_like(x0)
        step[k] = 1e-5
        numeric[k] = (energy(build(x0 + step).circuit)
                      - energy(build(x0 - step).circuit)) / 2e-5
    assert np.abs(analytic - numeric).max() < 1e-8
    assert np.linalg.norm(analytic) > 1e-3, "a zero gradient would make this vacuous"


def _small_integrals(n):
    rng = np.random.default_rng(11)
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = h + h.conj().T
    eri = rng.normal(size=(n,) * 4) + 1j * rng.normal(size=(n,) * 4)
    eri = eri + eri.transpose(2, 3, 0, 1)
    eri = 0.1 * (eri + eri.conj().transpose(1, 0, 3, 2))
    return h, eri
