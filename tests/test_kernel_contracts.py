"""⚠ The kernel-portability contract, enforced from the sources.

C++ portability is a **binding design constraint** on every performance-critical CI kernel:
no decision may block a later port, and when the port happens it must be mechanical — a new
implementation behind an unchanged contract, touching no caller and no test. Whether to spend
that effort is a separate, measured question; this file is about keeping the
option alive.

The realistic failure mode is not a decision to block the port. It is a convenience creeping
into a kernel one commit at a time — a dict here, a log line there, a global read — each
harmless on its own and invisible to every numerical test, because none of it changes an
answer. So the requirements are checked mechanically rather than remembered:

===  ==============================================================================
B1   plain arrays and scalars across the boundary — no dataclass, dict,
     ``Determinants``, logger, timer or budget object in a signature or a body
B2   no hash-based addressing (complete-CAS addressing is combinatorial rank)
B3   flat layout, never ragged; the excitation map is rectangular
B4   fixed dtypes, asserted at the boundary; never dtype-polymorphic output
B5   memory layout is part of the contract, asserted, not inherited
B6   caller-provided output buffer; input and output never alias ambiguously
B7   blocking is a parameter — never a config file, a budget or any global
B8   no logging, timing, resource check or exception raising inside the kernel loop
B9   Python callbacks only at the outermost loop
B10  any reduction whose order a threaded port would change is named in the docstring
===  ==============================================================================

The precedent is the import-direction tests and the ``stage_under_test`` structural
test: a boundary nobody checks is a boundary that has already been crossed.
"""
import ast
import inspect
import textwrap

import numpy as np
import pytest

import kuiva.ci.sigma        # noqa: F401  (registers the sigma kernels)
import kuiva.ci.strings      # noqa: F401  (registers the addressing + connection kernels)
import kuiva.dmrg.block      # noqa: F401  (registers the tensor-network pair-GEMM driver)
import kuiva.mcscf.casci     # noqa: F401  (registers the transition-density kernel)
import kuiva.rdm.rdm         # noqa: F401  (registers the RDM accumulation kernel)
import kuiva.util.native     # noqa: F401  (registers the probe; the KUIVA_KERNELS gate
#                                           itself fires inside kernels.specs() below, so
#                                           the walk sees the native specs when built)
from kuiva.ci import kernels
from kuiva.ci.strings import CASSpace, binomial_table, cas_dimension

#: Every performance-critical kernel that exists yet — the CASSCF plan's, the
#: tensor-network block-pair GEMM driver every network contraction reduces to, and the
#: determinant connection scan of the cheap CI (the first port candidate). A kernel
#: missing from the registry is a kernel nothing below can check, so the list is asserted
#: rather than derived.
REQUIRED_KERNELS = ("cas_rank", "cas_unrank", "excitation_map",
                    "sigma_gather_f", "sigma_gather_out", "rdm_accumulate",
                    "transition_density", "block_pair_gemm", "connections_scan",
                    "sparse_pair_dot")

#: Kernels that take an explicit thread budget (B7 applied to threads): the count comes in as an argument, it is never a global or an environment read.
THREADED_KERNELS = ("connections_scan", "block_pair_gemm", "sparse_pair_dot")

#: Only these may appear as a parameter annotation (B1). Compared as *strings*: the kernel
#: modules use ``from __future__ import annotations``, so annotations arrive unevaluated —
#: which is fine here, since what is being checked is what the signature declares.
ALLOWED_ANNOTATIONS = ("np.ndarray", "ndarray", "numpy.ndarray", "int", "float", "complex",
                       "bool")

#: Names whose appearance anywhere in a kernel body is a port blocker (B1/B7/B8): logging,
#: timing, the resource budget, the dispatch registry itself, and stdout.
FORBIDDEN_NAMES = ("log", "logger", "logging", "res", "resources", "timer", "timed",
                   "kernels", "get_logger", "print", "open", "warnings")

#: Hash-based or ragged containers (B2/B3).
FORBIDDEN_CALLS = ("dict", "set", "defaultdict", "Counter", "OrderedDict", "list",
                   "append", "extend")


def _kernel_source(impl):
    return textwrap.dedent(inspect.getsource(impl))


def _numpy_impl(name):
    """The NumPy reference implementation — what the AST-based checks are *about*.

    A compiled backend has no Python body to walk (its registered wrapper is one return
    statement), so the source-level contract is enforced on the reference implementation,
    which is also the one a port is written from.
    """
    return kernels.resolve(name, kernels.DEFAULT_BACKEND)


def _body_nodes(impl):
    """Every AST node of the kernel's **body**.

    Deliberately excludes the decorator list: ``@kernels.kernel("...")`` is registration, not
    kernel code, and a compiled backend registers itself the same way.
    """
    tree = ast.parse(_kernel_source(impl))
    node = tree.body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    return [child for statement in node.body for child in ast.walk(statement)]


# --- registration ------------------------------------------------------------------------

def test_every_plan_kernel_is_registered():
    missing = [name for name in REQUIRED_KERNELS if name not in kernels.kernel_names()]
    assert not missing, "unregistered kernels: {}".format(missing)


def test_every_kernel_has_the_default_backend():
    for name in REQUIRED_KERNELS:
        assert kernels.DEFAULT_BACKEND in kernels.backends_for(name)


def test_registering_a_second_implementation_of_the_same_name_is_refused():
    """Silent shadowing would make ``resolve`` return something nobody asked for."""
    kernels.register("test_only_kernel", "numpy", lambda: None)
    with pytest.raises(ValueError, match="already registered"):
        kernels.register("test_only_kernel", "numpy", lambda: None)
    kernels.register("test_only_kernel", "stub", lambda: None)     # a new backend is fine
    assert kernels.backends_for("test_only_kernel") == ("numpy", "stub")


# --- B1: plain arrays and scalars across the boundary ------------------------------------

@pytest.mark.parametrize("spec", kernels.specs(), ids=lambda s: "{}/{}".format(s.name,
                                                                              s.backend))
def test_signatures_carry_only_arrays_and_scalars(spec):
    if spec.name not in REQUIRED_KERNELS:
        pytest.skip("not a plan kernel")
    signature = inspect.signature(spec.impl)
    for name, parameter in signature.parameters.items():
        assert parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD), \
            "{}: {} is *args/**kwargs; a kernel takes a fixed argument list".format(
                spec.name, name)
        assert parameter.annotation in ALLOWED_ANNOTATIONS, \
            "{}: {} is annotated {!r}; only arrays and scalars cross the boundary".format(
                spec.name, name, parameter.annotation)
        assert parameter.default is parameter.empty, \
            "{}: {} has a default; a kernel's caller is explicit".format(spec.name, name)


# --- B2/B3/B7/B8: the body ---------------------------------------------------------------

@pytest.mark.parametrize("name", REQUIRED_KERNELS)
def test_no_global_state_logging_timing_or_budget_in_the_body(name):
    for child in _body_nodes(_numpy_impl(name)):
        if isinstance(child, ast.Name) and child.id in FORBIDDEN_NAMES:
            pytest.fail("{}: reads {!r}; a kernel that logs, times or asks a budget for a "
                        "number is a kernel that cannot be a kernel (B7/B8)"
                        .format(name, child.id))
        if isinstance(child, ast.Attribute) and child.attr in FORBIDDEN_NAMES:
            pytest.fail("{}: touches .{} (B7/B8)".format(name, child.attr))


@pytest.mark.parametrize("name", REQUIRED_KERNELS)
def test_no_hash_based_or_ragged_containers_in_the_body(name):
    for child in _body_nodes(_numpy_impl(name)):
        if isinstance(child, (ast.Dict, ast.Set, ast.DictComp, ast.SetComp, ast.ListComp)):
            pytest.fail("{}: builds a Python container; addressing is combinatorial rank "
                        "and payloads are rectangular arrays (B2/B3)".format(name))
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) \
                and child.func.id in FORBIDDEN_CALLS:
            pytest.fail("{}: calls {}() (B2/B3)".format(name, child.func.id))


@pytest.mark.parametrize("name", REQUIRED_KERNELS)
def test_nothing_raises_inside_a_kernel_loop(name):
    """B8. Entry checks raise; the work loop never does — that is what makes the loop a loop.

    Also B9 by construction: a raise inside the loop is the commonest way a Python-level
    interaction sneaks per element.
    """
    for child in _body_nodes(_numpy_impl(name)):
        if isinstance(child, (ast.For, ast.While)):
            for inner in ast.walk(child):
                if isinstance(inner, ast.Raise):
                    pytest.fail("{}: raises inside a loop (B8)".format(name))


@pytest.mark.parametrize("name", ["excitation_map", "sigma_gather_f", "sigma_gather_out",
                                  "rdm_accumulate", "transition_density"])
def test_blocked_kernels_take_their_block_size_as_a_parameter(name):
    """B7: the number comes in, it is never asked for."""
    assert "block" in inspect.signature(kernels.resolve(name, "numpy")).parameters


@pytest.mark.parametrize("name", THREADED_KERNELS)
def test_threaded_kernels_take_their_thread_count_as_a_parameter(name):
    """B7 applied to threads: an explicit argument on every
    backend — never ``omp_set_num_threads``, never an environment read in the kernel."""
    for backend in kernels.backends_for(name):
        assert "n_threads" in inspect.signature(kernels.resolve(name, backend)).parameters


@pytest.mark.parametrize("name", REQUIRED_KERNELS)
def test_every_kernel_states_its_reduction_order(name):
    """B10, and it is a *documentation* requirement precisely because it is unfalsifiable
    numerically until the port exists — by which time a mystery discrepancy has been paid
    for. The parity tolerance follows from this note: no note means bitwise.
    Checked on **every** backend: a compiled implementation's wrapper carries its own note,
    because its reduction behaviour is its own.
    """
    for backend in kernels.backends_for(name):
        doc = inspect.getdoc(kernels.resolve(name, backend)) or ""
        assert "Reduction order (B10)" in doc, \
            "{}/{}: no B10 reduction-order note in the docstring".format(name, backend)


# --- B4/B5/B6: asserted at the boundary, checked by calling ------------------------------

def _valid_arguments(name):
    """A correct call for every kernel, as a mutable list."""
    n, k = 6, 3
    binom = binomial_table(n, k)
    ndet, n_hole, n_empty = cas_dimension(n, k), cas_dimension(n, k - 1), n - k + 1
    space = CASSpace(n, k)
    if name == "cas_rank":
        return [space.masks.copy(), binom, k, np.empty(ndet, dtype=np.int64)], 0, 3
    if name == "cas_unrank":
        return ([np.arange(ndet, dtype=np.int64), binom, n, k,
                 np.empty(ndet, dtype=np.uint64)], 0, 4)
    if name == "excitation_map":
        return ([space.hole_masks().copy(), binom, n, k, 4,
                 np.empty((n_hole, n_empty), dtype=np.int32),
                 np.empty((n_hole, n_empty), dtype=np.int8),
                 np.empty((n_hole, n_empty), dtype=np.int8),
                 np.empty((ndet, k), dtype=np.int32),
                 np.empty((ndet, k), dtype=np.int8),
                 np.empty((ndet, k), dtype=np.int8)], 0, 5)
    arrays = list(space.excitation_arrays())
    if name == "sigma_gather_f":
        return ([np.ones(ndet, dtype=np.complex128)] + arrays
                + [n, 4, np.empty((ndet, n * n), dtype=np.complex128)], 0, 9)
    if name == "sigma_gather_out":
        return ([np.ones((ndet, n * n), dtype=np.complex128)] + arrays
                + [n, 4, np.empty(ndet, dtype=np.complex128)], 0, 9)
    if name == "rdm_accumulate":
        return ([np.ones((ndet, n * n), dtype=np.complex128),
                 np.ones(ndet, dtype=np.complex128), 0.5, 4,
                 np.zeros((n * n, n * n), dtype=np.complex128),
                 np.zeros((n, n), dtype=np.complex128)], 0, 4)
    if name == "transition_density":
        return ([np.ones((2, ndet), dtype=np.complex128),
                 np.ones((ndet, n * n), dtype=np.complex128), 4,
                 np.zeros((2, n * n), dtype=np.complex128)], 0, 3)
    if name == "block_pair_gemm":
        # two 2x3 A blocks against one 3x2 B block, both accumulating into one output
        return ([np.ones(12, dtype=np.complex128), np.array([0, 6, 12], dtype=np.int64),
                 np.ones(6, dtype=np.complex128), np.array([0, 6], dtype=np.int64),
                 np.array([[0, 0, 0, 0], [1, 0, 0, 1]], dtype=np.int64),
                 np.array([[2, 3, 2], [2, 3, 2]], dtype=np.int64),
                 np.zeros(4, dtype=np.complex128), np.array([0, 4], dtype=np.int64),
                 1], 0, 6)
    if name == "sparse_pair_dot":
        # two 4x3 A blocks (in_size 4, rest_dim 3) against one 2x4 CSR block (out_size 2,
        # 3 entries), both pairs accumulating into one (3, 2) output block
        return ([np.ones(24, dtype=np.complex128), np.array([0, 12, 24], dtype=np.int64),
                 np.array([2.0, 1.0 + 1.0j, 3.0], dtype=np.complex128),
                 np.array([0, 3, 1], dtype=np.int64),
                 np.array([0, 2, 3], dtype=np.int64),
                 np.array([[0, 0, 2]], dtype=np.int64),
                 np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
                 np.array([[4, 3, 2], [4, 3, 2]], dtype=np.int64),
                 np.zeros(6, dtype=np.complex128), np.array([0, 6], dtype=np.int64),
                 1], 0, 8)
    if name == "connections_scan":
        cap = 256                               # ample for the C(6,3) space's 190 pairs
        return ([space.masks.copy(), 0, ndet,
                 np.empty(cap, dtype=np.int64), np.empty(cap, dtype=np.int64),
                 np.empty(cap, dtype=np.int64), np.empty(cap, dtype=np.int64),
                 np.empty(cap, dtype=np.float64),
                 np.empty(cap, dtype=np.int64), np.empty(cap, dtype=np.int64),
                 np.empty((cap, 2), dtype=np.int64), np.empty((cap, 2), dtype=np.int64),
                 np.empty(cap, dtype=np.float64), 1], 0, 3)
    raise AssertionError(name)


@pytest.mark.parametrize("name", REQUIRED_KERNELS)
def test_a_correct_call_succeeds(name):
    """The control: everything below must fail for the reason claimed, not incidentally.

    These behaviour checks loop over :func:`kernels.backends_for`, so a compiled backend
    must assert the same contract with the same exception types and message keywords —
    B4/B5/B6 are boundary *behaviour*, not a property of the reference implementation.
    """
    for backend in kernels.backends_for(name):
        args, _, _ = _valid_arguments(name)
        kernels.resolve(name, backend)(*args)


@pytest.mark.parametrize("name", REQUIRED_KERNELS)
def test_wrong_input_dtype_is_rejected(name):
    """B4. ⚠ NumPy would happily cast and return a plausible answer of the wrong dtype."""
    for backend in kernels.backends_for(name):
        args, input_index, _ = _valid_arguments(name)
        original = args[input_index]
        args[input_index] = np.ascontiguousarray(original.real, dtype=np.float64)
        with pytest.raises(TypeError, match="must be"):
            kernels.resolve(name, backend)(*args)


@pytest.mark.parametrize("name", REQUIRED_KERNELS)
def test_wrong_output_dtype_is_rejected(name):
    """B4, and specifically: the output dtype is fixed, never inferred from the input."""
    for backend in kernels.backends_for(name):
        args, _, output_index = _valid_arguments(name)
        args[output_index] = np.zeros(args[output_index].shape, dtype=np.float32)
        with pytest.raises(TypeError, match="must be"):
            kernels.resolve(name, backend)(*args)


@pytest.mark.parametrize("name", REQUIRED_KERNELS)
def test_non_contiguous_output_is_rejected(name):
    """B5. Memory layout is part of the contract; a compiled backend cannot infer strides."""
    for backend in kernels.backends_for(name):
        args, _, output_index = _valid_arguments(name)
        out = args[output_index]
        padded = np.empty(tuple(s * 2 for s in out.shape), dtype=out.dtype)
        args[output_index] = padded[tuple(slice(None, None, 2) for _ in out.shape)]
        with pytest.raises(ValueError, match="C-contiguous"):
            kernels.resolve(name, backend)(*args)


@pytest.mark.parametrize("name", REQUIRED_KERNELS)
def test_an_output_aliasing_an_input_is_rejected(name):
    """B6. Silent aliasing is undefined behaviour in the compiled version, and here it would
    quietly read half-overwritten input."""
    for backend in kernels.backends_for(name):
        args, input_index, output_index = _valid_arguments(name)
        out = args[output_index]
        source = args[input_index]
        buffer = np.zeros(max(out.size, source.size) * 2, dtype=out.dtype)
        args[output_index] = buffer[:out.size].reshape(out.shape)
        args[input_index] = buffer.view(source.dtype)[:source.size].reshape(source.shape)
        with pytest.raises(ValueError, match="alias"):
            kernels.resolve(name, backend)(*args)


@pytest.mark.parametrize("name", ["excitation_map", "sigma_gather_f", "sigma_gather_out",
                                  "rdm_accumulate", "transition_density"])
def test_a_nonsense_block_size_is_rejected(name):
    args, _, _ = _valid_arguments(name)
    block_index = [i for i, a in enumerate(args) if isinstance(a, int)][-1]
    args[block_index] = 0
    with pytest.raises(ValueError, match="block"):
        kernels.resolve(name)(*args)


@pytest.mark.parametrize("name", THREADED_KERNELS)
def test_a_nonsense_thread_count_is_rejected(name):
    for backend in kernels.backends_for(name):
        args, _, _ = _valid_arguments(name)
        args[-1] = 0                              # n_threads is the last argument on both
        with pytest.raises(ValueError, match="thread"):
            kernels.resolve(name, backend)(*args)


# --- what must stay Python --------------------------------------------------------

def test_the_orchestration_says_it_is_orchestration():
    """⚠ the unprofiled-optimization warning cuts both ways: nobody should port the wrapper by mistake.

    The operator class, the budgeting and the timing are not kernels, and the module says so
    where a reader deciding what to port will look.
    """
    import kuiva.ci.sigma as sigma_module

    doc = sigma_module.__doc__ or ""
    assert "orchestration and stays" in doc
    assert "Reduction order (B10)" in (inspect.getdoc(sigma_module.sigma_gather_out_numpy)
                                       or "")
