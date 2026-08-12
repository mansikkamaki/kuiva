"""The compiled kernel backend: gate semantics, provenance, and per-kernel parity.

What is deliberately *not* here: functional coverage of the kernels themselves —
``kernels.backends_for`` parametrization runs the whole existing suite against the native
backend the moment it registers (that is the main acceptance machinery), and
``tests/test_kernel_contracts.py`` asserts the boundary behaviour on every backend. This
file holds only what nothing else states: the ``KUIVA_KERNELS`` gate's three modes, the
provenance tokens, and the **explicit** parity claims (bitwise, or 1e-13 with the
B10 note) on contract-test shapes and one production-shaped instance per kernel.

On a build-less clone every native-only test **skips** naming the build script — absence of
the build is a normal state of the default path, not a separate suite (no ``qc``-style
marker: absence of the build is normal, a separate interpreter is not).
"""
import numpy as np
import pytest

import kuiva.dmrg.block      # noqa: F401  (registers the numpy block_pair_gemm)
from kuiva.ci import kernels
from kuiva.ci.strings import Determinants, connections
from kuiva.util import native

NEEDS_BUILD = "native extension not built (bash scripts/bootstrap/95_native.sh)"


def _native_or_skip():
    native.activate()
    if not native.available():
        pytest.skip(NEEDS_BUILD)


@pytest.fixture
def fresh_gate(monkeypatch):
    """A reset gate whose environment the test controls; restored afterwards.

    Restoration is automatic: the monkeypatched environment is undone by pytest, and the
    next ``kernels.spec()`` anywhere re-runs the gate against the real environment.
    """
    native._reset_for_testing()
    yield monkeypatch
    native._reset_for_testing()


# --- the switch ---------------------------------------------------------------------------

def test_the_default_mode_is_auto(fresh_gate):
    fresh_gate.delenv("KUIVA_KERNELS", raising=False)
    assert native.mode() == "auto"


def test_an_invalid_mode_is_refused(fresh_gate):
    fresh_gate.setenv("KUIVA_KERNELS", "fortran77")
    with pytest.raises(ValueError, match="KUIVA_KERNELS"):
        native.mode()


def test_numpy_mode_never_loads_the_extension(fresh_gate):
    """KUIVA_KERNELS=numpy is the reproducibility setting: pure NumPy even with a build."""
    fresh_gate.setenv("KUIVA_KERNELS", "numpy")
    native.activate()
    assert not native.available()
    assert kernels.preferred_backend() == kernels.DEFAULT_BACKEND
    assert native.fingerprint_token() == "numpy"
    assert native.banner_entry() == "compiled kernels: disabled (KUIVA_KERNELS=numpy)"


def test_auto_mode_registers_the_build_when_present(fresh_gate):
    fresh_gate.delenv("KUIVA_KERNELS", raising=False)
    native.activate()
    if not native.available():
        pytest.skip(NEEDS_BUILD)
    assert kernels.preferred_backend() == "native"
    assert native.fingerprint_token() == "native:{}".format(native.build_id())
    assert native.build_id() in native.banner_entry()
    # auto still resolves numpy-only kernels: preference, not a hard requirement
    assert kernels.spec("cas_rank").backend == "numpy"


def test_an_explicit_native_request_is_refused_without_a_build(fresh_gate):
    """The gate philosophy: an explicit request is refused, never degraded.

    Simulated by pointing the import at a module that cannot exist, so the test is
    meaningful on a machine that *does* have the build.
    """
    fresh_gate.setenv("KUIVA_KERNELS", "native")
    fresh_gate.setattr(native, "_import_extension",
                       lambda: (_ for _ in ()).throw(ImportError("no build")))
    with pytest.raises(ImportError, match="95_native.sh"):
        native.activate()
    # and it refuses on EVERY resolution, not only the first
    with pytest.raises(ImportError, match="95_native.sh"):
        kernels.spec("cas_rank")
    assert native.fingerprint_token() == "native:unavailable"


def test_a_stale_api_version_is_refused_under_native(fresh_gate):
    """A .so predating a kernel-signature change may not register (staleness)."""
    import types

    stale = types.SimpleNamespace(API_VERSION=native.API_VERSION - 1, BUILD_ID="stale")
    fresh_gate.setenv("KUIVA_KERNELS", "native")
    fresh_gate.setitem(__import__("sys").modules, "kuiva._native", stale)
    with pytest.raises(ImportError, match="API_VERSION"):
        native.activate()


def test_lazy_loading_import_kuiva_has_no_side_effects(fresh_gate):
    """The gate fires on first resolution, not at import."""
    fresh_gate.delenv("KUIVA_KERNELS", raising=False)
    assert not native._STATE["activated"]
    kernels.spec("cas_rank")
    assert native._STATE["activated"]


# --- the probe ----------------------------------------------------------------------------

def test_probe_numpy_backend_always_exists():
    assert "numpy" in kernels.backends_for("native_probe")
    x = np.arange(4.0)
    out = np.empty(4)
    kernels.resolve("native_probe", "numpy")(x, out)
    assert np.array_equal(out, x + 1.0)


def test_probe_native_matches_numpy():
    _native_or_skip()
    x = np.linspace(-3.0, 7.0, 17)
    a = np.empty(17)
    b = np.empty(17)
    kernels.resolve("native_probe", "numpy")(x, a)
    kernels.resolve("native_probe", "native")(x, b)
    assert np.array_equal(a, b)


# --- parity: connections_scan (bitwise, even threaded) ------------------------------------

def _random_determinants(rng, ndet, n_spinor=20, n_elec=10):
    seen = set()
    masks = []
    while len(masks) < ndet:
        occ = rng.choice(n_spinor, size=n_elec, replace=False)
        m = np.uint64(0)
        for p in occ:
            m |= np.uint64(1) << np.uint64(p)
        if int(m) not in seen:
            seen.add(int(m))
            masks.append(m)
    return Determinants(np.array(masks, dtype=np.uint64), n_spinor, n_elec)


CONNECTION_FIELDS = ("single_i", "single_j", "single_from", "single_to", "single_phase",
                     "double_i", "double_j", "double_from", "double_to", "double_phase")


@pytest.mark.parametrize("n_threads", [1, 2, 4])
def test_connections_native_is_bitwise_at_every_thread_count(n_threads):
    """No FP reduction anywhere in this kernel, so threading may not move a single bit —
    a deviation here is a defect, never a tolerance case."""
    _native_or_skip()
    dets = _random_determinants(np.random.default_rng(11), 700)
    ref = connections(dets, backend="numpy")
    got = connections(dets, backend="native", n_threads=n_threads)
    for field in CONNECTION_FIELDS:
        a, b = getattr(ref, field), getattr(got, field)
        assert a.dtype == b.dtype and np.array_equal(a, b), field


def test_connections_native_bitwise_rectangular_and_odd_blocks():
    """row_limit and a block size that does not divide the row count exercise the wrapper's
    re-call/overflow protocol against the compiled kernel."""
    _native_or_skip()
    dets = _random_determinants(np.random.default_rng(12), 400)
    for kwargs in ({"row_limit": 23}, {"block": 37}, {"row_limit": 399, "block": 64}):
        ref = connections(dets, backend="numpy", **kwargs)
        got = connections(dets, backend="native", **kwargs)
        for field in CONNECTION_FIELDS:
            assert np.array_equal(getattr(ref, field), getattr(got, field)), (field, kwargs)


# --- parity: block_pair_gemm (bitwise serial; threaded per the B10 note) ------------------

def _random_pair_table(rng, n_out, npair, dmax):
    m_of = rng.integers(1, dmax, size=n_out)
    n_of = rng.integers(1, dmax, size=n_out)
    pairs, dims, a_shapes, b_shapes = [], [], [], []
    seen = set()
    # every output block gets a first (beta = 0) pair — the caller's invariant: tensordot
    # only allocates output blocks that at least one pair targets
    targets = list(range(n_out)) + [int(rng.integers(0, n_out))
                                    for _ in range(max(0, npair - n_out))]
    for io in targets:
        k = int(rng.integers(1, dmax))
        a_shapes.append((int(m_of[io]), k))
        b_shapes.append((k, int(n_of[io])))
        pairs.append((len(a_shapes) - 1, len(b_shapes) - 1, io, 1 if io in seen else 0))
        seen.add(io)
        dims.append((int(m_of[io]), k, int(n_of[io])))

    def pack(shapes):
        sizes = [m * n for m, n in shapes]
        off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        data = (rng.standard_normal(int(off[-1]))
                + 1j * rng.standard_normal(int(off[-1]))).astype(np.complex128)
        return data, off

    a_data, a_off = pack(a_shapes)
    b_data, b_off = pack(b_shapes)
    out_off = np.concatenate([[0], np.cumsum(m_of * n_of)]).astype(np.int64)
    return (a_data, a_off, b_data, b_off, np.asarray(pairs, dtype=np.int64),
            np.asarray(dims, dtype=np.int64), out_off)


def test_block_pair_gemm_native_serial_is_bitwise():
    """Same cblas_zgemm dispatch, same pair order, same beta treatment -> bitwise."""
    _native_or_skip()
    for seed, (n_out, npair, dmax) in enumerate([(1, 1, 4), (4, 12, 8), (16, 96, 33),
                                                 (40, 400, 64)]):
        case = _random_pair_table(np.random.default_rng(seed), n_out, npair, dmax)
        ref = np.empty(int(case[6][-1]), dtype=np.complex128)
        got = np.empty_like(ref)
        kernels.resolve("block_pair_gemm", "numpy")(*case[:6], ref, case[6], 1)
        kernels.resolve("block_pair_gemm", "native")(*case[:6], got, case[6], 1)
        assert np.array_equal(ref, got), (n_out, npair, dmax)


@pytest.mark.parametrize("n_threads", [2, 4])
def test_block_pair_gemm_native_threaded_holds_the_b10_band(n_threads):
    """Owner-computes preserves the pair accumulation order exactly; what remains is MKL's
    internal per-GEMM reduction, whose order depends on the MKL thread width (clamped to 1
    inside the region, ambient in the NumPy reference). Measured ~2e-16 relative on blocks
    with k >= ~32; the B10 note on the wrapper names it and the fixed 1e-13 band covers it.
    Small production-shaped blocks (the measured small-block regime) stay bitwise in practice, but
    the *claim* tested is the note's, not the lucky case."""
    _native_or_skip()
    for seed, (n_out, npair, dmax) in enumerate([(4, 12, 8), (16, 96, 33), (40, 400, 64)]):
        case = _random_pair_table(np.random.default_rng(seed + 100), n_out, npair, dmax)
        ref = np.empty(int(case[6][-1]), dtype=np.complex128)
        got = np.empty_like(ref)
        kernels.resolve("block_pair_gemm", "numpy")(*case[:6], ref, case[6], 1)
        kernels.resolve("block_pair_gemm", "native")(*case[:6], got, case[6], n_threads)
        scale = np.abs(ref).max()
        assert np.abs(ref - got).max() <= 1e-13 * scale


def test_block_pair_gemm_native_threaded_is_deterministic():
    """nt=2 and nt=4 clamp MKL identically and preserve the same per-output order, so two
    threaded runs must agree bitwise with each other whatever the team size."""
    _native_or_skip()
    case = _random_pair_table(np.random.default_rng(200), 24, 200, 48)
    a = np.empty(int(case[6][-1]), dtype=np.complex128)
    b = np.empty_like(a)
    kernels.resolve("block_pair_gemm", "native")(*case[:6], a, case[6], 2)
    kernels.resolve("block_pair_gemm", "native")(*case[:6], b, case[6], 4)
    assert np.array_equal(a, b)


# --- parity: sparse_pair_dot (bitwise, serial AND threaded per its B10 note) --------------

def _random_sparse_case(rng, n_out, n_csr, npair, dmax, density=0.4):
    """A flat-CSR pair table in the sparse_pair_dot contract's layout."""
    out_rest = rng.integers(1, dmax, size=n_out)          # rest_dim is per output block
    out_size = rng.integers(1, dmax, size=n_out)
    csr_rows, csr_cols, ip_parts, ix_parts, val_parts, meta = [], [], [], [], [], []
    ip0 = nz0 = 0
    for j in range(n_csr):
        rows, cols = int(rng.integers(1, dmax)), int(rng.integers(1, dmax))
        dense = (rng.standard_normal((rows, cols))
                 + 1j * rng.standard_normal((rows, cols)))
        dense[rng.random((rows, cols)) > density] = 0.0
        from scipy.sparse import csr_matrix
        csr = csr_matrix(dense)
        csr_rows.append(rows)
        csr_cols.append(cols)
        meta.append((ip0, nz0, rows))
        ip_parts.append(csr.indptr.astype(np.int64))
        ix_parts.append(csr.indices.astype(np.int64))
        val_parts.append(csr.data.astype(np.complex128))
        ip0 += rows + 1
        nz0 += int(csr.nnz)
    # every output block is targeted at least once; each pair picks a csr whose n_rows is
    # forced to the output's out_size by regenerating out_size from the csr (the wrapper's
    # invariant: out_size == the partner's n_rows)
    targets = list(range(n_out)) + [int(rng.integers(0, n_out))
                                    for _ in range(max(0, npair - n_out))]
    a_shapes, pairs, dims = [], [], []
    csr_of_out = {}                 # the wrapper's invariant: one out_size per out block
    for io in targets:
        ic = csr_of_out.setdefault(io, int(rng.integers(0, n_csr)))
        out_size[io] = csr_rows[ic]
        a_shapes.append((csr_cols[ic], int(out_rest[io])))
        pairs.append((len(a_shapes) - 1, ic, io))
        dims.append((csr_cols[ic], int(out_rest[io]), csr_rows[ic]))
    am_sizes = [m * n for m, n in a_shapes]
    am_off = np.concatenate([[0], np.cumsum(am_sizes)]).astype(np.int64)
    am_data = (rng.standard_normal(int(am_off[-1]))
               + 1j * rng.standard_normal(int(am_off[-1]))).astype(np.complex128)
    out_off = np.concatenate([[0], np.cumsum(out_rest * out_size)]).astype(np.int64)
    return (am_data, am_off, np.concatenate(val_parts),
            np.concatenate(ix_parts), np.concatenate(ip_parts),
            np.asarray(meta, dtype=np.int64), np.asarray(pairs, dtype=np.int64),
            np.asarray(dims, dtype=np.int64), out_off)


@pytest.mark.parametrize("n_threads", [1, 2, 4])
def test_sparse_pair_dot_native_is_bitwise_at_every_thread_count(n_threads):
    """Same per-pair scratch, same entry order, naive complex product, contraction off,
    owner-computes over output blocks: no reduction is reordered at any thread count, so
    parity is bitwise everywhere — the claim of the wrapper's B10 note, not a lucky case.
    """
    _native_or_skip()
    for seed, (n_out, n_csr, npair, dmax) in enumerate([(1, 1, 1, 4), (4, 3, 12, 8),
                                                        (16, 9, 96, 33), (40, 12, 300, 64)]):
        case = _random_sparse_case(np.random.default_rng(seed + 300), n_out, n_csr,
                                   npair, dmax)
        ref = np.zeros(int(case[8][-1]), dtype=np.complex128)
        got = np.zeros_like(ref)
        kernels.resolve("sparse_pair_dot", "numpy")(*case[:8], ref, case[8], 1)
        kernels.resolve("sparse_pair_dot", "native")(*case[:8], got, case[8], n_threads)
        assert np.array_equal(ref, got), (n_out, n_csr, npair, dmax, n_threads)


def test_dot_sparse_backends_agree_bitwise_on_a_block_tensor():
    """The wrapper level: one dot_sparse call, numpy vs native, production-shaped."""
    _native_or_skip()
    from kuiva.dmrg.block import BlockTensor, QuantumNumber as QN, Space
    from kuiva.dmrg.sparse import SparseW, dot_sparse

    rng = np.random.default_rng(11)
    mk = lambda ns: Space([(QN(q), int(rng.integers(1, 6))) for q in range(ns)])  # noqa: E731
    b_sp, c_sp = mk(3), mk(2)
    dense = BlockTensor.random((mk(2), b_sp, c_sp), (-1, 1, 1), QN(0), rng=rng)
    for blk in dense.blocks:
        blk[rng.random(blk.shape) > 0.3] = 0.0
    w = SparseW.from_block_tensor(dense)
    other = BlockTensor.random((mk(2), b_sp, c_sp), (1, -1, -1), QN(1), rng=rng)
    ref = _with_backend("numpy", lambda: dot_sparse(other, w, ([1, 2], [1, 2])))
    got = _with_backend("native", lambda: dot_sparse(other, w, ([1, 2], [1, 2])))
    assert np.array_equal(ref.sectors, got.sectors)
    for x, y in zip(ref.blocks, got.blocks):
        assert np.array_equal(x, y)


def _with_backend(backend, fn):
    previous = kernels.set_preferred_backend(backend)
    try:
        return fn()
    finally:
        kernels.set_preferred_backend(previous)


# --- provenance reaches the stored products ----------------------------------------------

def test_checkpoint_metadata_carries_the_kernel_backend(tmp_path, kuiva_caplog):
    """Checkpoint metadata gains the backend token, and a cross-backend restart warns —
    a threaded reduction is only 1e-13-equal, so this is a source-change-grade event."""
    h5py = pytest.importorskip("h5py")
    from test_checkpoint import make_checkpoint
    from kuiva.io import checkpoint as ckpt

    token = ckpt._kernel_backend_token()
    assert token == "numpy" or token.startswith("native:")

    path = tmp_path / "run.h5"
    ckpt.write_checkpoint(path, make_checkpoint())
    with h5py.File(str(path), "r") as handle:
        assert str(handle.attrs["kernel_backend"]) == token

    # rewrite the stored token to simulate a restart under the other backend
    with h5py.File(str(path), "r+") as handle:
        handle.attrs["kernel_backend"] = "native:0000000000000000-00000000"
    ckpt.read_checkpoint(path)
    assert any("kernel backend" in r.getMessage() for r in kuiva_caplog.records
               if r.levelname == "WARNING")
