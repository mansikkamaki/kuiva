// kuiva_native.cpp — the compiled ("native") kernel backend, one extension module.
//
// Built by `cd cpp && ./configure && make` into kuiva/_native<soabi>.so (git-ignored,
// in-place). Loaded by exactly one Python module, kuiva/util/native.py, which registers
// every kernel exported here under backend name "native" in the ci/kernels.py registry
//. Nothing else imports this module; nothing here is public API.
//
// Design rules this file lives under (the kernel-portability contract
// B1-B10 enforced by tests/test_kernel_contracts.py):
//   * plain arrays and scalars across the boundary (B1); dtypes, layout and aliasing are
//     asserted here at entry with the same exception types and message keywords as the
//     NumPy reference implementations, so the contract tests pass unchanged against both
//     backends (B4/B5/B6);
//   * nothing inside a work loop raises, logs, times or asks a budget for a number (B8);
//   * the thread count is an explicit argument, never omp_set_num_threads and never an
//     environment read (B7 applied to threads);
//   * parity against the NumPy backend is BITWISE for a serial run with the same reduction
//     order, and stays bitwise for the threaded paths implemented here because neither one
//     reorders a floating-point reduction (see the per-kernel notes below). The 1e-13
//     1e-13 relative tolerance is reserved for a future path that carries a B10 note.
//
// Threading model: OpenMP (libiomp5 — the SAME runtime MKL uses; a libgomp build would put
// two OpenMP runtimes in one process, which is a recorded trap, see the build script).
// Inside any parallel region MKL is clamped to one thread via mkl_set_num_threads_local();
// same-runtime nesting means MKL would detect this anyway, this is belt and braces
// (Intel oneAPI MKL threading/composability documentation, "MKL and OpenMP nesting").
//
// References:
//   * pybind11: W. Jakob, J. Rhinelander, D. Moldovan, "pybind11 — Seamless operability
//     between C++11 and Python", https://github.com/pybind/pybind11 (version pinned in
//     the pinned toolchain versions).
//   * Intel oneAPI Math Kernel Library (MKL): cblas_zgemm, mkl_set_num_threads_local.
//   * The XOR/popcount determinant connection scan is standard selected-CI practice
//     (Scemama & Giner, arXiv:1311.6244 (2013); Garniron et al., JCTC 15, 3591 (2019)) and
//     is a line-for-line port of kuiva/ci/strings.py's NumPy kernel, which carries the
//     full references and the sign convention.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <mkl.h>
#include <omp.h>

#include <bit>
#include <complex>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace py = pybind11;

using std::int64_t;
using std::uint64_t;
using cplx = std::complex<double>;

#ifndef KUIVA_NATIVE_BUILD_ID
#define KUIVA_NATIVE_BUILD_ID "unversioned"
#endif

// The interface version of this module. kuiva/util/native.py refuses (under
// KUIVA_KERNELS=native) or ignores (under auto) a module whose API_VERSION does not match
// its own — a stale .so must never register kernels with a drifted signature.
// History: 1 = Stage 1 (probe, connections_scan, block_pair_gemm); 2 = + sparse_pair_dot
// (the post-port re-profile, measured).
static constexpr int API_VERSION = 2;

// ---------------------------------------------------------------------------------------
// Boundary checks (B4/B5/B6). Message keywords ("must be", "C-contiguous", "alias") are
// part of the kernel contract: tests/test_kernel_contracts.py greps for them, and both
// backends must fail the same way for the same reason.
// ---------------------------------------------------------------------------------------

static void check_dtype(const py::array& a, const py::dtype& want, const char* name,
                        const char* wanted_name) {
    // Compare NumPy type numbers, not object identity: builtin dtype singletons are an
    // implementation detail of the NumPy version, and this must hold on 1.x and 2.x alike.
    if (a.dtype().num() != want.num())
        throw py::type_error(std::string(name) + " must be " + wanted_name);
}

static void check_carray(const py::array& a, const char* name) {
    if (!(a.flags() & py::array::c_style))
        throw py::value_error(std::string(name) + " must be C-contiguous");
}

static bool ranges_overlap(const py::array& x, const py::array& y) {
    // Flat address-range overlap. The buffers here are always dense C-contiguous blocks
    // (asserted above), so range overlap and NumPy's shares_memory agree.
    auto lo1 = reinterpret_cast<const char*>(x.data());
    auto hi1 = lo1 + static_cast<size_t>(x.nbytes());
    auto lo2 = reinterpret_cast<const char*>(y.data());
    auto hi2 = lo2 + static_cast<size_t>(y.nbytes());
    return lo1 < hi2 && lo2 < hi1;
}

static void check_no_alias(const py::array& out, const py::array& in, const char* what) {
    if (ranges_overlap(out, in))
        throw py::value_error(std::string("the output buffer may not alias ") + what);
}

// ---------------------------------------------------------------------------------------
// native_probe — the trivial registered kernel that makes the gate, the registry, the
// contract walk and the parity machinery testable before (and independently of) any real
// kernel. out[i] = x[i] + 1, float64.
// ---------------------------------------------------------------------------------------

static py::array native_probe(py::array x, py::array out) {
    auto f8 = py::dtype::of<double>();
    check_dtype(x, f8, "x", "float64");
    check_dtype(out, f8, "out", "float64");
    check_carray(x, "x");
    check_carray(out, "out");
    check_no_alias(out, x, "an operand");
    if (x.ndim() != 1 || out.ndim() != 1 || x.shape(0) != out.shape(0))
        throw py::value_error("x and out must be equal-length 1-D arrays");
    auto* xs = static_cast<const double*>(x.data());
    auto* os = static_cast<double*>(out.mutable_data());
    const int64_t n = static_cast<int64_t>(x.shape(0));
    for (int64_t i = 0; i < n; ++i) os[i] = xs[i] + 1.0;
    return out;
}

// ---------------------------------------------------------------------------------------
// connections_scan — the O(N^2) XOR/popcount determinant pair scan.
//
// Bit conventions are kuiva/ci/strings.py's and are NOT re-derived: this is a port of the
// NumPy kernel of the same name, phase formulas included (the module docstring there is
// the authoritative statement). Outputs are integer index arrays and +-1.0 phases — pure
// per-pair functions with NO floating-point reduction anywhere, so the threaded scan below
// is bitwise identical to the serial one by construction: rows are split into contiguous
// ascending chunks, each thread fills a private buffer, and the buffers are concatenated
// in chunk (= row) order.
// ---------------------------------------------------------------------------------------

namespace {

struct ScanBuffer {
    std::vector<int64_t> si, sj, sf, st;
    std::vector<double> sp;
    std::vector<int64_t> di, dj, df0, df1, dt0, dt1;
    std::vector<double> dp;
};

inline uint64_t below_mask(int p) { return (uint64_t(1) << p) - uint64_t(1); }

inline double phase_single(uint64_t det_j, int i, int a) {
    int s = std::popcount(det_j & below_mask(i));
    uint64_t d1 = det_j ^ (uint64_t(1) << i);
    s += std::popcount(d1 & below_mask(a));
    return (s & 1) ? -1.0 : 1.0;
}

inline double phase_double(uint64_t det_j, int i, int j, int a, int b) {
    int s = std::popcount(det_j & below_mask(i));
    uint64_t d1 = det_j ^ (uint64_t(1) << i);
    s += std::popcount(d1 & below_mask(j));
    uint64_t d2 = d1 ^ (uint64_t(1) << j);
    s += std::popcount(d2 & below_mask(b));
    uint64_t d3 = d2 | (uint64_t(1) << b);
    s += std::popcount(d3 & below_mask(a));
    return (s & 1) ? -1.0 : 1.0;
}

inline void scan_rows(const uint64_t* masks, int64_t n, int64_t lo, int64_t hi,
                      ScanBuffer& buf) {
    for (int64_t I = lo; I < hi; ++I) {
        const uint64_t mi = masks[I];
        for (int64_t J = I + 1; J < n; ++J) {
            const uint64_t mj = masks[J];
            const uint64_t x = mi ^ mj;
            const int c = std::popcount(x);
            if (c == 2) {
                const int to = std::countr_zero(x & mi);   // created in I
                const int fr = std::countr_zero(x & mj);   // annihilated in J
                buf.si.push_back(I);
                buf.sj.push_back(J);
                buf.sf.push_back(fr);
                buf.st.push_back(to);
                buf.sp.push_back(phase_single(mj, fr, to));
            } else if (c == 4) {
                const uint64_t bits_i = x & mi;
                const uint64_t bits_j = x & mj;
                const uint64_t lo_i = bits_i & (~bits_i + 1);
                const uint64_t lo_j = bits_j & (~bits_j + 1);
                const int a = std::countr_zero(lo_i);            // to, ascending
                const int b = std::countr_zero(bits_i ^ lo_i);
                const int fi = std::countr_zero(lo_j);           // from, ascending
                const int fj = std::countr_zero(bits_j ^ lo_j);
                buf.di.push_back(I);
                buf.dj.push_back(J);
                buf.df0.push_back(fi);
                buf.df1.push_back(fj);
                buf.dt0.push_back(a);
                buf.dt1.push_back(b);
                buf.dp.push_back(phase_double(mj, fi, fj, a, b));
            }
        }
    }
}

// Chunk boundaries balanced by pair count: row I owns n-1-I column partners, so the
// leading rows are the heavy ones and an equal-rows split would idle half the team.
// Contiguous ascending chunks keep the concatenation order equal to the serial row order.
inline std::vector<int64_t> balanced_bounds(int64_t n, int64_t row_start, int64_t row_stop,
                                            int nt) {
    std::vector<int64_t> bounds(nt + 1);
    bounds[0] = row_start;
    auto pairs_before = [&](int64_t r) {
        // total pairs in rows [row_start, r): sum_{I} (n - 1 - I)
        const double a = static_cast<double>(row_start);
        const double b = static_cast<double>(r);
        return (b - a) * (static_cast<double>(n) - 1.0) - 0.5 * (b * (b - 1.0) - a * (a - 1.0));
    };
    const double total = pairs_before(row_stop);
    int64_t r = row_start;
    for (int t = 1; t < nt; ++t) {
        const double want = total * t / nt;
        while (r < row_stop && pairs_before(r + 1) < want) ++r;
        bounds[t] = r;
    }
    bounds[nt] = row_stop;
    return bounds;
}

}  // namespace

static py::tuple connections_scan(py::array masks, int64_t row_start, int64_t row_stop,
                                  py::array s_i, py::array s_j, py::array s_from,
                                  py::array s_to, py::array s_phase,
                                  py::array d_i, py::array d_j, py::array d_from,
                                  py::array d_to, py::array d_phase, int64_t n_threads) {
    auto u8 = py::dtype("uint64");
    auto i8 = py::dtype("int64");
    auto f8 = py::dtype::of<double>();
    check_dtype(masks, u8, "masks", "uint64");
    check_carray(masks, "masks");
    const py::array* singles[5] = {&s_i, &s_j, &s_from, &s_to, &s_phase};
    const char* snames[5] = {"s_i", "s_j", "s_from", "s_to", "s_phase"};
    for (int q = 0; q < 5; ++q) {
        check_dtype(*singles[q], q < 4 ? i8 : f8, snames[q], q < 4 ? "int64" : "float64");
        check_carray(*singles[q], snames[q]);
        check_no_alias(*singles[q], masks, "the mask array");
        if (singles[q]->ndim() != 1)
            throw py::value_error(std::string(snames[q]) + " must be 1-D");
    }
    const py::array* doubles[5] = {&d_i, &d_j, &d_from, &d_to, &d_phase};
    const char* dnames[5] = {"d_i", "d_j", "d_from", "d_to", "d_phase"};
    for (int q = 0; q < 5; ++q) {
        bool idx = (q != 4);
        check_dtype(*doubles[q], idx ? i8 : f8, dnames[q], idx ? "int64" : "float64");
        check_carray(*doubles[q], dnames[q]);
        check_no_alias(*doubles[q], masks, "the mask array");
    }
    if (d_from.ndim() != 2 || d_from.shape(1) != 2 || d_to.ndim() != 2 || d_to.shape(1) != 2)
        throw py::value_error("d_from and d_to must have shape (capacity, 2)");
    if (d_i.ndim() != 1 || d_j.ndim() != 1 || d_phase.ndim() != 1)
        throw py::value_error("d_i, d_j and d_phase must be 1-D");
    const int64_t n = static_cast<int64_t>(masks.shape(0));
    if (row_start < 0 || row_stop < row_start || row_stop > n)
        throw py::value_error("row range must satisfy 0 <= row_start <= row_stop <= n");
    const int64_t cap1 = static_cast<int64_t>(s_i.shape(0));
    const int64_t cap2 = static_cast<int64_t>(d_i.shape(0));
    for (int q = 1; q < 5; ++q)
        if (static_cast<int64_t>(singles[q]->shape(0)) != cap1)
            throw py::value_error("single-excitation buffers must share one capacity");
    if (static_cast<int64_t>(d_j.shape(0)) != cap2
            || static_cast<int64_t>(d_from.shape(0)) != cap2
            || static_cast<int64_t>(d_to.shape(0)) != cap2
            || static_cast<int64_t>(d_phase.shape(0)) != cap2)
        throw py::value_error("double-excitation buffers must share one capacity");
    if (n_threads < 1)
        throw py::value_error("the thread count must be a positive integer");

    const uint64_t* mp = static_cast<const uint64_t*>(masks.data());
    int64_t* p_si = static_cast<int64_t*>(s_i.mutable_data());
    int64_t* p_sj = static_cast<int64_t*>(s_j.mutable_data());
    int64_t* p_sf = static_cast<int64_t*>(s_from.mutable_data());
    int64_t* p_st = static_cast<int64_t*>(s_to.mutable_data());
    double* p_sp = static_cast<double*>(s_phase.mutable_data());
    int64_t* p_di = static_cast<int64_t*>(d_i.mutable_data());
    int64_t* p_dj = static_cast<int64_t*>(d_j.mutable_data());
    int64_t* p_df = static_cast<int64_t*>(d_from.mutable_data());
    int64_t* p_dt = static_cast<int64_t*>(d_to.mutable_data());
    double* p_dp = static_cast<double*>(d_phase.mutable_data());

    int64_t n1 = 0, n2 = 0;
    bool wrote = false;
    {
        py::gil_scoped_release release;
        const int64_t rows = row_stop - row_start;
        int nt = static_cast<int>(n_threads);
        if (rows > 0 && rows < nt) nt = static_cast<int>(rows);
        if (nt < 1) nt = 1;
        std::vector<ScanBuffer> bufs(static_cast<size_t>(nt));
        if (nt == 1) {
            scan_rows(mp, n, row_start, row_stop, bufs[0]);
        } else {
            const std::vector<int64_t> bounds = balanced_bounds(n, row_start, row_stop, nt);
            #pragma omp parallel num_threads(nt)
            {
                const int T = omp_get_num_threads();
                const int t = omp_get_thread_num();
                // If the runtime granted fewer threads than asked, fold the tail chunks
                // into the last thread so every row is scanned exactly once, still in
                // ascending order per buffer.
                const int64_t lo = bounds[t];
                const int64_t hi = (t == T - 1) ? row_stop : bounds[t + 1];
                if (lo < hi) scan_rows(mp, n, lo, hi, bufs[static_cast<size_t>(t)]);
            }
        }
        for (const auto& b : bufs) {
            n1 += static_cast<int64_t>(b.si.size());
            n2 += static_cast<int64_t>(b.di.size());
        }
        if (n1 <= cap1 && n2 <= cap2) {
            wrote = true;
            int64_t o1 = 0, o2 = 0;
            for (const auto& b : bufs) {
                const int64_t k1 = static_cast<int64_t>(b.si.size());
                const int64_t k2 = static_cast<int64_t>(b.di.size());
                if (k1) {
                    std::memcpy(p_si + o1, b.si.data(), sizeof(int64_t) * k1);
                    std::memcpy(p_sj + o1, b.sj.data(), sizeof(int64_t) * k1);
                    std::memcpy(p_sf + o1, b.sf.data(), sizeof(int64_t) * k1);
                    std::memcpy(p_st + o1, b.st.data(), sizeof(int64_t) * k1);
                    std::memcpy(p_sp + o1, b.sp.data(), sizeof(double) * k1);
                }
                for (int64_t q = 0; q < k2; ++q) {
                    p_di[o2 + q] = b.di[static_cast<size_t>(q)];
                    p_dj[o2 + q] = b.dj[static_cast<size_t>(q)];
                    p_df[2 * (o2 + q)] = b.df0[static_cast<size_t>(q)];
                    p_df[2 * (o2 + q) + 1] = b.df1[static_cast<size_t>(q)];
                    p_dt[2 * (o2 + q)] = b.dt0[static_cast<size_t>(q)];
                    p_dt[2 * (o2 + q) + 1] = b.dt1[static_cast<size_t>(q)];
                    p_dp[o2 + q] = b.dp[static_cast<size_t>(q)];
                }
                o1 += k1;
                o2 += k2;
            }
        }
    }
    // wrote == false is the overflow protocol, not an error: the counts go back and the
    // wrapper reallocates and re-calls (B8: no raise for a capacity miss).
    (void)wrote;
    return py::make_tuple(n1, n2);
}

// ---------------------------------------------------------------------------------------
// block_pair_gemm — out[io] += A[ia] @ B[ib] over a matched pair table.
//
// The signature, semantics and beta convention are kuiva/dmrg/block.py's NumPy kernel of
// the same name (its docstring is the authoritative statement of the buffer layout).
//
// Serial path (n_threads == 1): pairs in table order, each pair one zgemm — the SAME
// Fortran zgemm call the NumPy kernel issues through SciPy's binding (see pair_gemm), so
// a serial run is bitwise by construction. zgemm3m would forfeit that and is deliberately
// not used.
//
// Threaded path: OWNER-COMPUTES over output blocks — output block io belongs to thread
// io % T, and each owner processes its pairs in table order, so the per-output pair
// accumulation order is IDENTICAL to the serial path at every thread count. ⚠ The one
// reduction not pinned by that is MKL's own, inside a single zgemm: it depends on the MKL
// thread width, and inside the parallel region MKL is clamped to 1 while a NumPy reference
// runs it ambient. Measured (2026-08-09): at any fixed MKL width, every thread count here
// is bitwise vs NumPy; across widths, blocks with k of order 32+ differ by ~2e-16 relative
// — within the fixed 1e-13 band, and equal to what pure NumPy shows between the same two MKL
// widths. The full B10 note lives on the Python wrapper (kuiva/util/native.py).
//
// MKL inside the parallel region is clamped to one thread (mkl_set_num_threads_local);
// same-runtime nesting (libiomp5) means MKL detects nesting anyway — belt and braces.
// ---------------------------------------------------------------------------------------

namespace {

// ⚠ The EXACT call the NumPy reference kernel makes, argument for argument: SciPy's
// zgemm binding on the transposed views — Fortran column-major C^T = B^T A^T on the same
// row-major buffers, with the pair's beta passed straight to the routine. One shared BLAS
// entry point on both backends is what makes bitwise parity a property of the contract:
// numpy.dot was measured (2026-08-09) serving size-1-operand products through its own
// build-dependent SIMD multiply, ~1 ulp away from every BLAS routine and unmatchable
// without reproducing a specific NumPy binary. cblas_zgemm with CblasColMajor is a thin
// shim onto the same Fortran zgemm scipy calls.
inline void pair_gemm(const cplx* a, const cplx* b, cplx* c, int64_t m, int64_t k,
                      int64_t n, bool accumulate) {
    const cplx one(1.0, 0.0);
    const cplx beta(accumulate ? 1.0 : 0.0, 0.0);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, static_cast<MKL_INT>(n),
                static_cast<MKL_INT>(m), static_cast<MKL_INT>(k), &one, b,
                static_cast<MKL_INT>(n), a, static_cast<MKL_INT>(k), &beta, c,
                static_cast<MKL_INT>(n));
}

}  // namespace

static py::array block_pair_gemm(py::array a_data, py::array a_offset, py::array b_data,
                                 py::array b_offset, py::array pairs, py::array dims,
                                 py::array out_data, py::array out_offset,
                                 int64_t n_threads) {
    auto c16 = py::dtype("complex128");
    auto i8 = py::dtype("int64");
    check_dtype(a_data, c16, "operand buffers", "complex128");
    check_dtype(b_data, c16, "operand buffers", "complex128");
    check_dtype(out_data, c16, "the output buffer", "complex128");
    check_dtype(a_offset, i8, "a_offset", "int64");
    check_dtype(b_offset, i8, "b_offset", "int64");
    check_dtype(out_offset, i8, "out_offset", "int64");
    check_dtype(pairs, i8, "pairs", "int64");
    check_dtype(dims, i8, "dims", "int64");
    for (const py::array* a : {&a_data, &b_data, &out_data, &a_offset, &b_offset,
                               &out_offset, &pairs, &dims})
        check_carray(*a, "operand and output buffers");
    check_no_alias(out_data, a_data, "an operand");
    check_no_alias(out_data, b_data, "an operand");
    if (pairs.ndim() != 2 || pairs.shape(1) != 4)
        throw py::value_error("pairs must have shape (npair, 4)");
    if (dims.ndim() != 2 || dims.shape(1) != 3 || dims.shape(0) != pairs.shape(0))
        throw py::value_error("dims must have shape (npair, 3)");
    if (n_threads < 1)
        throw py::value_error("the thread count must be a positive integer");

    const int64_t npair = static_cast<int64_t>(pairs.shape(0));
    const int64_t* pr = static_cast<const int64_t*>(pairs.data());
    const int64_t* dm = static_cast<const int64_t*>(dims.data());
    const int64_t* ao = static_cast<const int64_t*>(a_offset.data());
    const int64_t* bo = static_cast<const int64_t*>(b_offset.data());
    const int64_t* oo = static_cast<const int64_t*>(out_offset.data());
    const cplx* ad = static_cast<const cplx*>(a_data.data());
    const cplx* bd = static_cast<const cplx*>(b_data.data());
    cplx* od = static_cast<cplx*>(out_data.mutable_data());

    {
        py::gil_scoped_release release;
        if (n_threads == 1 || npair == 0) {
            for (int64_t p = 0; p < npair; ++p) {
                const int64_t m = dm[3 * p], k = dm[3 * p + 1], nn = dm[3 * p + 2];
                pair_gemm(ad + ao[pr[4 * p]], bd + bo[pr[4 * p + 1]], od + oo[pr[4 * p + 2]],
                          m, k, nn, pr[4 * p + 3] != 0);
            }
        } else {
            const int nt = static_cast<int>(n_threads);
            #pragma omp parallel num_threads(nt)
            {
                const int saved = mkl_set_num_threads_local(1);
                const int T = omp_get_num_threads();
                const int t = omp_get_thread_num();
                for (int64_t p = 0; p < npair; ++p) {
                    const int64_t io = pr[4 * p + 2];
                    if (static_cast<int>(io % T) != t) continue;
                    const int64_t m = dm[3 * p], k = dm[3 * p + 1], nn = dm[3 * p + 2];
                    pair_gemm(ad + ao[pr[4 * p]], bd + bo[pr[4 * p + 1]], od + oo[io],
                              m, k, nn, pr[4 * p + 3] != 0);
                }
                mkl_set_num_threads_local(saved);
            }
        }
    }
    return out_data;
}

// ---------------------------------------------------------------------------------------
// sparse_pair_dot — out[io] += (csr[ic] @ A[ia])^T over a matched pair table.
//
// The sparse-W application: the second tensor-network hot kernel, measured at 48% of a
// post-Stage-1c sweep's CPU, serial, share flat in the bond dimension
// (measured under the port gate). The signature, buffer layout and
// semantics are kuiva/dmrg/sparse.py's NumPy kernel of the same name (its docstring is the
// authoritative statement); the B10 reduction-order note there is what the bitwise claim
// below rests on.
//
// Bitwise parity, both paths: the NumPy kernel's engine is SciPy's csr_matvecs — per CSR
// row, entries in ascending stored order, each one component-wise multiply-accumulate into
// a zeroed per-pair scratch, then one add per element into the output. This kernel
// reproduces that arithmetic EXACTLY: the same per-pair scratch (never accumulating
// entries straight into the output, which would reorder the sum), the same naive complex
// product (r1*r2 - i1*i2, r1*i2 + i1*r2) on doubles — scipy's complex_wrapper formula, not
// std::complex operator* whose __muldc3 NaN fixup is a different function — and FP
// contraction OFF for the arithmetic loops, so no FMA regroups what scipy's build did not.
// Threading is owner-computes over output blocks (block io belongs to thread io % T, pairs
// walked in table order), identical to block_pair_gemm's scheme: the per-output pair order
// equals the serial one at every thread count, so the threaded path stays bitwise. No BLAS
// is called here, so there is no MKL width term (unlike block_pair_gemm's B10 note).
// ---------------------------------------------------------------------------------------

namespace {

// One pair: zero the scratch, run the CSR rows in entry order, then add the transposed
// scratch into the output block. Kept out-of-line from the OpenMP region so the serial and
// threaded paths are the same code by construction.
inline void sparse_pair_apply(const cplx* am, const int64_t* indptr, const int64_t* cols,
                              const cplx* vals, cplx* outp, cplx* piece, int64_t in_size,
                              int64_t rest, int64_t out_sz) {
    (void)in_size;                                  // implied by the column indices
    {
        // VALUE-SAFE FP for this block, and both halves were MEASURED, not assumed
        // (2026-08-10, exact-rational probe + pure-Python reconstruction against the
        // pinned Intel SciPy): (a) SciPy's csr_matvecs is the naive unfused sequence —
        // product components (r1*r2 - i1*i2, r1*i2 + i1*r2) with no FMA, accumulated
        // entry by entry — so any contraction here puts entries ~1 ulp off; (b) icpx
        // defaults to -fp-model=fast, under which `fp contract(off)` alone is NOT enough:
        // the vectorizer may still reassociate, and a contract(off) build measurably
        // diverged while a pure-Python replay of the stated order matched SciPy exactly.
        // float_control(precise) pins both. If the SciPy pin ever moves to a build that
        // fuses, this is the block to revisit (the reduction ORDER is fixed either way;
        // only operand-level rounding is at stake).
        #pragma float_control(precise, on)
        #pragma clang fp contract(off)
        std::memset(piece, 0, sizeof(cplx) * static_cast<size_t>(out_sz * rest));
        double* pd = reinterpret_cast<double*>(piece);
        const double* ad = reinterpret_cast<const double*>(am);
        for (int64_t r = 0; r < out_sz; ++r) {
            double* prow = pd + 2 * r * rest;
            for (int64_t e = indptr[r]; e < indptr[r + 1]; ++e) {
                const cplx v = vals[e];
                const double vr = v.real(), vi = v.imag();
                const double* arow = ad + 2 * cols[e] * rest;
                for (int64_t k = 0; k < rest; ++k) {
                    const double ar = arow[2 * k], ai = arow[2 * k + 1];
                    const double tr = vr * ar - vi * ai;
                    const double ti = vr * ai + vi * ar;
                    prow[2 * k] += tr;
                    prow[2 * k + 1] += ti;
                }
            }
        }
        double* od = reinterpret_cast<double*>(outp);
        for (int64_t r = 0; r < out_sz; ++r) {
            const double* prow = pd + 2 * r * rest;
            for (int64_t k = 0; k < rest; ++k) {
                od[2 * (k * out_sz + r)] += prow[2 * k];
                od[2 * (k * out_sz + r) + 1] += prow[2 * k + 1];
            }
        }
    }
}

}  // namespace

static py::array sparse_pair_dot(py::array am_data, py::array am_offset,
                                 py::array csr_values, py::array csr_indices,
                                 py::array csr_indptr, py::array csr_meta, py::array pairs,
                                 py::array dims, py::array out_data, py::array out_offset,
                                 int64_t n_threads) {
    auto c16 = py::dtype("complex128");
    auto i8 = py::dtype("int64");
    check_dtype(am_data, c16, "operand buffers", "complex128");
    check_dtype(csr_values, c16, "operand buffers", "complex128");
    check_dtype(out_data, c16, "the output buffer", "complex128");
    for (const py::array* a : {&am_offset, &csr_indices, &csr_indptr, &csr_meta, &pairs,
                               &dims, &out_offset})
        check_dtype(*a, i8, "index tables", "int64");
    for (const py::array* a : {&am_data, &csr_values, &csr_indices, &csr_indptr, &out_data,
                               &am_offset, &csr_meta, &pairs, &dims, &out_offset})
        check_carray(*a, "operand and output buffers");
    check_no_alias(out_data, am_data, "an operand");
    check_no_alias(out_data, csr_values, "an operand");
    if (pairs.ndim() != 2 || pairs.shape(1) != 3)
        throw py::value_error("pairs must have shape (npair, 3)");
    if (dims.ndim() != 2 || dims.shape(1) != 3 || dims.shape(0) != pairs.shape(0))
        throw py::value_error("dims must have shape (npair, 3)");
    if (csr_meta.ndim() != 2 || csr_meta.shape(1) != 3)
        throw py::value_error("csr_meta must have shape (n_csr, 3)");
    if (n_threads < 1)
        throw py::value_error("the thread count must be a positive integer");

    const int64_t npair = static_cast<int64_t>(pairs.shape(0));
    const int64_t* pr = static_cast<const int64_t*>(pairs.data());
    const int64_t* dm = static_cast<const int64_t*>(dims.data());
    const int64_t* ao = static_cast<const int64_t*>(am_offset.data());
    const int64_t* oo = static_cast<const int64_t*>(out_offset.data());
    const int64_t* mt = static_cast<const int64_t*>(csr_meta.data());
    const int64_t* ipd = static_cast<const int64_t*>(csr_indptr.data());
    const int64_t* ixd = static_cast<const int64_t*>(csr_indices.data());
    const cplx* vd = static_cast<const cplx*>(csr_values.data());
    const cplx* ad = static_cast<const cplx*>(am_data.data());
    cplx* od = static_cast<cplx*>(out_data.mutable_data());

    int64_t max_piece = 0;                          // scratch size, found outside the loop
    for (int64_t p = 0; p < npair; ++p) {
        const int64_t sz = dm[3 * p + 1] * dm[3 * p + 2];
        if (sz > max_piece) max_piece = sz;
    }

    {
        py::gil_scoped_release release;
        if (n_threads == 1 || npair == 0) {
            std::vector<cplx> piece(static_cast<size_t>(max_piece));
            for (int64_t p = 0; p < npair; ++p) {
                const int64_t ic = pr[3 * p + 1];
                sparse_pair_apply(ad + ao[pr[3 * p]], ipd + mt[3 * ic],
                                  ixd + mt[3 * ic + 1], vd + mt[3 * ic + 1],
                                  od + oo[pr[3 * p + 2]], piece.data(), dm[3 * p],
                                  dm[3 * p + 1], dm[3 * p + 2]);
            }
        } else {
            const int nt = static_cast<int>(n_threads);
            #pragma omp parallel num_threads(nt)
            {
                const int T = omp_get_num_threads();
                const int t = omp_get_thread_num();
                std::vector<cplx> piece(static_cast<size_t>(max_piece));
                for (int64_t p = 0; p < npair; ++p) {
                    const int64_t io = pr[3 * p + 2];
                    if (static_cast<int>(io % T) != t) continue;
                    const int64_t ic = pr[3 * p + 1];
                    sparse_pair_apply(ad + ao[pr[3 * p]], ipd + mt[3 * ic],
                                      ixd + mt[3 * ic + 1], vd + mt[3 * ic + 1],
                                      od + oo[io], piece.data(), dm[3 * p],
                                      dm[3 * p + 1], dm[3 * p + 2]);
                }
            }
        }
    }
    return out_data;
}

// ---------------------------------------------------------------------------------------

PYBIND11_MODULE(_native, m) {
    m.doc() = "Kuiva compiled kernel backend. Loaded ONLY by kuiva/util/native.py; "
              "see cpp/kuiva_native.cpp and kuiva/ci/kernels.py.";
    m.attr("API_VERSION") = py::int_(API_VERSION);
    m.attr("BUILD_ID") = KUIVA_NATIVE_BUILD_ID;
    m.def("native_probe", &native_probe, py::arg("x").noconvert(),
          py::arg("out").noconvert());
    m.def("connections_scan", &connections_scan, py::arg("masks").noconvert(),
          py::arg("row_start"), py::arg("row_stop"), py::arg("s_i").noconvert(),
          py::arg("s_j").noconvert(), py::arg("s_from").noconvert(),
          py::arg("s_to").noconvert(), py::arg("s_phase").noconvert(),
          py::arg("d_i").noconvert(), py::arg("d_j").noconvert(),
          py::arg("d_from").noconvert(), py::arg("d_to").noconvert(),
          py::arg("d_phase").noconvert(), py::arg("n_threads"));
    m.def("block_pair_gemm", &block_pair_gemm, py::arg("a_data").noconvert(),
          py::arg("a_offset").noconvert(), py::arg("b_data").noconvert(),
          py::arg("b_offset").noconvert(), py::arg("pairs").noconvert(),
          py::arg("dims").noconvert(), py::arg("out_data").noconvert(),
          py::arg("out_offset").noconvert(), py::arg("n_threads"));
    m.def("sparse_pair_dot", &sparse_pair_dot, py::arg("am_data").noconvert(),
          py::arg("am_offset").noconvert(), py::arg("csr_values").noconvert(),
          py::arg("csr_indices").noconvert(), py::arg("csr_indptr").noconvert(),
          py::arg("csr_meta").noconvert(), py::arg("pairs").noconvert(),
          py::arg("dims").noconvert(), py::arg("out_data").noconvert(),
          py::arg("out_offset").noconvert(), py::arg("n_threads"));
}
