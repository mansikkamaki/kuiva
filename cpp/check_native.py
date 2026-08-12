"""Verify a freshly built kernel backend. Run by ``make check``.

Three questions, in the order in which they fail:

1. does the extension import and register, and does its interface version match the gate's?
2. does a trivial kernel round-trip through the registry (the boundary itself: dtypes,
   layout, the caller-provided output buffer)?
3. does a real kernel, on a production-shaped instance, agree with its NumPy implementation
   **bitwise**?

The third is the one worth having. A compiled kernel that agrees to a few digits is a kernel
that moves committed reference numbers, and it does so on a machine nobody is looking at; the
parity tolerance is therefore fixed in advance rather than argued afterwards. Bitwise is the
requirement for a port that keeps the reduction order, which the kernels checked here do.
"""
import sys

import numpy as np

from kuiva.ci import kernels
from kuiva.ci.strings import Determinants, connections
from kuiva.util import native


def main():
    native.activate()
    if not native.available():
        print("FAIL: the extension built but did not register")
        return 1
    print("  build id  {}".format(native.build_id()))

    # The probe kernel: the boundary, with no arithmetic worth arguing about.
    x = np.arange(5.0)
    out = np.empty(5)
    kernels.resolve("native_probe", "native")(x, out)
    if not np.array_equal(out, x + 1.0):
        print("FAIL: the probe kernel returned a wrong answer")
        return 1

    # One real kernel: the determinant connection scan, on a spread-out sample of a
    # 12-spinor, 6-electron space.
    rng = np.random.default_rng(1)
    masks = rng.choice(np.arange(1 << 12, dtype=np.uint64), size=400, replace=False)
    masks = masks[np.array([bin(int(m)).count("1") for m in masks]) == 6][:200]
    dets = Determinants(masks, 12, 6)
    a = connections(dets, backend="numpy")
    b = connections(dets, backend="native")
    for field in ("single_i", "single_j", "single_from", "single_to", "single_phase",
                  "double_i", "double_j", "double_from", "double_to", "double_phase"):
        if not np.array_equal(getattr(a, field), getattr(b, field)):
            print("FAIL: connections_scan differs from the NumPy backend in {}".format(field))
            return 1

    print("  OK  probe + connections_scan bitwise vs numpy ({} singles, {} doubles)"
          .format(a.n_single, a.n_double))
    return 0


if __name__ == "__main__":
    sys.exit(main())
