"""Example 1 -- the front end: a scalar SCF and the multireference starting point.

    source setup.sh          # once per shell
    python scf_and_reference.py

Runs in a few seconds on a neon atom and writes ``output/scf_and_reference.out``.

WHAT THIS SHOWS
---------------
Every Kuiva calculation begins with the same two stages, and this example is nothing but
those two:

    ScalarSCF   a scalar-relativistic X2C self-consistent field, run through PySCF, which
                supplies real orbitals, the overlap matrix, the one-electron X2C
                Hamiltonian and the two-electron integrals. This is the only place PySCF
                is used; nothing after it calls PySCF again.

    Reference   the multireference starting point: the raw AO basis is orthonormalized
                (removing linear dependence), each real orbital is expanded into a Kramers
                pair of two-component spinors, and the two-electron integrals are
                factorized. Everything downstream -- CI, CASSCF, DMRG, NEVPT2, the property
                files -- is built on this object.

Spin-orbit coupling is *ingested* here but not felt yet: the SCF stays scalar and the
spinor guess is exactly Kramers paired. SOC first changes an answer in the CASSCF, which is
example 2 onwards.

WHAT TO LOOK FOR IN THE OUTPUT
------------------------------
* the memory pre-flight table, printed before the SCF: Kuiva sizes the whole calculation
  from the basis-set dimensions and refuses to start if the plan does not fit the limit;
* the Hamiltonian block, naming the two independent choices behind the name ``X2C-AMF``:
  how the four-component problem was decoupled, and which two-electron spin-orbit
  screening the Hamiltonian already contains;
* the orthonormal working basis and how many directions (if any) it dropped;
* the spinor expansion: two spinors per orbital, with a Kramers pairing deviation of zero;
* the Cholesky decomposition of the two-electron integrals and its error bound;
* the same front end run a second time with ``fitting="cholesky-direct"``, which evaluates
  the integrals as the decomposition asks for them and never builds the array -- compare the
  two pre-flight tables, and note that the answers are identical;
* a table of checks, all of which must pass, and the timing and memory summaries.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

import kuiva
from kuiva.integrals.transform import transform_1e
from kuiva.spinor.expand import time_reverse
from kuiva.util import output as out
from kuiva.util import resources as res
from kuiva.util import timing
from kuiva.util.logging import add_file_handler, get_logger

NAME = "scf_and_reference"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

# One logger per module, named for the module. Everything an example prints goes through
# the logger and through kuiva.util.output, never through bare `print`: that is what keeps
# the output file aligned and greppable.
log = get_logger("examples." + NAME)


def prepare_output() -> Path:
    """Write this run to output/<name>.out, with a scratch spin-orbit cache of its own.

    Two independent reasons for the cache directory. The atomic mean-field spin-orbit
    correction (below) is expensive and is cached on disk between jobs, keyed on the
    element, basis and configuration -- so a demonstration that used your real cache would
    be fast or slow depending on what you happened to have computed before, and its timings
    would mean nothing. And an example must not write into a cache you rely on.
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT / "amf-cache"
    shutil.rmtree(cache, ignore_errors=True)
    os.environ["KUIVA_AMF_CACHE"] = str(cache)
    path = OUTPUT / (NAME + ".out")
    add_file_handler(path)
    return path


def main() -> int:
    outfile = prepare_output()
    out.banner(log, kuiva.__version__, "example 1: the front end (Ne atom)")
    out.entry(log, "output file", str(outfile.relative_to(HERE)))

    # ----------------------------------------------------------------------------------
    # 1. The system.
    # ----------------------------------------------------------------------------------
    # Kuiva's input is a Python API, not a text format. A Molecule carries the geometry (in
    # Angstrom), the charge, the spin as 2S, and a basis-set name from the registry. The
    # basis assignment is checked here and now: an element the family does not cover, or a
    # mixture of families targeting incompatible relativistic treatments, is refused at
    # construction rather than producing a plausible wrong number later.
    #
    # x2c-SVPall-2c is the Karlsruhe segmented family recontracted for two-component work.
    # It is a sensible default through radon. All-electron bases only -- X2C needs the core,
    # so an ECP basis is refused.
    neon = kuiva.Molecule(atoms=[("Ne", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c",
                          charge=0, spin=0)

    # ----------------------------------------------------------------------------------
    # 2. The scalar-relativistic SCF.
    # ----------------------------------------------------------------------------------
    # Every stage class obeys the same contract: the constructor takes the finished
    # upstream stage plus options and validates everything it can immediately, .run() is
    # the only expensive call and returns the stage itself, results are plain attributes,
    # and .summary() prints the headline numbers.
    #
    #   memory_gb   the working memory this calculation may commit to its own arrays. Kuiva
    #               has no built-in limit and never guesses one: with neither a value here
    #               nor a configured default it refuses to start. `source setup.sh` asks
    #               for the default once and writes it to ~/.config/kuiva/defaults.conf.
    #
    # Not passed, and worth knowing about because it costs something: `method` defaults to
    # "X2C-AMF", which means the ingested spin-orbit operator carries the two-electron
    # picture change -- one four-component atomic calculation per unique element, a fifth
    # of a second for neon and up to tens of minutes for a lanthanide, paid once ever and
    # cached on disk. It matters: without it, spin-orbit splittings come out 5 to 30 per
    # cent too large. `screening="none"` opts out where the cost is not worth it.
    scf = kuiva.ScalarSCF(neon, memory_gb=2.0).run()

    out.section(log, "The finished SCF stage")
    log.info("%s", scf.summary())

    # ----------------------------------------------------------------------------------
    # 3. The multireference starting point.
    # ----------------------------------------------------------------------------------
    # Orthonormalization, spinor expansion and integral factorization, in one stage. The
    # defaults are the production ones: canonical orthogonalization with a 1e-7 threshold
    # on the overlap eigenvalues, and a Cholesky decomposition of the two-electron
    # integrals at 1e-8. Cholesky rather than density fitting, because a Cholesky threshold
    # is an error bound you choose and a fitting error is not bounded at all.
    ref = kuiva.Reference(scf).run()

    out.section(log, "The finished Reference stage")
    log.info("%s", ref.summary())

    # The low-level container stays reachable, and is what the module drivers take. The
    # stage classes are a thin layer over the same functions, never a replacement for them.
    data = ref.reference.data           # the ingestion boundary: arrays, no PySCF objects
    orth = ref.reference.orth           # the orthonormal working basis
    spinors = ref.reference.spinors     # the Kramers-paired spinor guess
    factors = ref.reference.factors     # three-index two-electron factors

    # ----------------------------------------------------------------------------------
    # 4. Check what came out. Each of these is exact by construction, so a failure is a
    #    real defect rather than a tolerance question.
    # ----------------------------------------------------------------------------------
    out.section(log, "Checks on the ingested reference")

    # (a) The working basis is orthonormal: X^T S X = 1. This is the load-bearing boundary
    #     of the whole program -- every stage after it assumes an orthonormal basis -- so it
    #     is worth one matrix product. The round trip below uses the metric adjoint X^T S
    #     rather than X^T: coefficients and operators transform differently in a
    #     non-orthogonal basis, and going AO -> working -> AO must be the identity.
    ortho_err = float(np.max(np.abs(orth.x.T @ data.s_ao @ orth.x - np.eye(orth.nwork))))
    roundtrip = float(np.max(np.abs(orth.to_ao(orth.to_working(data.mo_coeff))
                                    - data.mo_coeff)))

    out.subsection(log, "Orthonormal working basis")
    out.entries(log, [
        ("max |X^T S X - 1|", ortho_err, "", "must be ~1e-14", out.SCI_FMT),
        ("orbital round-trip error", roundtrip, "", "AO -> working -> AO", out.SCI_FMT),
        ("directions dropped", orth.x.shape[1] - orth.nwork, "",
         "linear dependence removed at the threshold"),
    ])

    # (b) The spinor guess is exactly Kramers paired: column 2p+1 is the time reverse of
    #     column 2p, with T = -i sigma_y K. Kuiva's spinor conventions are fixed in
    #     kuiva/spinor/expand.py and nowhere else -- spin-blocked rows [alpha; beta],
    #     interleaved Kramers-pair columns -- because the CI's determinant addressing
    #     depends on them. T^2 = -1 is the statement that makes Kramers degeneracy a
    #     theorem for an odd electron count, so it is checked rather than assumed.
    c = spinors.c
    t2_err = float(np.max(np.abs(time_reverse(time_reverse(c)) + c)))

    out.subsection(log, "Spinor guess")
    out.entries(log, [
        ("spinors", spinors.nspinor, "", "{} Kramers pairs".format(spinors.nspinor // 2)),
        ("Kramers pairing deviation", spinors.partner_deviation(), "",
         "|barred - T(unbarred)|", out.SCI_FMT),
        ("max |T^2 C + C|", t2_err, "", "T^2 = -1", out.SCI_FMT),
        ("orthonormality error", spinors.orthonormality_error(), "", "", out.SCI_FMT),
    ])

    # (c) The two-electron integrals were factorized rather than stored: the four-index
    #     array is never held for the full orbital space, and blocks are assembled on
    #     demand. Rebuild a few AO integrals from the factors and compare against the exact
    #     ones. The error must sit at the decomposition threshold -- not merely be "small".
    from pyscf import ao2mo                      # front end only; not used past this point
    eri_exact = ao2mo.restore(1, data.eri, data.nao)
    l_square = factors.unpack(slice(None))                          # (naux, nao, nao)
    eri_fit = np.einsum("Pmn,Pkl->mnkl", l_square, l_square, optimize=True)
    cholesky_err = float(np.max(np.abs(eri_fit - eri_exact)))

    out.subsection(log, "Two-electron integrals")
    out.entries(log, [
        ("factorization route", factors.origin),
        ("Cholesky vectors", factors.naux, "",
         "{:.1f} per AO function".format(factors.naux / data.nao)),
        ("decomposition threshold", factors.tol, "Eh", "", out.SCI_FMT),
        ("largest neglected diagonal", factors.residual, "Eh", "", out.SCI_FMT),
        ("max |(pq|rs) rebuilt - exact|", cholesky_err, "Eh",
         "over all {n}^4 AO integrals".format(n=data.nao), out.SCI_FMT),
    ])

    # (c2) The same factorization without ever storing the integrals. `fitting=
    #      "cholesky-direct"` evaluates each column of two-electron integrals when the
    #      pivoting asks for it, instead of building the whole array first. That array grows
    #      as the fourth power of the basis, so on a large system it is what decides whether
    #      the calculation starts at all -- the pre-flight table printed by this second SCF
    #      no longer has a line for it. The threshold, the error bound and the pivot rule are
    #      the same, and so is the answer; what changes is only how the numbers are obtained.
    #
    #      One consequence to know about: this route decomposes inside the SCF stage, because
    #      that is the only point where the integrals can still be evaluated. `cholesky_tol`
    #      and `orbit_pivots` therefore belong to ScalarSCF here rather than to Reference.
    out.section(log, "The same integrals, never stored (fitting='cholesky-direct')")
    scf_direct = kuiva.ScalarSCF(neon, memory_gb=2.0, fitting="cholesky-direct").run()
    ref_direct = kuiva.Reference(scf_direct).run()
    direct = ref_direct.reference.factors

    l_direct = direct.unpack(slice(None))
    eri_direct = np.einsum("Pmn,Pkl->mnkl", l_direct, l_direct, optimize=True)
    direct_vs_exact = float(np.max(np.abs(eri_direct - eri_exact)))
    direct_vs_stored = float(np.max(np.abs(eri_direct - eri_fit)))

    out.subsection(log, "Integral-direct route against the stored one")
    out.entries(log, [
        ("integral array stored", "no" if scf_direct.data.eri is None else "yes", "",
         "the O(nao^4) array the other route holds"),
        ("Cholesky vectors", direct.naux, "",
         "same count as the stored route" if direct.naux == factors.naux else "DIFFERENT"),
        ("max |(pq|rs) rebuilt - exact|", direct_vs_exact, "Eh", "", out.SCI_FMT),
        ("max |(pq|rs) direct - stored|", direct_vs_stored, "Eh",
         "the two routes agree to machine precision", out.SCI_FMT),
    ])
    out.note(log, "the vectors themselves need not match element for element: any orthogonal")
    out.note(log, "mixing of them reproduces the same integrals, and the integrals are what")
    out.note(log, "every later stage contracts.")

    # (d) The one-electron Hamiltonian the multireference layer will use. With spin-orbit
    #     coupling ingested this is the full two-component X2C operator in the spin-blocked
    #     AO basis: complex, Hermitian, and not block diagonal in spin -- that off-diagonal
    #     part is the spin-orbit coupling. Note that the SCF used a *different* operator
    #     (the spin-free sfx2c1e one) on purpose: orbitals are a basis the CASSCF
    #     re-optimizes, so the scalar set is a guess, while the operator whose expectation
    #     value is the energy is this one.
    h_mo = transform_1e(ref.reference.h_one_electron(), ref.reference.spinors_in_ao())
    herm_err = float(np.max(np.abs(h_mo - h_mo.conj().T)))
    soc = data.soc

    out.subsection(log, "One-electron Hamiltonian (two-component)")
    out.entries(log, [
        ("dimension", "{0} x {0}".format(h_mo.shape[0]), "", "spinor basis"),
        ("hermiticity error", herm_err, "Eh", "", out.SCI_FMT),
        ("largest spin-free element", float(np.max(np.abs(soc.h_sf))), "Eh", "",
         out.SCI_FMT),
        ("largest spin-orbit element", float(np.max(np.abs(soc.w))), "Eh",
         "the whole reason for the two-component treatment", out.SCI_FMT),
    ])
    out.note(log, "the Hamiltonian provenance printed above travels with every stored")
    out.note(log, "product: a property file that does not say whether it was screened is")
    out.note(log, "not interpretable, the difference being 5-30% on every splitting.")

    # (e) What the orbitals are. A spinor has no isosurface, so the way to answer "is this
    #     the orbital I meant?" is a reduced population analysis -- which fraction of each
    #     spinor's density sits on which atom and in which angular-momentum shell. Reported
    #     per Kramers pair, because a single spinor's populations are basis dependent
    #     inside a degenerate manifold while the pair sum is not.
    ref.population_analysis(level="frontier", n_frontier=3)
    out.note(log, "the reduced ORBITAL populations are the robust number here. The Loewdin")
    out.note(log, "atomic CHARGES in the same table are the weakest quantity this code")
    out.note(log, "produces -- do not read one as an oxidation state.")

    # ----------------------------------------------------------------------------------
    # 5. Assert. An example that only prints numbers cannot fail, and one that cannot fail
    #    demonstrates nothing.
    # ----------------------------------------------------------------------------------
    checks = {
        "the SCF converged": bool(scf.converged),
        "the working basis is orthonormal": ortho_err < 1e-12,
        "orbitals survive the AO -> working -> AO round trip": roundtrip < 1e-10,
        "the spinor guess is exactly Kramers paired": spinors.partner_deviation() < 1e-14,
        "time reversal squares to -1": t2_err < 1e-14,
        "the Cholesky error respects its threshold": cholesky_err < 10.0 * factors.tol,
        "the integral-direct route stores no integral array": scf_direct.data.eri is None,
        "the integral-direct route keeps the same error bound":
            direct_vs_exact < 10.0 * direct.tol,
        "the two factorization routes give the same integrals": direct_vs_stored < 1e-12,
        "the two-component Hamiltonian is Hermitian": herm_err < 1e-10,
        "spin-orbit coupling was ingested": bool(data.has_soc),
    }
    failures = report(checks)

    timing.summary(log)
    res.summary(log)
    return 1 if failures else 0


def report(checks) -> int:
    """Print the check table and return the number of failures."""
    out.section(log, "Result")
    table = out.Table(log, [out.Column("check", "{}", 62, align="<"),
                            out.Column("verdict", "{}", 10, align="<")])
    table.start()
    for label, ok in checks.items():
        table.row(label, "ok" if ok else "FAILED")
    table.end()
    failed = [label for label, ok in checks.items() if not ok]
    for label in failed:
        log.error("check failed: %s", label)
    out.entry(log, "checks", "FAILED ({})".format(len(failed)) if failed
              else "all {} passed".format(len(checks)))
    return len(failed)


if __name__ == "__main__":
    raise SystemExit(main())
