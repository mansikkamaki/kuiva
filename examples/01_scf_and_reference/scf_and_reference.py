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
* a third run with ``factors="scratch"``: the finished factor rows spilled to a scratch file
  and streamed back, so the later stages get their memory without the rows changing by a bit
  (the default, ``factors="auto"``, does this exactly when the in-core plan does not fit);
* the nuclear charge model, the third choice behind the Hamiltonian's name: a point charge
  by default, a finite (Gaussian) distribution on request, and what the difference is worth
  on an atom as light as neon;
* the SCF's convergence controls, which is where a real calculation first stops: an SCF that
  runs out of cycles **refuses** (everything downstream is built on its orbitals), the
  internal stability analysis says whether the converged solution is a minimum at all, and
  ``guess_from=`` starts one SCF from another's orbitals -- projecting them when the basis
  differs, through the same projector the CASSCF basis projection uses;
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
    #      You rarely need to pick: the default (`fitting="auto"`) reads the memory
    #      pre-flight and takes the stored route wherever its plan fits the configured limit
    #      -- as it did for the first SCF above -- and this direct route where it does not,
    #      saying so on the output's "two-electron route" line. The two routes cost the same
    #      processor time to within a few per cent on anything beyond ~160 basis functions,
    #      so the array is the entire decision. Passing `fitting=` explicitly, as done here
    #      for the demonstration, pins the route regardless of the plan.
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

    # (c3) Where the finished factor rows live is a second, independent choice. `factors=
    #      "scratch"` spills them to a file in the configured scratch directory the moment
    #      the decomposition ends, and every consumer streams them back in the same
    #      sequential blocks it always read -- freeing their memory for the CI workspace and
    #      the orbital-optimization blocks of the later stages. The default
    #      (`factors="auto"`) does this exactly when the in-core plan exceeds the memory
    #      limit, announced on the output's own residence line; here it is forced for the
    #      demonstration. The rows are the same rows: what changes is where they wait, and
    #      the pre-flight table shows the factor line as a transient of the two-electron
    #      phase instead of a resident carried into every later one.
    out.section(log, "The same factors, resident on scratch (factors='scratch')")
    scf_spill = kuiva.ScalarSCF(neon, memory_gb=2.0, fitting="cholesky-direct",
                                factors="scratch").run()
    spilled = kuiva.Reference(scf_spill).run().reference.factors
    l_spill = spilled.unpack(slice(None))
    spill_vs_direct = float(np.max(np.abs(l_spill - l_direct)))
    out.entries(log, [
        ("factor rows resident in memory", "no" if spilled.is_spilled else "yes", "",
         "streamed from scratch in sequential blocks"),
        ("max |L spilled - L in-core|", spill_vs_direct, "", "same rows, read from disk",
         out.SCI_FMT),
    ])

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
    out.note(log, "the reduced ORBITAL populations are the robust number here. No atomic")
    out.note(log, "CHARGE is printed, deliberately: the Loewdin charge was measured with the")
    out.note(log, "wrong sign on ionic textbook compounds, and was withdrawn from the report.")

    # ----------------------------------------------------------------------------------
    # 5. When the SCF will not converge -- which on a real open-shell metal complex is
    #    where a calculation first stops, and neon is far too well behaved to show it. What
    #    is demonstrated here is the surface, on a system where the answers are known.
    # ----------------------------------------------------------------------------------
    out.section(log, "The SCF's convergence controls")

    # (a) An SCF that runs out of cycles REFUSES. Everything downstream is built on these
    #     orbitals, and an unconverged iteration returns whichever step the budget stopped
    #     on -- so "the CASSCF will re-optimize them anyway" is a hope, not a property. The
    #     message names the levers; `allow_unconverged_scf=True` proceeds deliberately.
    #
    #     The levers themselves, in the order worth trying them: `level_shift=` (an energy
    #     added to the virtual orbitals, which stops the occupations swapping back and
    #     forth), `damp=` (mix in the previous Fock matrix), `diis="adiis"` (the energy-based
    #     DIIS variant, much better in the first iterations of a hard open-shell case), and
    #     `second_order=True` (the CIAH Newton solver, which converges cases the first-order
    #     iteration cannot -- at a higher cost per iteration and far fewer of them).
    refused = ""
    try:
        kuiva.ScalarSCF(neon, memory_gb=2.0, max_cycle=2).run()
    except RuntimeError as exc:
        refused = str(exc).split("(")[0].strip()     # not split(".") -- the energy has one

    out.subsection(log, "An SCF that did not converge")
    out.entries(log, [("max_cycle", 2, "", "deliberately far too few"),
                      ("outcome", "refused"),
                      ("message", refused)])

    # (b) Is the converged solution a minimum at all? `stability="check"` runs the internal
    #     stability analysis; `stability="follow"` rotates into the unstable mode and
    #     re-solves. A closed-shell neon atom is stable, and saying so costs one Davidson --
    #     but an open-shell transition metal can converge, report every diagnostic clean and
    #     still be sitting on a saddle point of the SCF energy, tenths of an Eh above the
    #     solution one rotation away. Nothing else in the front end can see that.
    #
    #     It rides along on the larger-basis calculation below rather than costing an SCF of
    #     its own -- which is how it is meant to be used: on the run you were doing anyway.
    large = kuiva.Molecule(atoms=[("Ne", (0.0, 0.0, 0.0))], basis="x2c-TZVPall-2c")
    cold = kuiva.ScalarSCF(large, memory_gb=2.0, stability="check").run()

    out.subsection(log, "Internal stability of the converged solution")
    out.entries(log, [
        ("stability analysis", "internal", "", "external is a different question"),
        ("verdict", "stable" if cold.stable else "UNSTABLE"),
        ("energy", cold.energy, "Eh", "unchanged by the check", out.E_FMT),
    ])
    out.note(log, "`stable` is None when the analysis was not asked for, which is not the")
    out.note(log, "same as True -- compare against True and False rather than truthiness.")

    # (c) Starting from another calculation's orbitals. Over the same AO basis they are used
    #     as they are (the potential-energy-surface case: the geometry may differ); over a
    #     different basis they are projected onto it, through the same projector the CASSCF
    #     basis projection uses. It is a guess: it changes the cost, and it may not change
    #     the answer -- which is what the check below asserts, rather than the saving.
    warm = kuiva.ScalarSCF(large, memory_gb=2.0, guess_from=scf).run()

    out.subsection(log, "The same SCF from a smaller basis' orbitals")
    out.entries(log, [
        ("small-basis energy (SVP)", scf.energy, "Eh", "the orbitals being carried",
         out.E_FMT),
        ("cold start (TZVP)", cold.energy, "Eh", "", out.E_FMT),
        ("from the projected guess", warm.energy, "Eh", "", out.E_FMT),
        ("difference", abs(warm.energy - cold.energy), "Eh",
         "a guess may change the cost and may not change the answer", out.SCI_FMT),
    ])

    # ----------------------------------------------------------------------------------
    # 6. The nuclear charge model. The Hamiltonian's name ("X2C-AMF") fixes two things --
    #    how the four-component problem is decoupled and which two-electron screening is in
    #    it -- and says nothing about the third: what the nucleus is. Kuiva's default is a
    #    point charge, which is what every reference number shipped with it was produced
    #    with; the alternative is the finite Gaussian distribution of Visscher and Dyall
    #    (At. Data Nucl. Data Tables 67, 207 (1997)).
    # ----------------------------------------------------------------------------------
    # One statement for the whole molecule, and every consumer inherits it: the molecular
    # integrals, the four-component atomic solve behind the spin-orbit screening, the
    # free-atom reference orbitals, the fragments of a local decoupling. Each of them reads
    # the model off the built molecule rather than being handed it separately -- an atomic
    # mean field solved over a different nucleus from the integrals it corrects would be
    # Hermitian, of entirely plausible size, and wrong.
    #
    # It costs an atomic four-component solve again, because the model is part of the cache
    # key: a lanthanide will pay its tens of minutes a second time. Neon does not notice.
    finite = kuiva.Molecule(atoms=[("Ne", (0.0, 0.0, 0.0))], basis="x2c-SVPall-2c",
                            charge=0, spin=0, nuclear_model="gaussian")
    fscf = kuiva.ScalarSCF(finite, memory_gb=2.0).run()

    out.section(log, "The nuclear charge model")
    h_point = scf.data.soc.h_sf
    h_shift = (np.max(np.abs(h_point - fscf.data.soc.h_sf))
               / np.max(np.abs(h_point)))
    w_shift = (np.max(np.abs(scf.data.soc.w - fscf.data.soc.w))
               / np.max(np.abs(scf.data.soc.w)))

    out.entries(log, [
        ("default model", scf.data.soc.nuclear.label()),
        ("this run's model", fscf.data.soc.nuclear.label()),
        ("point-nucleus SCF energy", scf.energy, "Eh", "", out.E_FMT),
        ("finite-nucleus SCF energy", fscf.energy, "Eh", "", out.E_FMT),
        ("difference", fscf.energy - scf.energy, "Eh",
         "positive: a spread-out nucleus attracts less", out.SCI_FMT),
        ("shift in the spin-free operator", h_shift, "", "relative", out.SCI_FMT),
        ("shift in the spin-orbit operator", w_shift, "", "relative", out.SCI_FMT),
    ])
    out.note(log, "The effect is concentrated at the nucleus and grows steeply with Z: it is")
    out.note(log, "beyond any comparison at neon and reaches ~3e-3 of a mercury j-splitting.")
    out.note(log, "It is the first thing to match against a four-component program, several")
    out.note(log, "of which default to a Gaussian nucleus where Kuiva defaults to a point.")
    out.note(log, "The model is written into the Hamiltonian's provenance, so a property")
    out.note(log, "file made from this run says which nucleus it describes.")

    # ----------------------------------------------------------------------------------
    # 7. Assert. An example that only prints numbers cannot fail, and one that cannot fail
    #    demonstrates nothing.
    # ----------------------------------------------------------------------------------
    checks = {
        "the SCF converged": bool(scf.converged),
        "an unconverged SCF refuses": bool(refused),
        "the neon SCF solution is internally stable": cold.stable is True,
        "a projected guess reaches the same solution as a cold start":
            abs(warm.energy - cold.energy) < 1e-9,
        "the working basis is orthonormal": ortho_err < 1e-12,
        "orbitals survive the AO -> working -> AO round trip": roundtrip < 1e-10,
        "the spinor guess is exactly Kramers paired": spinors.partner_deviation() < 1e-14,
        "time reversal squares to -1": t2_err < 1e-14,
        "the Cholesky error respects its threshold": cholesky_err < 10.0 * factors.tol,
        "the default route resolution stored the integrals here":
            scf.data.fit_route == "conventional",
        "the integral-direct route stores no integral array": scf_direct.data.eri is None,
        "the integral-direct route keeps the same error bound":
            direct_vs_exact < 10.0 * direct.tol,
        "the two factorization routes give the same integrals": direct_vs_stored < 1e-12,
        "spilled factors are read back as the rows that were written":
            spilled.is_spilled and spill_vs_direct < 1e-12,
        "the two-component Hamiltonian is Hermitian": herm_err < 1e-10,
        "spin-orbit coupling was ingested": bool(data.has_soc),
        "the default nuclear model is a point charge":
            scf.data.soc.nuclear.model == "point",
        "a finite nucleus is recorded in the Hamiltonian's provenance":
            fscf.data.soc.provenance()["nuclear"]["model"] == "gaussian",
        "a finite nucleus raises a light atom's energy, slightly":
            0.0 < fscf.energy - scf.energy < 1e-4,
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
