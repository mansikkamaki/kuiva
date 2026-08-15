"""Example 8 -- Slater-Condon parameters and spin-orbit constants of a free atom.

    source setup.sh          # once per shell
    python slater_condon.py

Runs in about a minute on a scandium atom and writes ``output/slater_condon.out`` and the
machine-readable ``output/scandium.scp``.

WHAT THIS SHOWS
---------------
This is not part of the multireference pipeline. It is a *special-purpose method* that lives
beside it: given one atom or ion and a configuration you choose, it produces the radial
parameters that a semi-empirical model of that atom is written in --

    F^k(a,b)      the direct (Coulomb) parameters of a pair of shells
    G^k(a,b)      the exchange parameters
    R^k(ab;cd)    the general cross parameters, which have no shorter name
    zeta_nl       the one-electron spin-orbit constant of each shell

Crystal-field, ligand-field and multiplet models are usually parameterized against experiment.
Computing the same parameters ab initio says what they would be for a *specified* state of the
free ion, with no fitting, which is what makes them comparable with a fitted set.

The atom here is neutral scandium in its ground configuration, [Ar] 3d1 4s2, and the shells
asked for are 3p, 3d and 4s -- three different angular momenta, which is what makes every kind
of parameter appear at once. It is a cheap stand-in for the calculation the feature was built
for: a lanthanide such as Dy(I) [Xe] 4f9 5d1 6s1, where the shells are 4f, 5d and 6s and where
the four-component atomic solve behind the spin-orbit constants takes tens of minutes instead
of the fifty seconds it takes here.

WHY AN AVERAGE OF CONFIGURATION, AND NOT AN ORDINARY SCF
--------------------------------------------------------
A radial parameter belongs to a *shell*, so the calculation it comes from must have shells: an
ordinary open-shell SCF puts its unpaired electron into one particular orbital of a degenerate
set, which breaks the spherical symmetry of the atom and gives the three 3d orbitals three
different radial functions. The reference here spreads each open shell equally over all its
orbitals and minimizes the true configuration-average energy, so the solution is spherical and
each shell has one radial function. That is a well-defined state of its own -- deliberately
not the atom's ground state, which no single spherical ensemble is.

WHAT TO LOOK FOR IN THE OUTPUT
------------------------------
* the parameter table: seventeen parameters, and the seventeen are forced by angular momentum
  alone. F^1, F^3, ... never appear (parity), a d shell has F^0, F^2 and F^4 and no F^6, and
  the exchange parameter of a shell with itself is not listed because it *is* the direct one;
* the cross parameter R^2(3p 3d; 3p 4s). In the chemists' notation it is the integral
  (3p 3p | 4s 3d), and it survives at k = 2 and nowhere else. It is the structural twin of the
  (4f 4f | 6s 5d) term that a lanthanide model needs, which is why this example asks for three
  shells rather than two;
* R^1(3p 3p; 3d 4s) is **negative**. A cross parameter may be; F^0 may not, being the Coulomb
  repulsion of two densities;
* the spin-orbit constants: 3p and 3d have one, and the 4s shell is *absent from the table
  rather than listed as zero*, because l.s vanishes identically for an s shell and there is
  nothing to fit;
* the diagnostics printed beside the numbers, which answer different questions. The **shell
  anisotropy** says whether the converged solution was actually spherical; the **class
  residual** says whether the expansion the radial parameters are defined by reproduces the
  integrals. Neither implies the other, so both are printed and both are checked below. The
  spin-orbit fit carries a residual of its own, and it is checked against a *looser* bound
  on purpose: its floor is the X2C decoupling's rounding rather than the fit's, so it sits
  several orders higher and it moves with the linear-algebra library the machine has;
* the effect of the two-electron screening on zeta, computed twice here: the one-electron X2C
  operator alone makes spin-orbit splittings 5-30 per cent too large, and the atomic mean
  field supplies what it is missing.

READING THE NUMBERS
-------------------
Two rules bind any parameter quoted from a calculation like this one, and both have produced
convincing-looking errors elsewhere:

* **state which construction produced it.** These are frozen average-of-configuration orbitals
  of one fixed configuration. They are not self-consistent values for any particular term, and
  they contain no correlation. A parameter set fitted to experiment is not the same object:
  fitted parameters absorb correlation, and Hartree-Fock-level values are known to come out
  20-30 per cent high for that reason;
* **never compare against a value obtained in a different basis set.** No parameter can recover
  a basis truncation. The split-valence basis used here is chosen for speed.

The scandium 3d spin-orbit splitting is printed against experiment for scale only. The measured
a2D5/2 - a2D3/2 separation of Sc I is 168.34 cm^-1 (NIST Atomic Spectra Database); what comes
out below is roughly a quarter larger, which is what both rules above predict and is not a
failure of anything.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict

import kuiva
from kuiva.extras import read_parameters, slater_condon_parameters
from kuiva.util import output as out
from kuiva.util import timing
from kuiva.util.logging import add_file_handler, get_logger

NAME = "slater_condon"
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"

log = get_logger("examples." + NAME)

#: Neutral scandium in its ground configuration. The charge is derived from the electron
#: count, so it can never disagree with the configuration.
ELEMENT, CONFIGURATION = "Sc", "[Ar] 3d1 4s2"

#: The shells to compute parameters among. Three different angular momenta, which is what
#: produces the genuine cross parameters; the default would be the open shells alone (3d).
SHELLS = ("3p", "3d", "4s")

BASIS = "x2c-SVPall-2c"

#: Every parameter that exists for these three shells, worked out from the selection rules:
#: k must couple the two shells of each electron pair, with |l - l'| <= k <= l + l' and
#: l + l' + k even. Anything not in this set is identically zero and is not a parameter.
EXPECTED = {
    "F0(3p,3p)", "F2(3p,3p)", "F0(3p,3d)", "F2(3p,3d)", "F0(3p,4s)",
    "F0(3d,3d)", "F2(3d,3d)", "F4(3d,3d)", "F0(3d,4s)", "F0(4s,4s)",
    "G1(3p,3d)", "G3(3p,3d)", "G1(3p,4s)", "G2(3d,4s)",
    "R1(3p 3p;3d 4s)", "R2(3d 3d;3d 4s)", "R2(3p 3d;3p 4s)",
}

#: Sc I a2D5/2 - a2D3/2, NIST Atomic Spectra Database. An anchor, not a target; the file
#: header explains why a frozen average-of-configuration value in a split-valence basis is
#: expected to sit above it.
EXPERIMENT_CM = 168.34

#: An average-of-configuration SCF is constrained to spherical solutions, so its Fock operator
#: is spherical to about 1e-13, while a symmetry-broken one sits at 1e-1. The bound is placed
#: between them, far from both.
ANISOTROPY_BOUND = 1e-6

#: The expansion the radial parameters are defined by is exact, so its residual is roundoff.
#: Measured here: 1e-13 and below, on any machine.
RESIDUAL_BOUND = 1e-8

#: ⚠ The spin-orbit fit is a *different* diagnostic with a *different* floor, and it may not
#: share the bound above. Its floor is the X2C decoupling's own rounding rather than the fit,
#: so it sits far higher and it moves with the linear-algebra library: this scandium 3d shell
#: measures 5.5e-09 against Intel MKL and 1.5e-08 against a stock BLAS. A bound of 1e-8 across
#: both is a coin toss, which is exactly what an example may not assert.
ZETA_RESIDUAL_BOUND = 1e-6


def prepare_output() -> Path:
    """Write this run to output/<name>.out, with a scratch spin-orbit cache of its own.

    The four-component atomic solve behind the screened spin-orbit constants is cached on
    disk, and an example whose result depends on what is already in the developer's cache is
    not a demonstration of anything -- so this one clears its own cache and pays the fifty
    seconds every time.
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
    out.banner(log, kuiva.__version__, "example 8: Slater-Condon parameters (Sc 3d 4s)")
    out.entry(log, "output file", str(outfile.relative_to(HERE)))

    # ----------------------------------------------------------------------------------
    # 1. The calculation. One call: the average-of-configuration scalar X2C SCF, the shell
    #    extraction, the radial parameters, the spin-orbit constants, the report and the
    #    machine-readable file.
    #
    #    zeta=True (the default) is what makes this cost anything: it needs the
    #    two-component operator, whose default screening is one four-component atomic solve
    #    per element -- fifty seconds for scandium, tens of minutes for a lanthanide, paid
    #    once ever because it is cached. With zeta=False the whole run is under a second.
    # ----------------------------------------------------------------------------------
    result = slater_condon_parameters(
        ELEMENT, CONFIGURATION, basis=BASIS, shells=SHELLS, memory_gb=2.0, conv_tol=1e-11,
        file=OUTPUT / "scandium.scp",
        title="Sc I [Ar] 3d1 4s2 -- example 8")

    # ----------------------------------------------------------------------------------
    # 2. The same atom without the two-electron picture change, for the spin-orbit
    #    constants only. It needs no four-component solve, so it costs a second.
    # ----------------------------------------------------------------------------------
    unscreened = slater_condon_parameters(
        ELEMENT, CONFIGURATION, basis=BASIS, shells=SHELLS, memory_gb=2.0, conv_tol=1e-11,
        screening="none", report=False)

    out.section(log, "What the two-electron screening does to zeta")
    table = out.Table(log, [
        out.Column("shell", "{:s}", 6),
        out.Column("X2C-1e [cm^-1]", out.CM_FMT, 16),
        out.Column("X2C-AMF [cm^-1]", out.CM_FMT, 17),
        out.Column("change", "{:+.1f}%", 10),
    ])
    table.start("one-electron X2C against the default, which adds the atomic mean field")
    for constant in result.spin_orbit:
        bare = unscreened.spin_orbit[constant.shell].zeta_cm
        table.row(constant.shell, bare, constant.zeta_cm,
                  100.0 * (constant.zeta_cm / bare - 1.0))
    table.end("the screening always lowers a spin-orbit constant; the one-electron operator "
              "misses the shielding of the nucleus by the other electrons")

    d_shell = result.spin_orbit["3d"]
    out.entries(log, [
        ("3d spin-orbit constant", d_shell.zeta_cm, "cm^-1", "", out.CM_FMT),
        ("implied 2D5/2 - 2D3/2 splitting", d_shell.splitting_cm, "cm^-1",
         "(2l+1) zeta / 2, one electron, frozen orbitals", out.CM_FMT),
        ("experiment (NIST)", EXPERIMENT_CM, "cm^-1", "an anchor, not a target", out.CM_FMT),
    ])
    out.note(log, "a frozen average-of-configuration constant in a split-valence basis is")
    out.note(log, "expected above the measured splitting: the orbitals do not relax, there")
    out.note(log, "is no correlation, and a fitted parameter absorbs both.")

    # ----------------------------------------------------------------------------------
    # 3. The file. It is the product: an external model-fitting code reads it, so it
    #    carries its own conventions, its provenance and both units. Reading it back here
    #    is what makes it a demonstration rather than a claim.
    # ----------------------------------------------------------------------------------
    stored = read_parameters(OUTPUT / "scandium.scp")
    out.section(log, "The parameter file")
    out.entries(log, [
        ("file", "output/scandium.scp"),
        ("format version", stored["header"]["format_version"]),
        ("parameters stored", len(stored["parameters"])),
        ("spin-orbit constants stored", len(stored["zeta"])),
        ("screening recorded in the header", stored["header"]["spin_orbit_screening"]),
    ])
    out.note(log, "the header states the Condon-Shortley ordering of R^k(ab;cd) verbatim and")
    out.note(log, "carries the full provenance record, because a stored parameter that does")
    out.note(log, "not say which Hamiltonian produced it is not interpretable.")

    # ----------------------------------------------------------------------------------
    # 4. Assert. Everything here is a selection rule, an inequality that physics forces, or
    #    a round trip -- nothing is a tolerance fitted to what this code happens to print.
    # ----------------------------------------------------------------------------------
    values = result.parameters.as_dict()
    checks: Dict[str, bool] = {}
    checks["the parameter set is exactly what the selection rules allow"] = (
        set(values) == EXPECTED)
    checks["no F^k of odd k exists (parity)"] = not [
        p for p in result.parameters if p.kind == "F" and p.k % 2]
    checks["a d shell has F^0, F^2, F^4 and no F^6"] = "F6(3d,3d)" not in values
    checks["the exchange parameter of a shell with itself is not listed separately"] = not [
        p for p in result.parameters if p.kind == "G" and p.shells[0] == p.shells[1]]
    checks["the (4f 4f | 6s 5d) analogue survives at exactly one k"] = (
        "R2(3p 3d;3p 4s)" in values
        and [p.k for p in result.parameters if p.shells == ("3p", "3d", "3p", "4s")] == [2])

    checks["every F^0 is positive (a Coulomb repulsion of two densities)"] = all(
        p.value > 0.0 for p in result.parameters.of_kind("F") if p.k == 0)
    checks["F^0 > F^2 > F^4 > 0 within the 3d shell"] = (
        values["F0(3d,3d)"] > values["F2(3d,3d)"] > values["F4(3d,3d)"] > 0.0)
    checks["exchange is smaller than the direct parameter of the same pair"] = (
        values["G2(3d,4s)"] < values["F0(3d,4s)"])
    checks["a cross parameter is free to be negative"] = values["R1(3p 3p;3d 4s)"] < 0.0

    checks["the 4s shell has no spin-orbit constant at all"] = (
        {c.shell for c in result.spin_orbit} == {"3p", "3d"})
    checks["both constants are positive and the core shell's is far larger"] = (
        0.0 < result.spin_orbit["3d"].zeta < result.spin_orbit["3p"].zeta)
    checks["the two-electron screening lowers every constant"] = all(
        c.zeta < unscreened.spin_orbit[c.shell].zeta for c in result.spin_orbit)

    checks["the SCF converged and the solution is spherical"] = (
        result.data.converged and result.anisotropy < ANISOTROPY_BOUND)
    checks["every class is reproduced by its radial parameters to roundoff"] = (
        result.parameters.max_relative_residual < RESIDUAL_BOUND)
    checks["the spin-orbit fits sit at their own floor"] = (
        result.spin_orbit.max_relative_residual < ZETA_RESIDUAL_BOUND)
    checks["the file round-trips at full precision"] = all(
        abs(stored["parameters"][label]["value"] - value) < 1e-12
        for label, value in values.items())
    checks["the file records which screening produced its constants"] = (
        stored["header"]["spin_orbit_screening"] == "x2camf")

    failures = report(checks)
    timing.summary(log)
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
