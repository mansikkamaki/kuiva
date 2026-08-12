"""Neutral vs ionic atomic reference: the X2CAMF sensitivity study.

Why this is a deliverable and not an optional extra
---------------------------------------------------
X2CAMF takes its two-electron picture change from an **isolated atom**. Every target system in
The target system class is an *ion in a ligand field*, so the obvious objection is that the atomic mean
field is taken over the wrong state. An in-house Breit-Pauli AMFI was rejected partly on
the grounds that this question is answerable *within* X2CAMF by varying the reference
configuration — which makes measuring it an obligation, not a curiosity. This script is that
measurement.

⚠ What it does and does not bound
----------------------------------
It brackets the effect of the reference **configuration**, from the neutral atom to the free
ion. It does **not** bound the effect of the ligand field, which no atomic calculation can
reach; the free ion is the more extreme of the two ends, so the spread measured here is an
upper bound on the configuration sensitivity and a *lower* bound on nothing at all. Read it as
"how much does this choice matter", not as an error bar on the method.

Three numbers are recorded per ion, and the third is the one that matters:

* ``max |dh_sf|`` and ``max |dw|`` — the size of the correction itself. Matrix-element maxima,
  so they are basis-dependent and only comparable between two runs in the *same* basis (which
  these are).
* the **valence j-splitting** from a self-consistent two-component SCF of the ion, corrected
  each way. This is an observable, and it is what a cross-code comparison would actually be made on.

⚠ Cost — and why contracting is **not** the escape hatch it is for a closed shell
-----------------------------------------------------------------------------------
The uncontracted four-component atomic solve grows as roughly ``n4c^4``, and
the lanthanides are the expensive end of that curve rather than more of the same (Xe alone is
177 s wall / 1390 s CPU). The cost study recorded ``uncontract=False`` as the dramatic way out —
contracted Kr with Gaunt runs in seconds against 36 s uncontracted.

**That does not carry over to an open shell, and this file originally claimed it did.**
Measured on Ti(3+): 187 s CPU uncontracted (``n4c`` 396) against 196 s CPU contracted
(``n4c`` 168) — the same cost from a 2.4x smaller problem, because average of configuration in
the contracted basis needs far more SCF cycles. Ce(3+) was killed after 1130 s wall / 4400 s
CPU still inside its first *contracted* solve.

So contracting buys nothing here while costing decoupling accuracy (the
per-family measurements
measured that at -0.5% to -1.7% on a splitting for calcium), and the default is
**uncontracted**. ``--contracted`` remains available, but as a comparison rather than as a
saving. Judge either by ``cpu_seconds``, never by wall time.

Every ion is bounded and every record is written as soon as it exists.

Run:  python tests/generate/amf_sensitivity.py [--only Ce,Dy] [--contracted]
      (with ``external/env.sh`` sourced). Writes ``tests/reference/amf_sensitivity.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                          # noqa: E402
from pyscf import gto                                                       # noqa: E402

import thermal                                                              # noqa: E402
from kuiva.amf import amf_correction                                        # noqa: E402
from kuiva.amf.atomic import clear_cache                                    # noqa: E402

OUT = REPO / "tests/reference/amf_sensitivity.json"
BASIS = "x2c-SVPall-2c"
HARTREE_CM = 219474.6313632


@dataclass(frozen=True)
class Ion:
    """One species, with the two reference configurations to compare.

    ⚠ **The shell under test is derived from the configuration, never from an index.** The
    observable is the j-splitting of the ion's *open* shell, which is the highest-energy
    occupied manifold and spans all ``4l+2`` of its spinors because average of configuration
    populates the whole of it. Hard-coding "the top six" or an electron-count offset gets this
    wrong in two different ways: Ti(3+) has 19 electrons but 28 partly occupied spinors, and
    the manifold of interest is 10 wide, not 6. This is the rule that an active
    space must be stated in physical terms, applied to an observable.
    """

    symbol: str
    charge: int
    ionic: str

    @property
    def probe_shell(self) -> Tuple[int, int]:
        """``(l, electrons)`` of the shell whose j-splitting is the observable.

        The ion's open shell where it has one. ⚠ Where it does **not** — Lu(3+) is [Xe]4f14,
        closed — the frontier shell of the highest occupied channel is used instead, which for
        Lu(3+) is the full 4f manifold and whose ``f_5/2``-``f_7/2`` splitting is exactly the
        quantity of interest. A closed-shell ion is not a case to skip here: Lu(3+) is the
        *cleanest* test of the f-block default precisely because no average of configuration
        is involved on either side.
        """
        from kuiva.amf.configuration import AtomicConfiguration

        config = AtomicConfiguration.parse(self.ionic)
        open_shells = config.open_shells()
        if len(open_shells) > 1:
            raise ValueError("{}{:+d} has {} open shells; the observable is ambiguous".format(
                self.symbol, self.charge, len(open_shells)))
        if open_shells:
            return open_shells[0]
        l = max(i for i, n in enumerate(config.occupations) if n)
        return l, 4 * l + 2

    @property
    def label(self) -> str:
        from kuiva.amf.configuration import SHELL_LETTERS
        l, q = self.probe_shell
        return "{}^{}".format(SHELL_LETTERS[l], q)


#: The sensitivity study covers Dy(3+) and Ce(3+). Ti(3+) is the cheap control — the ion
#: behind the reference systems ``ticl3``/``ti2cl6``, affordable enough to re-run after any change.
#:
#: ⚠ **Lu is the decisive case for the f-block default and is listed first among the
#: lanthanides for that reason.** Lu(3+) is [Xe]4f14, i.e. **closed shell**, so its reference
#: needs no average of configuration at all and converges like a noble gas; neutral Lu is
#: [Xe]4f14 5d1 6s2, a single open ``d`` shell. That makes the neutral-vs-trivalent comparison
#: for Lu the cleanest measurement of what the default costs — no open-shell machinery on
#: either side of it, and nothing else varying.
IONS: Tuple[Ion, ...] = (
    Ion("Ti", 3, "[Ar]3d1"),
    Ion("Lu", 3, "[Xe]4f14"),
    Ion("Ce", 3, "[Xe]4f1"),
    Ion("Dy", 3, "[Xe]4f9"),
    Ion("Yb", 3, "[Xe]4f13"),
)


def spectrum_of(mol, configuration, correction=None):
    """Occupied two-component spinor energies [Eh], ascending, for an **open-shell ion**.

    ⚠ The closed-shell construction of ``tests/test_amf_correction.self_consistent_spectrum``
    cannot be used here and does not fail quietly: a plain aufbau GHF on an open-shell ion
    picks arbitrarily among a degenerate frontier manifold and does not converge (measured on
    Ti(3+): the assertion fires rather than a wrong number coming back, which is the good
    outcome). The ion's spectrum has to be computed in the **same averaged state** the mean
    field was taken over, so this goes through ``average_of_configuration_ghf`` — the same
    two-component AOC SCF the open-shell energy functional uses, and through the same shared
    filling rule as the four-component side.

    ⚠ **A failure here is usually not a failure here.** The angular-momentum assignment inside
    that SCF is a population analysis of its own orbitals, so a corrupted ``hcore`` makes it
    report something like *"the s channel offers 8 orbitals"* on a basis that plainly has 14.
    That is what a broken correction looks like one level downstream, and it happened: before
    the metric threshold was shared with the four-component solve (open-shell work), Ce's
    correction carried 473 Eh of time-reversal-odd noise and this SCF was the thing that
    complained. ``amf_atomic_correction`` now refuses such a correction outright, so the error
    arrives where the cause is.
    """
    from pyscf.x2c import x2c

    from kuiva.spinor.expand import decompose_two_component, two_component_operator
    from test_amf_open_shell import average_of_configuration_ghf

    helper = x2c.SpinOrbitalX2CHelper(mol)
    h = two_component_operator(*decompose_two_component(np.asarray(helper.get_hcore())))
    if correction is not None:
        h = h + correction.hamiltonian()
    mf = average_of_configuration_ghf(mol, configuration, h)
    if not mf.converged:
        raise RuntimeError("the two-component AOC SCF did not converge")
    occ = np.asarray(mf.mo_occ)
    return np.sort(np.asarray(mf.mo_energy)[occ > 1e-12])


def measure(ion: Ion, uncontract: bool, checkpoint=None) -> Dict:
    """Measure one ion. ``checkpoint(record)`` is called after **each reference**, not only at
    the end, because a lanthanide needs two four-component solves of ~35 minutes each and a run
    stopped between them must still yield the first. Writing per *ion* was
    not enough — that was learned by losing 50 minutes of Ce to a timeout."""
    from kuiva.amf.configuration import AtomicConfiguration

    mol = gto.M(atom=[(ion.symbol, (0.0, 0.0, 0.0))], basis=BASIS, charge=ion.charge,
                spin=(int(gto.charge(ion.symbol)) - ion.charge) % 2, verbose=0)
    l, _ = ion.probe_shell
    width = 4 * l + 2
    # The *ion's own* configuration, which is what its two-component spectrum is computed in.
    # It is not the same thing as the reference configuration the mean field is taken over —
    # that is exactly the variable this study sweeps.
    state = AtomicConfiguration.parse(ion.ionic)
    record: Dict = {"element": ion.symbol, "charge": ion.charge, "basis": BASIS,
                    "uncontracted": uncontract, "valence_shell": ion.label,
                    "state": state.canonical, "manifold_width": width}

    def splitting(correction=None):
        """j-splitting [cm^-1] of the open shell: the spread of its ``4l+2`` spinors, which
        are the highest-energy occupied ones because average of configuration populates the
        whole manifold."""
        e = spectrum_of(mol, state, correction)[-width:]
        return float((e[-1] - e[0]) * HARTREE_CM)

    # ⚠ The neutral reference is named explicitly rather than left to the default. Since the
    # f-block default became the trivalent ion, passing ``None`` here would compare the ionic
    # reference with itself — a study that reports 0.00% for every lanthanide and looks like a
    # result. The whole point is that these two are different calculations.
    record["uncorrected_splitting_cm"] = splitting()
    if checkpoint:
        checkpoint(record)
    for name, configuration in (("neutral", AtomicConfiguration.ground(ion.symbol)),
                                ("ionic", ion.ionic)):
        correction = amf_correction(mol, method="x2camf", configuration=configuration,
                                    uncontract=uncontract)
        record[name] = {
            "configuration": correction.configurations[ion.symbol],
            "max_dh_sf": correction.spin_free_scale,
            "max_dw": correction.spin_orbit_scale,
            "splitting_cm": splitting(correction),
        }
        if checkpoint:
            checkpoint(record)

    a, b = record["neutral"], record["ionic"]
    record["sensitivity"] = {
        "dw_relative": abs(b["max_dw"] - a["max_dw"]) / (a["max_dw"] or 1.0),
        "dh_sf_relative": abs(b["max_dh_sf"] - a["max_dh_sf"]) / (a["max_dh_sf"] or 1.0),
        "splitting_relative": abs(b["splitting_cm"] - a["splitting_cm"])
        / (abs(a["splitting_cm"]) or 1.0),
        "splitting_difference_cm": b["splitting_cm"] - a["splitting_cm"],
    }
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated element symbols")
    ap.add_argument("--contracted", action="store_true",
                    help="decouple in the basis as given instead of the primitive one. ⚠ For "
                         "an open shell this is NOT cheaper (module docstring) and it costs "
                         "decoupling accuracy; it is a comparison, not a "
                         "saving")
    ap.add_argument("--max-wall", type=float, default=540.0,
                    help="total wall budget [s]; no new ion is started past it")
    args = ap.parse_args(argv)

    wanted = {s.strip().capitalize() for s in args.only.split(",") if s.strip()}
    uncontract = not args.contracted
    document: Dict = {
        "schema": 1,
        "generator": "tests/generate/amf_sensitivity.py",
        "purpose": ("how much the X2CAMF correction depends on the atomic reference "
                    "configuration, neutral atom vs free ion"),
        "basis": BASIS,
        "uncontracted": uncontract,
        "construction": ("splittings from a self-consistent two-component SCF; see "
                         "stated: the frozen-orbital construction gives a different "
                         "number for the same operator"),
        "environment": thermal.describe_environment(),
        "records": {},
    }
    if OUT.is_file():
        try:
            document["records"] = json.loads(OUT.read_text()).get("records", {})
        except ValueError:
            pass
    OUT.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for ion in IONS:
        if wanted and ion.symbol not in wanted:
            continue
        elapsed = time.time() - started
        if elapsed > args.max_wall:
            print("[budget] {:.0f} s of {:.0f} s used; stopping before {}".format(
                elapsed, args.max_wall, ion.symbol), flush=True)
            break
        key = "{}{:+d}/{}".format(ion.symbol, ion.charge,
                                  "uncontracted" if uncontract else "contracted")
        clear_cache()

        def save(partial, _key=key):
            """Persist what exists so far, marked incomplete until the ion finishes."""
            document["records"][_key] = dict(partial, status="partial")
            OUT.write_text(json.dumps(document, indent=2, sort_keys=True))
            print("   [checkpoint] {} {}".format(
                _key, ", ".join(k for k in ("neutral", "ionic") if k in partial) or "state"),
                flush=True)

        try:
            with thermal.track_resources() as res:
                record = measure(ion, uncontract, checkpoint=save)
            record["resources"] = res.as_dict()
            record["status"] = "ok"
        except Exception as exc:                                            # noqa: BLE001
            record = {"element": ion.symbol, "charge": ion.charge,
                      "status": "{}: {}".format(type(exc).__name__, str(exc)[:200])}
        document["records"][key] = record
        OUT.write_text(json.dumps(document, indent=2, sort_keys=True))
        if record["status"] == "ok":
            s = record["sensitivity"]
            print("{:22s} {} splitting  uncorrected {:9.2f}  neutral {:9.2f}  ionic {:9.2f} cm^-1  "
                  "({:+.2f}%)   max|dw| {:+.2f}%   {}".format(
                      key, record["valence_shell"], record["uncorrected_splitting_cm"],
                      record["neutral"]["splitting_cm"],
                      record["ionic"]["splitting_cm"], 100 * s["splitting_relative"],
                      100 * s["dw_relative"], record["resources"].get("summary", "")),
                  flush=True)
        else:
            print("{:22s} {}".format(key, record["status"]), flush=True)

    print("\nwrote {}".format(OUT.relative_to(REPO)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
