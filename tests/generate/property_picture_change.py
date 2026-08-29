"""Characterize the X2C picture change on the magnetic moment operator.

**What this decides.** The property operators are the bare non-relativistic ``L`` and ``S`` by
default — the same choice OpenMolcas RASSI makes, which is what keeps a cross-code comparison of a
property dump like-for-like — and the size of that approximation has never been measured. This
script measures it, and the answer decides whether the correction (``property_picture_change=True``,
v0.7.0) is worth having at all.

⚠ **The thresholds were fixed before any number existed**, so a verdict cannot be argued into
existence afterwards. The correction is worth implementing as a supported option if *any* of:

* a phase-invariant g value moves by more than **1 %** relative (``G_LANDE_RTOL``, the band the
  suite asserts free-ion Lande factors at);
* a free-ion ``2J+1`` degeneracy moves by more than **0.05 cm^-1** (``FREE_ION_DEGENERACY_TOL_CM``,
  asserted as a *physical* requirement rather than a tolerance);
* the TiCl3 ground-doublet ``g_perp``/``g_par`` moves by more than **0.3 %**, the margin at which
  the local validation record currently agrees with OpenMolcas.

A result below all three is a result, not a failure: it converts a standing "unmeasured" warning
into a bound, and it rules the picture change out as the cause of the ~0.5 % Dy(3+) Lande residual.

**The design that makes this cheap and confound-free: one calculation, two operators.** The
correction changes no wavefunction — same orbitals, same CI vectors, same states — so both variants
are evaluated on a single converged run and the difference is *exactly* the operator. Nothing here
re-converges anything to change a property.

Five measurements, in increasing distance from a matrix norm:

1. **Operator norms**, projected onto the **occupied+active** spinor block. ⚠ **Never global**: the
   global norm is core-dominated, is *non-monotonic* in Z (12 % Ne, 55 % Xe, 40 % Rn), and moves by
   8x on one molecule from decontracting the basis alone, which changes no physics. It measures the
   basis representation, not the observable.
2. **The np^1 isoelectronic series** B -> Al -> Ga -> In -> Tl. One valence p electron throughout, so
   every g factor is purely angular and the analytic Lande values 2/3 and 4/3 hold exactly; the
   picture-change shift is therefore read straight off the deviation, with no orbital or basis
   confound in between. ⚠ **The point is the Z trend, not any one atom**: a correction that does not
   grow smoothly with Z is an implementation error, and a single atom could never say so.
3. **4f, zero confound**: Ce(3+) 4f^1 and Yb(3+) 4f^13 — one electron against one hole, where the
   multiplet order must invert. Same argument as (2) and a different shell.
4. **Dy(3+) 4f^9**, the many-electron f shell: does the correction explain the ~0.5 % Lande residual
   recorded for it? A one-electron correction on the same shell should give a shift close to Ce and
   Yb's, so (3) and (4) check each other.
5. **TiCl3**, the one complex: the ground-doublet g values and the five-doublet pattern, against the
   numbers the local validation record already holds.

⚠ **Reductions only.** Degeneracy patterns, relative energies, and ``Tr_block(mu_i mu_j)`` with its
principal g values. No matrix element of ``mu`` is a number in this study, because the dump fixes no
phase convention and degenerate states mix arbitrarily.

⚠ **State the construction with every number and never compare across bases.** Everything here is
``x2c-SVPall-2c`` at SA-CASSCF over the stated roots. The screening is recorded per system: the
np^1 series and the 4f ions run ``screening="none"`` so the run needs no four-component atomic solve
(and Ce(3+) is run *both* ways as a control on whether the shift depends on that choice at all),
while TiCl3 keeps the default ``x2camf``.

Run:  ``python tests/generate/property_picture_change.py [--out temp/property_picture_change.json]``
      ``... [--only b,al,ga] [--skip dy3p]``
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kuiva.interface import api                                          # noqa: E402
from kuiva.props.dump import property_matrices                           # noqa: E402
from kuiva.props.multiplet import G_ELECTRON, lande_g                    # noqa: E402
from tests.generate import systems                                       # noqa: E402
from tests.generate.progress import Heartbeat                            # noqa: E402

#: ⚠ Hard wall budget for the whole run. Every result is on disk the moment it is computed
#: (the file is rewritten after each system), so a kill costs only the system in flight.
WALL_BUDGET_S = 55 * 60

#: Per-system wall budget. Cheapest first, so a killed run still yields the cheap ones.
SYSTEM_BUDGET_S = 20 * 60

#: The decision thresholds of the module docstring, carried into the output so the verdict is
#: recorded beside the data it was taken from rather than in someone's memory.
THRESHOLDS = {"g_relative": 0.01, "degeneracy_cm": 0.05, "ticl3_g_relative": 0.003}


@dataclasses.dataclass
class Case:
    """One system and the protocol applied to it. ``analytic`` is ``(l, s)`` when the levels have
    a closed-form Lande factor, i.e. wherever the active space holds one electron or one hole."""

    key: str
    screening: str
    n_states: int
    analytic: Optional[tuple] = None
    label_suffix: str = ""

    @property
    def name(self) -> str:
        return self.key + self.label_suffix


#: ⚠ Ordered by measured cost, cheapest first. ``dy3p`` is last because it is the only one that
#: can plausibly exceed the per-system budget: 2002 determinants over 134 state-averaged roots.
CASES: List[Case] = [
    Case("b", "none", 6, analytic=(1, 0.5)),
    Case("al", "none", 6, analytic=(1, 0.5)),
    Case("ga", "none", 6, analytic=(1, 0.5)),
    Case("in", "none", 6, analytic=(1, 0.5)),
    Case("tl", "none", 6, analytic=(1, 0.5)),
    Case("ce3p", "none", 14, analytic=(3, 0.5)),
    Case("yb3p", "none", 14, analytic=(3, 0.5)),
    # The control: the same ion with the two-electron picture change on. If the shift moves, the
    # correction is entangled with the screening and every other row needs re-reading.
    Case("ce3p", "x2camf", 14, analytic=(3, 0.5), label_suffix="+x2camf"),
    Case("ticl3", "x2camf", 10),
    Case("dy3p", "none", 134, analytic=(5, 2.5)),
]


def _occupied_active_block(spaces, n_elec_inactive: int) -> np.ndarray:
    """Indices of the occupied+active spinors — the block a valence property actually samples."""
    return np.concatenate([np.asarray(spaces.inactive, dtype=int),
                           np.asarray(spaces.active, dtype=int)])


def operator_norms(reference, outcome) -> Dict[str, float]:
    """Measurement 1: how much the picture change moves the operator, on the block that matters.

    ⚠ Projected onto occupied+active. A global norm here would be core-dominated and would report
    tens of percent for an effect that is four orders smaller where any electron actually is.
    """
    from kuiva.props.dump import spinor_operator, spinor_operators
    from kuiva.spinor.expand import spin_operator

    props = reference.data.properties
    moment = props.moment_operator()
    if moment is None:
        return {}
    l_mo, s_mo = spinor_operators(outcome.coeff, props.two_component(),
                                  spin_operator(reference.data.s_ao))
    bare = l_mo + 2.0 * s_mo
    got = spinor_operator(outcome.coeff, moment)
    idx = _occupied_active_block(outcome.active.spaces, 0)
    sub = np.ix_(idx, idx)
    rel_block = max(float(np.max(np.abs(got[k][sub] - bare[k][sub])))
                    / (float(np.max(np.abs(bare[k][sub]))) or 1.0) for k in range(3))
    rel_global = max(float(np.max(np.abs(got[k] - bare[k])))
                     / (float(np.max(np.abs(bare[k]))) or 1.0) for k in range(3))
    return {"occupied_active": rel_block, "global_do_not_quote": rel_global,
            "n_occupied_active": int(idx.size)}


def multiplet_rows(matrices, tol_cm: float = 1.0) -> List[Dict[str, object]]:
    """The phase-invariant reduction, as plain records."""
    rows = []
    for block in matrices.analyse(tol_cm=tol_cm):
        rows.append({
            "size": int(block.size), "j": float(block.j),
            "energy_cm": float(block.energy_cm), "spread_cm": float(block.spread_cm),
            "g_iso": float(block.g_iso),
            "g_values": [float(g) for g in block.g_values],
        })
    return rows


def run_case(case: Case, beat, memory_gb: float = 4.0) -> Dict[str, object]:
    """One system: one CASSCF, then both property operators on the same states.

    ⚠ ``memory_gb`` is deliberately well under the machine's free memory rather than close to
    it. The accounted budget is a *lower* bound on RSS — the front-end SCF's allocation is
    dynamic and explicitly outside it — so a limit set near the free memory is a limit that
    gets the process killed by the kernel instead of refused by the budget, and a kernel kill
    leaves no message and no partial result.
    """
    system = systems.get(case.key)
    t0 = time.time()
    cpu0 = time.process_time()

    mol = api.Molecule(atoms=system.atoms, basis=system.basis,
                       charge=system.charge, spin=system.spin)
    reference = api.spinor_reference(mol, screening=case.screening, memory_gb=memory_gb,
                                     property_picture_change=True)
    beat("{}: reference done".format(case.name))
    # ⚠ Through the shared helper, never rebuilt here: a plain (atom, l) selection takes the
    # LOWEST pairs of that character, which above boron is the 2p core. That is how this
    # series was measured before 2026-08-29 — converged, unremarkable, and undetectable from
    # the g values, since a p^1 shell is Lande 2/3 whichever shell it sits in.
    outcome = api.casscf(reference, n_states=case.n_states, mode="second-order",
                         conv_grad=1e-6, report=False,
                         **systems.character_selection(system))
    beat("{}: casscf converged={}".format(case.name, outcome.converged))

    props = reference.data.properties
    tdm = outcome.ci.transition_densities()
    args = (outcome.coeff, outcome.active.spaces, tdm, outcome.ci.total_energies)

    bare = property_matrices(*args, dataclasses.replace(props, picture_change=None),
                             reference.data.s_ao)
    corrected = property_matrices(*args, props, reference.data.s_ao)

    row: Dict[str, object] = {
        "key": case.key, "label": system.label, "screening": case.screening,
        "basis": system.basis, "n_states": case.n_states,
        "construction": "SA-CASSCF over {} roots, CAS({}, {} spinors), {}".format(
            case.n_states, system.nelecas, 2 * system.ncas, system.basis),
        "converged": bool(outcome.converged),
        "identity_residual": float(props.picture_change.identity_residual),
        "operator_norms": operator_norms(reference, outcome),
        "bare": multiplet_rows(bare),
        "corrected": multiplet_rows(corrected),
    }

    # The quantity the thresholds are applied to: the relative shift of each block's g.
    shifts, deg_shifts = [], []
    for b, c in zip(row["bare"], row["corrected"]):
        shifts.append((c["g_iso"] / b["g_iso"] - 1.0) if b["g_iso"] else 0.0)
        deg_shifts.append(c["spread_cm"] - b["spread_cm"])
    row["g_shift_relative"] = shifts
    row["degeneracy_shift_cm"] = deg_shifts
    row["worst_g_shift"] = float(max(abs(s) for s in shifts)) if shifts else 0.0
    row["worst_degeneracy_shift_cm"] = float(max(abs(d) for d in deg_shifts)) if deg_shifts else 0.0

    if case.analytic is not None:
        l, s = case.analytic
        dev = []
        for b, c in zip(row["bare"], row["corrected"]):
            target = lande_g(l, s, b["j"])
            dev.append({"j": b["j"], "analytic_ge2": float(target),
                        "bare_rel": b["g_iso"] / target - 1.0,
                        "corrected_rel": c["g_iso"] / target - 1.0})
        row["vs_analytic"] = dev

    row["wall_s"] = time.time() - t0
    row["cpu_s"] = time.process_time() - cpu0
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="temp/property_picture_change.json")
    ap.add_argument("--only", default="", help="comma-separated case names")
    ap.add_argument("--skip", default="", help="comma-separated case names")
    ap.add_argument("--memory-gb", type=float, default=4.0,
                    help="working-memory limit per calculation; keep it well under the free "
                         "memory, since the accounted budget is a lower bound on RSS")
    args = ap.parse_args()

    only = {s for s in args.only.split(",") if s}
    skip = {s for s in args.skip.split(",") if s}
    cases = [c for c in CASES if (not only or c.name in only) and c.name not in skip]

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # ⚠ The budget is enforced *inside* the run, so a run that hits it exits with everything it
    # has already computed on disk. An external timeout leaves nothing behind.
    hb = Heartbeat("property_picture_change", budget_seconds=WALL_BUDGET_S,
                   meta={"cases": [c.name for c in cases]})

    def beat(message: str) -> None:
        hb.tick(len(results["systems"]), stage=message)

    started = time.time()
    results: Dict[str, object] = {
        "what": "the X2C picture change on the magnetic moment operator",
        "thresholds": THRESHOLDS,
        "g_electron": G_ELECTRON,
        "systems": {},
    }

    for case in cases:
        if hb.expired or time.time() - started > WALL_BUDGET_S:
            results.setdefault("incomplete", []).append(case.name)
            beat("wall budget exhausted before {}".format(case.name))
            continue
        beat("starting {}".format(case.name))
        try:
            results["systems"][case.name] = run_case(case, beat, args.memory_gb)
        except Exception as exc:                                     # noqa: BLE001
            # A failure on one system must not cost the ones already computed.
            results["systems"][case.name] = {"error": "{}: {}".format(type(exc).__name__, exc)}
            beat("{} FAILED: {}".format(case.name, type(exc).__name__))
        with open(out_path, "w") as fh:                              # rewritten after each system
            json.dump(results, fh, indent=2, sort_keys=True)

    print("\n{:<14} {:>4} {:>12} {:>12} {:>10} {:>9}".format(
        "case", "Z", "op(val)", "worst dg/g", "deg [cm-1]", "wall s"))
    for name, row in results["systems"].items():
        if "error" in row:
            print("{:<14} {}".format(name, row["error"]))
            continue
        norms = row.get("operator_norms") or {}
        print("{:<14} {:>4} {:>12.3e} {:>12.3e} {:>10.2e} {:>9.1f}".format(
            name, "", norms.get("occupied_active", float("nan")),
            row["worst_g_shift"], row["worst_degeneracy_shift_cm"], row["wall_s"]))
    hb.finish(n_systems=len(results["systems"]))
    print("\nwritten to", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
