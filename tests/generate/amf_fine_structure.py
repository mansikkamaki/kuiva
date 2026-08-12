"""Atomic fine structure with and without the AMF correction.

The stated exit criterion is *"NIST fine structure for Bi 6p: the +16% error is
substantially reduced"*. This generator measures it, and reports two comparisons per atom
because they answer different questions and only one of them is sharp:

* **against the four-component reference for the same atom in the same basis** — free,
  in-process, and the statement that the *picture change* is right.
  Expected at the fraction-of-a-percent level.
* **against experiment** — an anchor at the 30% level , never an accuracy
  claim, and ⚠ **only where the measured splitting is genuinely a one-particle quantity**: a
  closed shell, or one electron or hole outside one. For a multi-electron open shell the
  measured levels are *term* energies set by electron repulsion as well as spin-orbit coupling,
  and comparing them to a mean-field spinor splitting measures the missing correlation rather
  than the picture change. Bi is exactly that case and its anchor is deliberately absent; see
  :class:`Atom`. Atomic multiplet energies become the right comparison once there is a CI to
  compute them with, and they will then verify far more than this does.

⚠ Three things must be held fixed or this measures the wrong thing
-------------------------------------------------------------------
1. **Which SCF construction produced the splitting** . Everything here is a
   **self-consistent two-component AOC SCF**; the frozen-scalar-orbital construction of
   ``tests/test_soc_ingestion.py`` disagrees by 30% on the *same* operator and happens to land
   near the four-component answer for reasons unrelated to screening.
2. **The interaction.** The residual against experiment contains at least
   two separable effects: Gaunt (roughly half of it for Ne) and the reference
   configuration. Both interactions are recorded per atom so the Bi criterion can be read with
   the interaction held fixed instead of attributing Gaunt's effect to the configuration.
3. **The basis, on both sides.** The four-component reference is taken in the same
   basis the correction was decoupled in.

Atom set: Ne as the light control (its residual is the +16% this work started from) and
**Bi**, the criterion. This is a budgeted run, not a test.

⚠ **Gaunt scales far worse than Dirac-Coulomb at high ``Z``, and the light-atom ratio does not
extrapolate.** Neon costs 1 s for Coulomb and 2 s for Gaunt, a factor of two. Bismuth costs
**3412 s wall / 7.5 CPU-hours** for Coulomb and was **abandoned after 3.4 h wall / 27 CPU-hours
without converging** for Gaunt, i.e. at least 5x and probably much more — the Gaunt operator
works over the small-component basis, which is where a decontracted heavy-element set is
largest. Budget a heavy-element Gaunt run as its own multi-hour job, never as "Coulomb plus a
bit", and do not size it by scaling a light atom.

Run:  python tests/generate/amf_fine_structure.py [--only Bi] [--interaction coulomb,gaunt]
      (with ``external/env.sh`` sourced). Writes ``tests/reference/amf_fine_structure.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pyscf import gto                                                       # noqa: E402

import thermal                                                              # noqa: E402
from amf_sensitivity import spectrum_of                                     # noqa: E402
from kuiva.amf import amf_correction                                        # noqa: E402
from kuiva.amf.atomic import atomic_solution, clear_cache                   # noqa: E402
from kuiva.amf.configuration import AtomicConfiguration                     # noqa: E402

OUT = REPO / "tests/reference/amf_fine_structure.json"
BASIS = "x2c-SVPall-2c"
HARTREE_CM = 219474.6313632


@dataclass(frozen=True)
class Atom:
    """One atom, its reference configuration, and — where one exists — an experimental anchor.

    ⚠ **`experiment_cm` is ``None`` unless the experimental splitting is genuinely a
    one-particle quantity**, and that restriction is the point (user decision, recorded per
    a recorded decision). What Kuiva computes here is the spread of the valence spinors in a
    **mean field**. For a closed shell, or a single electron or hole outside one, the measured
    fine structure is the same observable to within the 30% experimental-anchor band.
    For a **multi-electron open shell it is not**: Bi's 6p^3 levels are *term* energies set by
    electron repulsion and spin-orbit coupling together, so comparing them to a one-particle
    splitting measures the missing correlation, not the picture change.

    Reporting one anyway is not a harmless conservatism — it was measured, and it reads as a
    +38% method error where the four-component agreement is **+0.018%**. Atomic multiplet
    energies become comparable once there is a CI to compute them with; until then the
    four-component reference carries the criterion, and it is the sharper of the two anyway.
    """

    symbol: str
    configuration: str
    valence: str
    experiment_cm: Optional[float]
    experiment_note: str


ATOMS = (
    Atom("Ne", "1s2 2s2 2p6", "2p", 780.4,
         "Ne(+) 2p^5 fine structure, NIST ASD — a single hole outside a closed shell, so the "
         "measured splitting is a one-particle quantity. ⚠ The mean field is over *neutral* "
         "Ne, which has one more electron screening, and that is part of the residual"),
    Atom("Bi", "[Xe]4f14 5d10 6s2 6p3", "6p", None,
         "⚠ deliberately absent. Bi's 6p^3 fine structure (4S(3/2)-2D(3/2) = 11419 cm^-1, "
         "NIST ASD) is a *term* splitting of a three-electron open shell, not a one-particle "
         "one — the four-component reference is itself +38% above it. Deferred to a "
         "comparison of atomic multiplet energies once a CI exists to compute them"),
)


def valence_width(atom: Atom) -> int:
    """Spinors the valence shell spans — ``4l+2`` for an open shell, and the whole of it,
    because average of configuration populates the entire manifold."""
    config = AtomicConfiguration.parse(atom.configuration)
    open_shells = config.open_shells()
    if open_shells:
        return 4 * open_shells[0][0] + 2
    # A closed valence shell: take the l of the outermost occupied channel.
    l = max(l for l, n in enumerate(config.occupations) if n)
    return 4 * l + 2


def measure(atom: Atom, interaction: str) -> Dict:
    # ⚠ The two-component calculation is run in the **primitive** basis, not the contracted
    # molecular one, so that it and the four-component reference live in the same space.
    # A measurement recorded what the alternative costs: comparing a contracted two-component
    # result against a primitive four-component reference gives +0.272% for Ne where the
    # like-for-like residual on the same code path is -0.003%, and -16% for Ca in
    # cc-pVDZ-X2C. That difference is a basis change, not the correction, and it would be
    # read here as method error.
    contracted = gto.M(atom=[(atom.symbol, (0.0, 0.0, 0.0))], basis=BASIS,
                       spin=int(gto.charge(atom.symbol)) % 2, verbose=0)
    parsed = contracted._basis[atom.symbol]
    mol = gto.M(atom=[(atom.symbol, (0.0, 0.0, 0.0))],
                basis={atom.symbol: gto.uncontract(parsed)},
                spin=int(gto.charge(atom.symbol)) % 2, verbose=0)
    config = AtomicConfiguration.parse(atom.configuration)
    width = valence_width(atom)

    four_component = atomic_solution(atom.symbol, parsed, configuration=config,
                                     interaction=interaction, uncontract=True)
    e4c = four_component.occupied_energies()[-width:]
    reference_cm = float((e4c[-1] - e4c[0]) * HARTREE_CM)

    def splitting(correction=None):
        e = spectrum_of(mol, config, correction)[-width:]
        return float((e[-1] - e[0]) * HARTREE_CM)

    uncorrected = splitting()
    correction = amf_correction(mol, method="x2camf", configuration=config,
                                interaction=interaction, uncontract=True)
    corrected = splitting(correction)
    # ⚠ Assembled into a name and returned at the end, not returned inline: an earlier version
    # returned this literal directly and left the ``experiment_cm`` block below unreachable, so
    # every regenerated record silently lost its experimental anchor while the committed one
    # (written before the refactor) still had it. The generator crashed on printing rather than
    # on writing, so the damaged record reached the file first.
    record = {
        "element": atom.symbol, "basis": BASIS, "interaction": interaction,
        "nao_contracted": int(contracted.nao), "nao_primitive": int(mol.nao),
        "basis_note": ("both the two-component calculation and the four-component reference "
                       "are in the primitive basis; comparing across a contraction is a "
                       "basis change, not method error"),
        "configuration": config.canonical, "valence_shell": atom.valence,
        "valence_spinors": width,
        "construction": "self-consistent two-component average-of-configuration SCF",
        "four_component_cm": reference_cm,
        "four_component_converged": bool(four_component.converged),
        "uncorrected_cm": uncorrected,
        "corrected_cm": corrected,
        "uncorrected_vs_4c": (uncorrected - reference_cm) / reference_cm,
        "corrected_vs_4c": (corrected - reference_cm) / reference_cm,
        "experiment_cm": atom.experiment_cm,
        "experiment_note": atom.experiment_note,
    }
    if atom.experiment_cm:
        record["uncorrected_vs_experiment"] = (
            (uncorrected - atom.experiment_cm) / atom.experiment_cm)
        record["corrected_vs_experiment"] = (
            (corrected - atom.experiment_cm) / atom.experiment_cm)
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated element symbols")
    ap.add_argument("--interaction", default="coulomb",
                    help="comma-separated subset of coulomb,gaunt,breit. ⚠ Record both "
                         "coulomb and gaunt before reading a residual against experiment: "
                         "Gaunt accounts for roughly half of it at the light end")
    ap.add_argument("--max-wall", type=float, default=540.0,
                    help="total wall budget [s]; no new atom is started past it")
    args = ap.parse_args(argv)

    wanted = {s.strip().capitalize() for s in args.only.split(",") if s.strip()}
    interactions = [s.strip() for s in args.interaction.split(",") if s.strip()]
    document: Dict = {
        "schema": 1,
        "generator": "tests/generate/amf_fine_structure.py",
        "purpose": ("atomic fine structure with and without the X2CAMF two-electron "
                    "picture-change correction"),
        "basis": BASIS,
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

    for atom in ATOMS:
        if wanted and atom.symbol not in wanted:
            continue
        for interaction in interactions:
            elapsed = time.time() - started
            if elapsed > args.max_wall:
                print("[budget] {:.0f} s of {:.0f} s used; stopping before {} {}".format(
                    elapsed, args.max_wall, atom.symbol, interaction), flush=True)
                return 0
            key = "{}/{}".format(atom.symbol, interaction)
            clear_cache()
            try:
                with thermal.track_resources() as res:
                    record = measure(atom, interaction)
                record["resources"] = res.as_dict()
                record["status"] = "ok"
            except Exception as exc:                                        # noqa: BLE001
                record = {"element": atom.symbol, "interaction": interaction,
                          "status": "{}: {}".format(type(exc).__name__, str(exc)[:200])}
            document["records"][key] = record
            OUT.write_text(json.dumps(document, indent=2, sort_keys=True))
            if record["status"] == "ok":
                anchor = ("experiment {:9.1f} ({:+.1f}% -> {:+.1f}%)".format(
                    record["experiment_cm"], 100 * record["uncorrected_vs_experiment"],
                    100 * record["corrected_vs_experiment"])
                    if record.get("experiment_cm") else
                    "experiment: n/a (term splitting, needs CI)")
                print("{:12s} {:3s}  4c {:10.2f}   1e-X2C {:10.2f} ({:+.1f}%)   "
                      "X2C+AMF {:10.2f} ({:+.3f}%)   {}  {}".format(
                          key, record["valence_shell"], record["four_component_cm"],
                          record["uncorrected_cm"], 100 * record["uncorrected_vs_4c"],
                          record["corrected_cm"], 100 * record["corrected_vs_4c"],
                          anchor, record["resources"].get("summary", "")), flush=True)
            else:
                print("{:12s} {}".format(key, record["status"]), flush=True)

    print("\nwrote {}".format(OUT.relative_to(REPO)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
