"""Kuiva's X2CAMF correction against the authors' own implementation.

What this adds that the four-component reference could not
-------------------------------------
``tests/reference/x2camf_dirac.json`` compares Kuiva's *result* against DIRAC's
four-component atoms and finds 0.003-0.005% on a j-splitting. That is the strongest statement
available about whether the answer is right, and it is silent about **where** a disagreement
would be, because a four-component calculation performs no picture change and therefore has
no term corresponding to any of Kuiva's.

The ``x2camf`` plugin (https://github.com/Warlocat/x2camf, built by
an optional external plugin) is the reference implementation of the same method by the
group that published it, so its correction **is** term-by-term comparable — which is exactly
what proper attribution asks of a method taken from someone else's paper: check
it against what they wrote, not only against a different theory that happens to agree.

⚠ The plugin is optional and this generator is not run at test time. ``pytest`` asserts against
the committed JSON; ``tests/test_x2camf_plugin.py`` re-runs the live comparison only when the
plugin is importable, and skips otherwise.

What is recorded per atom, basis and interaction
-------------------------------------------------
1. **The term-by-term difference** of ``dh_sf`` and ``dw``, in the **primitive** basis (where
   the decoupling happens) and in the **molecular** basis (where the correction is used).
   Recording both is not redundancy — the difference between the two conventions lives almost
   entirely in the high-exponent primitive space the contraction projects out, so the two
   numbers differ by two orders of magnitude and only one of them describes what a calculation
   would see.
2. **Both plugin variants.** ``"pcc"`` is the full two-electron picture change and the
   counterpart of Kuiva's ``dG``; ``"soc"`` is the plugin's default entry point and carries
   **only the spin-dependent part**. Recording ``soc`` is what makes the size of the
   two-electron *scalar* picture change visible as a measurement rather than a claim.
3. **The energy functional against four-component Dirac-Coulomb in the same basis** — the
   check that discriminates (the energy-functional check). This is what says which of the two
   conventions is sharper, and by how much.
4. **The valence j-splitting** from a self-consistent two-component SCF, which is what says
   whether any of it matters for the observable X2CAMF exists to get right.

⚠ Three things that had to be pinned, and would each have presented as method error
-------------------------------------------------------------------------------------
* **The basis.** The plugin decontracts with ``mole.uncontracted_basis``, Kuiva with
  ``Mole.decontract_basis(aggregate=True)``. They agree — checked on the *overlap matrices*
  inside :func:`kuiva.amf.x2camf_plugin._decontracted`, because an AO **ordering** difference
  is invisible to a shape check and fatal to a term-by-term comparison.
* **The reference configuration.** The plugin's interface takes an atomic number and nothing
  else, so it always uses the **neutral** atom. Kuiva's f-block default is M(3+)
, so every comparison here is run against Kuiva's neutral reference and
  the atom set is closed-shell besides.
* **The interaction.** The plugin defaults to ``with_gaunt=True, with_gauge=True``, i.e. full
  **Breit** — not Coulomb. Every call passes both flags explicitly.

Cost (CPU seconds, never wall time)
-----------------------------------------------------
The plugin's own atomic solver is fast (its spherical-symmetry four-component SCF is seconds
even for Xe). **Kuiva is the slow half**, as in the DIRAC comparison: each record needs an
uncontracted four-component atomic solve, which grows as roughly ``n4c^4``. Ne and Ar together
are under a minute; Kr and Xe are minutes each and are therefore **not in the default atom
set** — pass them by name and expect to spend the ten-minute budget on them.

Run:  python tests/generate/x2camf_plugin.py [--only Ne,Ar] [--basis x2c-SVPall-2c,dyallv2z]
      (with ``external/env.sh`` sourced). Writes ``tests/reference/x2camf_plugin.json``.

References
----------
* X2CAMF: J. Liu, L. Cheng, J. Chem. Phys. 148, 144108 (2018), doi:10.1063/1.5023750.
* The plugin: C. Zhang, L. Cheng, J. Phys. Chem. A 126, 4537 (2022),
  doi:10.1021/acs.jpca.2c02181; https://github.com/Warlocat/x2camf.
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
from pyscf.gto import mole                                                  # noqa: E402

import thermal                                                              # noqa: E402
from amf_sensitivity import spectrum_of                                     # noqa: E402
from test_amf_decouple import x2camf_energy                                 # noqa: E402
from kuiva.amf import amf_correction                                        # noqa: E402
from kuiva.amf.atomic import atomic_solution, clear_cache                   # noqa: E402
from kuiva.amf.configuration import AtomicConfiguration                     # noqa: E402
from kuiva.amf.correction import AMFCorrection                              # noqa: E402
from kuiva.amf import x2camf_plugin as plugin                               # noqa: E402

OUT = REPO / "tests/reference/x2camf_plugin.json"
HARTREE_CM = 219474.6313632

#: Both project-relevant bases. ``x2c-SVPall-2c`` is the default a molecular calculation
#: would actually use — and the basis in which the comparison is decided, since that is where
#: the correction is applied. ``dyallv2z`` is natively primitive and is the basis of the
#: DIRAC four-component anchor, so a plugin-vs-Kuiva number in it can be bisected against a third
#: program rather than argued about between two.
BASES = ("x2c-SVPall-2c", "dyallv2z")

#: Closed-shell only, and deliberately: the plugin has no configuration input, so an ion
#: cannot be a controlled comparison, and an open-shell neutral would compare Kuiva's
#: average-of-configuration against the plugin's without a way to check they averaged the same
#: state. Ne and Ar are the default set on cost; Kr and Xe are affordable only on purpose.
DEFAULT_ATOMS = ("Ne", "Ar")


@dataclass(frozen=True)
class Atom:
    symbol: str
    valence: str
    valence_spinors: int = 6


ATOMS: Tuple[Atom, ...] = (
    Atom("Ne", "2p"), Atom("Ar", "3p"), Atom("Kr", "4p"), Atom("Xe", "5p"),
)
BY_SYMBOL = {a.symbol: a for a in ATOMS}


def _splitting_cm(energies: np.ndarray, width: int) -> float:
    """Spread of the top ``width`` occupied spinor energies, in cm^-1.

    The same quantity ``AtomicDiracSolution.shell_splitting`` returns and the same one
    ``tests/generate/x2camf_dirac.py`` parses out of DIRAC, so the three are comparable
    without any convention being restated.
    """
    valence = np.sort(np.asarray(energies))[-width:]
    return float((valence[-1] - valence[0]) * HARTREE_CM)


def _scales(h_sf: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    return {"max_dh_sf": float(np.max(np.abs(h_sf))) if h_sf.size else 0.0,
            "max_dw": float(np.max(np.abs(w))) if w.size else 0.0}


def _difference(a: Tuple[np.ndarray, np.ndarray],
                b: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Term-by-term, absolute and relative. Relative to the **larger** of the two scales, so
    the number does not depend on which side is called the reference."""
    out: Dict[str, float] = {}
    for name, x, y in (("dh_sf", a[0], b[0]), ("dw", a[1], b[1])):
        scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1e-300)
        out["abs_" + name] = float(np.max(np.abs(x - y)))
        out["rel_" + name] = out["abs_" + name] / scale
    return out


def measure(atom: Atom, basis_name: str, interaction: str) -> Dict:
    """One (atom, basis, interaction) record. Everything in it comes from one pair of solves."""
    clear_cache()
    symbol = atom.symbol
    configuration = AtomicConfiguration.ground(symbol)
    primitive = mole.uncontracted_basis(gto.basis.load(basis_name, symbol))
    molecular = gto.M(atom=[(symbol, (0.0, 0.0, 0.0))], basis=basis_name, verbose=0)
    prim_mol = gto.M(atom=[(symbol, (0.0, 0.0, 0.0))], basis={symbol: primitive}, verbose=0)

    record: Dict = {
        "element": symbol, "basis": basis_name, "interaction": interaction,
        "configuration": configuration.canonical,
        "valence_shell": atom.valence, "valence_spinors": atom.valence_spinors,
        "nao_molecular": int(molecular.nao), "nao_primitive": int(prim_mol.nao),
        "plugin_version": plugin.version(),
    }

    # -- the two corrections, in the basis where the decoupling happens ---------------------
    # ⚠ `uncontract=False` on an already-primitive molecule, not `uncontract=True` on the
    # contracted one: the point is to compare the matrices the decoupling produced, before a
    # contraction that projects part of the difference away.
    kuiva_prim = amf_correction(prim_mol, method="x2camf", interaction=interaction,
                                configuration=configuration, uncontract=False)
    plug_prim = {v: plugin.plugin_correction(prim_mol, interaction=interaction, variant=v,
                                             configuration=configuration)
                 for v in plugin.VARIANTS}
    record["primitive_basis"] = {
        "kuiva": _scales(kuiva_prim.h_sf, kuiva_prim.w),
        "plugin_pcc": _scales(*plug_prim["pcc"]),
        "plugin_soc": _scales(*plug_prim["soc"]),
        "kuiva_vs_plugin_pcc": _difference((kuiva_prim.h_sf, kuiva_prim.w),
                                           plug_prim["pcc"]),
        "kuiva_vs_plugin_soc": _difference((kuiva_prim.h_sf, kuiva_prim.w),
                                           plug_prim["soc"]),
    }

    # -- the same in the molecular basis: where the correction is actually used -------------
    kuiva_mol = amf_correction(molecular, method="x2camf", interaction=interaction,
                               configuration=configuration, uncontract=True)
    plug_mol = plugin.plugin_correction(molecular, interaction=interaction, variant="pcc",
                                        configuration=configuration)
    record["molecular_basis"] = {
        "kuiva": _scales(kuiva_mol.h_sf, kuiva_mol.w),
        "plugin_pcc": _scales(*plug_mol),
        "kuiva_vs_plugin_pcc": _difference((kuiva_mol.h_sf, kuiva_mol.w), plug_mol),
    }

    # -- the energy functional, which is what says whose convention is sharper --------------
    # In the primitive basis, against the four-component solve in the same basis.
    reference = atomic_solution(symbol, primitive, configuration=configuration,
                                interaction=interaction, uncontract=False)
    zero = np.zeros((2 * prim_mol.nao, 2 * prim_mol.nao), dtype=complex)
    energies = {
        "four_component": float(reference.e_tot),
        "none": x2camf_energy(prim_mol, zero),
        "kuiva": x2camf_energy(prim_mol, kuiva_prim.hamiltonian()),
    }
    for v, (h_sf, w) in plug_prim.items():
        energies["plugin_" + v] = x2camf_energy(
            prim_mol, AMFCorrection(h_sf=h_sf, w=w).hamiltonian())
    record["energy_functional"] = {
        "total_eh": energies,
        "error_eh": {k: v - energies["four_component"]
                     for k, v in energies.items() if k != "four_component"},
    }

    # -- the observable, from a self-consistent two-component SCF ---------------------------
    # ⚠ Which construction produced a splitting must always be stated: the
    # frozen-scalar-orbital one disagrees with this by 30% on the *same* operator.
    width = atom.valence_spinors
    plug_correction = AMFCorrection(h_sf=plug_mol[0], w=plug_mol[1], method="x2camf-external",
                                    interaction=interaction)
    record["splitting_cm"] = {
        "four_component": float(reference.shell_splitting(width) * HARTREE_CM),
        "one_electron_x2c": _splitting_cm(spectrum_of(molecular, configuration, None), width),
        "kuiva": _splitting_cm(spectrum_of(molecular, configuration, kuiva_mol), width),
        "plugin_pcc": _splitting_cm(spectrum_of(molecular, configuration, plug_correction),
                                    width),
    }
    record["splitting_construction"] = ("self-consistent two-component SCF in the molecular "
                                        "(contracted) basis; the four-component column is in "
                                        "the primitive basis the decoupling used")
    return record


def _new_document() -> Dict:
    return {
        "schema": 1,
        "generator": "tests/generate/x2camf_plugin.py",
        "code": "x2camf plugin (github.com/Warlocat/x2camf), pinned in "
                "x2camf plugin commit",
        "purpose": ("term-by-term comparison of Kuiva's X2CAMF correction against the "
                    "authors' own implementation"),
        "controlled": {
            "reference_configuration": ("neutral atom on both sides — the plugin takes no "
                                        "configuration input"),
            "contraction": ("both decontract; the primitive bases are checked to agree on "
                            "their overlap matrices before anything is compared"),
            "interaction": ("passed explicitly to the plugin, which defaults to full Breit "
                            "(with_gaunt and with_gauge both True)"),
            "speed_of_light": "PySCF's lib.param.LIGHT_SPEED, which both codes read",
        },
        "variants": {
            "pcc": "the full two-electron picture change — the counterpart of Kuiva's dG",
            "soc": ("the plugin's default entry point: the spin-DEPENDENT part only, with no "
                    "two-electron scalar picture change at all"),
        },
        "environment": thermal.describe_environment(),
        "records": {},
    }


def _write(document: Dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, indent=2, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=",".join(DEFAULT_ATOMS),
                    help="comma-separated element symbols (Kr and Xe cost minutes each)")
    ap.add_argument("--basis", default=",".join(BASES))
    ap.add_argument("--interaction", default="coulomb,gaunt")
    ap.add_argument("--max-wall", type=float, default=540.0,
                    help="total wall budget [s]; no new record is started past it ")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args(argv)

    if not plugin.available():
        print("the x2camf plugin is not installed; build it from its own repository",
              file=sys.stderr)
        return 2

    document = _new_document()
    if not args.fresh and OUT.is_file():
        try:
            document["records"] = json.loads(OUT.read_text()).get("records", {})
        except ValueError:
            pass

    symbols = [s.strip().capitalize() for s in args.only.split(",") if s.strip()]
    bases = [b.strip() for b in args.basis.split(",") if b.strip()]
    interactions = [i.strip() for i in args.interaction.split(",") if i.strip()]
    started = time.time()
    rc = 0
    for symbol in symbols:
        if symbol not in BY_SYMBOL:
            print("unknown atom {!r}".format(symbol), file=sys.stderr)
            rc = 1
            continue
        for basis_name in bases:
            for interaction in interactions:
                elapsed = time.time() - started
                if elapsed > args.max_wall:
                    print("[budget] {:.0f} s of {:.0f} s used; stopping before {} {} "
                          "{}".format(elapsed, args.max_wall, symbol, basis_name,
                                      interaction), flush=True)
                    _write(document)
                    return rc
                key = "{}/{}/{}".format(symbol, basis_name, interaction)
                with thermal.track_resources() as res:
                    try:
                        record = measure(BY_SYMBOL[symbol], basis_name, interaction)
                        record["status"] = "ok"
                    except Exception as exc:                                # noqa: BLE001
                        record = {"element": symbol, "basis": basis_name,
                                  "interaction": interaction,
                                  "status": "{}: {}".format(type(exc).__name__, exc)}
                        rc = 1
                record["resources"] = res.as_dict()
                document["records"][key] = record
                _write(document)                    # incremental, within the ten-minute ad-hoc budget
                if record["status"] == "ok":
                    m = record["molecular_basis"]["kuiva_vs_plugin_pcc"]
                    s = record["splitting_cm"]
                    print("[plugin] {:28s} molecular rel diff: dh_sf {:.2e} dw {:.2e} | "
                          "splitting 4c {:.2f} kuiva {:.2f} plugin {:.2f} cm^-1  ({})".format(
                              key, m["rel_dh_sf"], m["rel_dw"], s["four_component"],
                              s["kuiva"], s["plugin_pcc"], res.summary()), flush=True)
                else:
                    print("[plugin] {:28s} {}".format(key, record["status"]), flush=True)

    _write(document)
    print("\nwrote {}  ({} records)".format(OUT.relative_to(REPO), len(document["records"])))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
