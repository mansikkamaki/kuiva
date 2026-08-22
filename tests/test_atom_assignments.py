"""Per-atom assignments: the addressing scheme, the curated oxidation-state table, and the
end-to-end per-atom bases and reference configurations.

Tier 0 for the pure resolution machinery (`kuiva.basis.atommap`, `kuiva.amf.oxidation`) and
cheap end-to-end checks through the front end. The rule under test throughout: **1-based
numbering in all input and output** (user decision), most-specific key wins, and anything
naming an atom or element the molecule does not have is refused rather than ignored.
"""
import numpy as np
import pytest

from kuiva.amf.configuration import AtomicConfiguration
from kuiva.amf.oxidation import (accepted_configurations, canonical_configuration,
                                 common_states, resolve_reference_configuration)
from kuiva.basis.atommap import parse_atom_key, resolve_atom_assignments
from kuiva.interface import Molecule, api
from kuiva.interface.pyscf_bridge import build_mole

TIF3 = [("Ti", (0.0, 0.0, 0.0)), ("F", (1.7796, 0.0, 0.0)),
        ("F", (-0.8898, 1.5412, 0.0)), ("F", (-0.8898, -1.5412, 0.0))]
SYMS = [a[0] for a in TIF3]


# --- The addressing scheme ------------------------------------------------------------------

def test_key_kinds_and_one_based_numbering():
    assert parse_atom_key("Ti", SYMS) == ("element", "Ti")
    assert parse_atom_key("F2", SYMS) == ("atom", 1)       # 1-based in, 0-based internally
    assert parse_atom_key(3, SYMS) == ("atom", 2)
    assert parse_atom_key("3", SYMS) == ("atom", 2)


def test_wrong_element_label_is_refused_not_reinterpreted():
    with pytest.raises(ValueError, match="refused rather than reinterpreted"):
        parse_atom_key("F1", SYMS)                          # atom 1 is Ti
    with pytest.raises(ValueError, match="1-based"):
        parse_atom_key(5, SYMS)
    with pytest.raises(ValueError, match="names no atom"):
        parse_atom_key("O", SYMS)


def test_precedence_most_specific_wins():
    values, specific = resolve_atom_assignments(
        {"default": "a", "F": "b", 3: "c"}, SYMS, what="test")
    assert values == ["a", "b", "c", "b"]
    assert specific == [False, False, True, False]


def test_scalar_on_multi_element_molecule_is_refused():
    with pytest.raises(ValueError, match="pass a mapping"):
        resolve_atom_assignments("+3", SYMS, what="oxidation state", allow_scalar=False)


# --- The curated table ----------------------------------------------------------------------

def test_canonical_configurations_from_oxidation_states():
    assert canonical_configuration("Ti", 3).occupations == (6, 12, 1)     # [Ar]3d1
    assert canonical_configuration("Ce", 4).occupations == (10, 24, 20)   # [Xe]4f0
    assert canonical_configuration("Cl", -1).occupations == (6, 12)       # [Ar]
    assert 3 in common_states("Ti") and 4 in common_states("Ce")


def test_ambiguous_states_accept_several_configurations():
    """User decision: where the literature genuinely admits more than one occupation,
    every accepted one resolves silently — but an oxidation-state *input* still produces
    exactly one canonical configuration."""
    ni0 = [a.occupations for a in accepted_configurations("Ni", 0)]
    assert (8, 12, 8) in ni0 and (7, 12, 9) in ni0          # 3d8 4s2 and 3d9 4s1
    ni2 = [a.occupations for a in accepted_configurations("Ni", 2)]
    assert (6, 12, 8) in ni2 and (7, 12, 7) in ni2          # 3d8 and 3d7 4s1


def test_warnings_uncommon_state_excited_config_anion(kuiva_caplog):
    resolve_reference_configuration("Fe", "+5")             # uncommon oxidation state
    resolve_reference_configuration("Fe", "[Ar]4s2 3d4")    # excited Fe(2+) reference
    resolve_reference_configuration("O", AtomicConfiguration((4, 6)))  # anion
    text = " ".join(r.getMessage() for r in kuiva_caplog.records)
    assert "not a common oxidation state" in text
    assert "excited or unusual" in text
    assert "anion" in text


def test_common_inputs_are_silent(kuiva_caplog):
    cfg, is_default = resolve_reference_configuration("Ti", "+3")
    assert cfg.occupations == (6, 12, 1) and not is_default
    resolve_reference_configuration("Ni", "[Ar]3d9 4s1")    # accepted alternate of Ni(0)
    cfg, is_default = resolve_reference_configuration("Ce", None)
    assert is_default                                       # the AMF default: Ce(3+)
    assert not kuiva_caplog.records


def test_impossible_channel_is_refused():
    with pytest.raises(ValueError, match="f channel"):
        resolve_reference_configuration("Ne", "1s2 2s2 2p5 4f1")


# --- End to end through the front end -------------------------------------------------------

def test_per_atom_basis_changes_only_the_named_atom():
    plain = build_mole(Molecule(TIF3, basis="x2c-SVPall-2c", spin=1))
    mixed = build_mole(Molecule(TIF3, basis={"default": "x2c-SVPall-2c",
                                             3: "x2c-TZVPall-2c"}, spin=1))
    assert mixed.__dict__["_kuiva_atom_labels"] == ["Ti", "F2", "F3", "F4"]
    # exactly one F moved from SVP (14 functions) to TZVP (31)
    assert mixed.nao - plain.nao == 17
    assert plain.__dict__["_kuiva_atom_labels"] == ["Ti", "F", "F", "F"]


def test_per_atom_configuration_reaches_charges_and_warns(kuiva_caplog):
    """Two atoms of one element with different reference states: decorated labels, separate
    reference entries, per-label provenance, and the not-comparable warning."""
    mol = Molecule([("O", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.208))],
                   basis="x2c-SVPall-2c", spin=2)
    ref = api.spinor_reference(mol, screening="none", memory_gb=4.0,
                               atomic_reference=True, configuration={2: "-2"})
    q = ref.atomic_reference_charges(report=True)
    assert q.any_non_default
    assert set(q.configurations) == {"O1", "O2"}
    assert q.charge.sum() == pytest.approx(0.0, abs=1e-8)
    # the two identical atoms get different charges ONLY because their references differ
    assert abs(q.charge[0] - q.charge[1]) > 0.05
    assert any("NOT comparable" in r.getMessage() for r in kuiva_caplog.records)


def test_atom_labels_are_one_based_in_output():
    mol = build_mole(Molecule(TIF3, basis="x2c-SVPall-2c", spin=1))
    from kuiva.interface.pyscf_bridge import ao_layout
    layout = ao_layout(mol)
    assert layout.atom_label(0) == "1 Ti"
    assert layout.atom_label(3) == "4 F"
