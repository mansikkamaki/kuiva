"""The vocabulary: what an automatic active space is *asked* for.

A **target** is a physical statement of something the active space has to describe -- "the
valence 4f shells of the two dysprosiums", "the radical HOMO of the bridge", "the double
shell of the cerium". It is not a set of orbitals and it names no index: resolving a target
against a reference is :mod:`kuiva.autocas.candidates`' job, and what comes out carries the
statement with it so that another program can reproduce the same selection.

Five classes, one per row of the table below, and **nothing else is a target**. Each one
answers a different question about a calculation, has its own count rule, and sits at its own
place in a fixed priority order -- which is the order the classes are *added* in and the
reverse of the order they are *dropped* in when the space is too large:

=========== ========== ================================================ ==================
class       priority   what it is                                       count
=========== ========== ================================================ ==================
Shell          1       a centre's whole valence ``l`` shell             fixed: 2l+1 pairs
Frontier       2       a fragment's singly occupied orbitals            stated
Bridge         3       ligand orbitals between two centres             bounded, pruned
Bonding        4       metal-ligand bonding partners of a shell         bounded, pruned
DoubleShell    5       the correlating shell of the same ``l``          fixed, whole
=========== ========== ================================================ ==================

⚠ **Priority is fixed and is not a knob.** It is a statement about physics -- the mechanism
before the correlation refinement -- and it is what makes a dropped class attributable. A
class the user asked for by name is still dropped when the space is over budget; what
"requested" changes is that the class is offered at all.

Spellings
---------
Every class has a plain spelling, so a script states its targets without importing anything
from here and the top level needs no new name:

.. code-block:: python

    "shells"                                  # every detected d/f centre's valence shell
    ("shell", "Dy")                           # that element's centres
    ("shell", ("Ti1", "Ti2"), "d")            # these atoms, this shell
    ("frontier", ("N1", "N2"))                # the fragment's SOMOs
    ("frontier", ("N1", "N2"), 1, 1)          # ... plus one occupied and one empty pair
    ("bridge", ("Dy1", "Dy2"))                # ligand orbitals between the two sites
    ("bridge", ("Dy1", "Dy2"), ("O5",), 2)    # ... on named atoms, at most two pairs
    ("bonding", "Fe")                         # the shell's bonding partners
    ("bonding", "Fe", 3)
    ("double", "Ce")                          # the correlating shell

:func:`parse` reads all of these, and every target writes its own spelling back
(:meth:`spelling`), so a resolved request can be printed as something a user could have
typed. Round-tripping is asserted in the suite, because a description nobody can re-enter is
provenance that cannot be checked.

⚠ Atom addressing is the project's one scheme
---------------------------------------------
Atoms are named as everywhere else (:mod:`kuiva.basis.atommap`): an element symbol, an atom
label ``"Dy2"``, a 1-based atom number, or a sequence of those for a fragment. **Numbering
is 1-based in every statement here**, and :func:`resolve_atoms` is the one place it becomes
the internal 0-based index.

⚠ An element symbol **pools** every atom of that element, and that is a deliberate
difference from :func:`kuiva.mcscf.casci.active_space_by_character`, which refuses an
ambiguous symbol. The reason the refusal exists there is that a ``character=`` statement has
to say which centre it means; here the pooling *is* the statement, it is printed with the
atoms it resolved to, and the equivalent-centre case (two symmetric metals whose canonical d
orbitals are combinations over both) is the normal one rather than the exception. Where one
of several like atoms is meant, its label says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

__all__ = ["Bonding", "Bridge", "DoubleShell", "Frontier", "PRIORITY", "Shell", "Target",
           "angular_momentum_letter", "parse", "parse_targets", "resolve_atoms",
           "atom_labels"]

#: Priority of each class: the order classes are added to the space in, and the reverse of
#: the order they are dropped in over budget. ⚠ Not configurable -- see the module docstring.
PRIORITY = {"shell": 1, "frontier": 2, "bridge": 3, "bonding": 4, "double": 5}

#: Default number of bridge candidate pairs offered to the probe.
DEFAULT_BRIDGE_PAIRS = 4
#: Default number of bonding candidate pairs offered per centre.
DEFAULT_BONDING_PAIRS = 2


def angular_momentum_letter(ell: int) -> str:
    """``2 -> "d"``. The spelling a statement is written in; no principal quantum number."""
    return "spdfghi"[int(ell)]


def _keys(value) -> Tuple[object, ...]:
    """One atom key or a sequence of them -> a tuple of keys, unresolved.

    A string is one key even though it is a sequence of characters, which is the one thing
    a naive ``tuple(value)`` gets wrong -- ``"Dy"`` would become ``("D", "y")`` and name
    nothing.
    """
    if isinstance(value, str):
        return (value,)
    if hasattr(value, "__index__") and not isinstance(value, bool):
        return (int(value),)          # an atom number, including a numpy integer
    if value is None:
        raise ValueError("an atom key is an element symbol, a label like 'Dy2' or a 1-based "
                         "atom number, not None")
    keys = tuple(value)
    if not keys:
        raise ValueError("a fragment names at least one atom")
    return keys


def _one_key(value) -> object:
    """The spelling a single-key tuple is written back as: the key itself."""
    keys = _keys(value)
    return keys[0] if len(keys) == 1 else keys


def _ell(value) -> Optional[int]:
    if value is None:
        return None
    from ..mcscf.casci import _angular_momentum          # one resolver, project-wide
    return _angular_momentum(value)


@dataclass(frozen=True)
class Target:
    """Base of the five classes: the class name, its priority, and its statement."""

    @property
    def cls(self) -> str:                                # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def priority(self) -> int:
        return PRIORITY[self.cls]

    @property
    def fixed(self) -> bool:
        """Whether the class is taken whole or not at all (no pruning of its candidates)."""
        return False

    def spelling(self):                                  # pragma: no cover - abstract
        raise NotImplementedError

    def statement(self, labels: Sequence[str] = ()) -> str:   # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True)
class Shell:
    """A centre's whole valence shell of one angular momentum.

    ``atoms = None`` is the default target set: every centre detection finds
    (:mod:`kuiva.autocas.centres`). ``l = None`` is read off the centre's own free-atom
    reference -- the channel that is *open* there -- never from a table of element blocks.

    ⚠ **Never pruned and never cut.** The count is ``2l+1`` pairs per centre because that is
    what a shell is; the empty members of a d manifold *are* the ligand-field spectrum, so an
    occupation or entropy criterion that drops them is measuring the probe and not the
    physics. A shell that does not fit the budget is a refusal, not a smaller shell.
    """

    atoms: Optional[Tuple[object, ...]] = None
    l: Optional[int] = None

    @property
    def cls(self) -> str:
        return "shell"

    @property
    def priority(self) -> int:
        return PRIORITY["shell"]

    @property
    def fixed(self) -> bool:
        return True

    @property
    def detected(self) -> bool:
        """Whether this target means "whatever centres were detected"."""
        return self.atoms is None

    def spelling(self):
        if self.atoms is None:
            return "shells"
        if self.l is None:
            return ("shell", _one_key(self.atoms))
        return ("shell", _one_key(self.atoms), angular_momentum_letter(self.l))

    def statement(self, labels: Sequence[str] = ()) -> str:
        where = " and ".join(labels) if labels else "every detected centre"
        shell = ("the valence shell" if self.l is None
                 else "the valence {} shell".format(angular_momentum_letter(self.l)))
        return "{} of {}".format(shell, where)


@dataclass(frozen=True)
class Frontier:
    """A fragment's singly occupied orbitals -- the radical HOMO -- and its neighbours.

    ``n_occ`` occupied and ``n_vir`` empty pairs of largest population on the fragment are
    added beside the singly occupied ones; both default to zero, i.e. *the SOMOs only*, which
    is the statement "this bridge is a radical and its unpaired electron is in the active
    space".
    """

    atoms: Tuple[object, ...]
    n_occ: int = 0
    n_vir: int = 0

    @property
    def cls(self) -> str:
        return "frontier"

    @property
    def priority(self) -> int:
        return PRIORITY["frontier"]

    @property
    def fixed(self) -> bool:
        return False

    def spelling(self):
        if self.n_occ or self.n_vir:
            return ("frontier", _one_key(self.atoms), int(self.n_occ), int(self.n_vir))
        return ("frontier", _one_key(self.atoms))

    def statement(self, labels: Sequence[str] = ()) -> str:
        where = "+".join(labels) if labels else "the fragment"
        text = "the singly occupied Kramers pairs on {}".format(where)
        if self.n_occ or self.n_vir:
            text += (" plus the {} occupied and {} empty pairs of largest population there"
                     .format(self.n_occ, self.n_vir))
        return text


@dataclass(frozen=True)
class Bridge:
    """Ligand orbitals that carry the interaction between two centres.

    ``sites`` are the two centres (each an atom key or a fragment of them); ``atoms`` are the
    bridging atoms, or ``None`` to detect them by contact with both sites.

    ⚠ **Its candidates are ranked by mutual information with *both* sites' shells, never by
    entropy in absolute terms**: a low-dimensional bridge shares at most ``ln 2`` with either
    ion, so an absolute entanglement cut ranks a real superexchange pathway below every
    metal orbital. The ranking is relative and the keep/drop decision is the probe's
    spectrum.
    """

    sites: Tuple[Tuple[object, ...], Tuple[object, ...]]
    atoms: Optional[Tuple[object, ...]] = None
    n_pairs: int = DEFAULT_BRIDGE_PAIRS

    @property
    def cls(self) -> str:
        return "bridge"

    @property
    def priority(self) -> int:
        return PRIORITY["bridge"]

    @property
    def fixed(self) -> bool:
        return False

    def spelling(self):
        sites = tuple(_one_key(s) for s in self.sites)
        if self.atoms is None and self.n_pairs == DEFAULT_BRIDGE_PAIRS:
            return ("bridge", sites)
        if self.n_pairs == DEFAULT_BRIDGE_PAIRS:
            return ("bridge", sites, _one_key(self.atoms))
        return ("bridge", sites,
                None if self.atoms is None else _one_key(self.atoms), int(self.n_pairs))

    def statement(self, labels: Sequence[str] = ()) -> str:
        where = " and ".join(labels) if labels else "the bridging atoms"
        return ("at most {} Kramers pairs on {}, occupied and empty, of largest population "
                "there".format(self.n_pairs, where))


@dataclass(frozen=True)
class Bonding:
    """The metal-ligand bonding combinations that carry the centre's shell character.

    The pairs immediately *below* the shell's projection cut, together with their antibonding
    partners: the covalent admixture a shell-only active space leaves out, and the class that
    is about dynamic correlation rather than about the mechanism -- which is why it is tested
    last and dropped first.
    """

    atoms: Tuple[object, ...]
    n_pairs: int = DEFAULT_BONDING_PAIRS

    @property
    def cls(self) -> str:
        return "bonding"

    @property
    def priority(self) -> int:
        return PRIORITY["bonding"]

    @property
    def fixed(self) -> bool:
        return False

    def spelling(self):
        if self.n_pairs == DEFAULT_BONDING_PAIRS:
            return ("bonding", _one_key(self.atoms))
        return ("bonding", _one_key(self.atoms), int(self.n_pairs))

    def statement(self, labels: Sequence[str] = ()) -> str:
        where = " and ".join(labels) if labels else "the centre"
        return ("at most {} Kramers pairs of {} below the shell's projection cut, with their "
                "antibonding partners".format(self.n_pairs, where))


@dataclass(frozen=True)
class DoubleShell:
    """The correlating shell of the same ``l`` -- the ``4f`` shell's ``5f``-like partner.

    ⚠ **Accepted or dropped whole, and never by what it carries.** At the reference it is
    empty and at the cheap CI's level it stays empty, which is a structural blindness rather
    than a threshold to lower: the only evidence about a correlating shell is what it does to
    the *spectrum*.
    """

    atoms: Tuple[object, ...]
    l: Optional[int] = None

    @property
    def cls(self) -> str:
        return "double"

    @property
    def priority(self) -> int:
        return PRIORITY["double"]

    @property
    def fixed(self) -> bool:
        return True

    def spelling(self):
        if self.l is None:
            return ("double", _one_key(self.atoms))
        return ("double", _one_key(self.atoms), angular_momentum_letter(self.l))

    def statement(self, labels: Sequence[str] = ()) -> str:
        where = " and ".join(labels) if labels else "the centre"
        shell = ("the correlating shell" if self.l is None
                 else "the correlating {} shell".format(angular_momentum_letter(self.l)))
        return "{} of {}".format(shell, where)


_SPELLINGS = """    "shells"
    ("shell", atoms)                      ("shell", atoms, l)
    ("frontier", atoms)                   ("frontier", atoms, n_occ, n_vir)
    ("bridge", (site_a, site_b))          ("bridge", (site_a, site_b), atoms[, n_pairs])
    ("bonding", atoms)                    ("bonding", atoms, n_pairs)
    ("double", atoms)                     ("double", atoms, l)"""


def _count(value, what: str) -> int:
    """A non-negative count, from anything integral (a numpy integer included)."""
    n = int(value)
    if n < 0:
        raise ValueError("{} cannot be negative; got {!r}".format(what, value))
    return n


def parse(spec):
    """One target spelling -> its dataclass. A target instance passes through unchanged."""
    if isinstance(spec, (Shell, Frontier, Bridge, Bonding, DoubleShell)):
        return spec
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in ("shells", "shell"):
            return Shell()
        raise ValueError(
            "{!r} is not a target. The only bare spelling is \"shells\" (every detected d/f "
            "centre's valence shell); everything else names what it applies to:\n{}"
            .format(spec, _SPELLINGS))
    try:
        parts = tuple(spec)
    except TypeError:
        raise ValueError("a target is a string or a tuple, not {!r}".format(spec))
    if not parts or not isinstance(parts[0], str):
        raise ValueError("a target tuple starts with its class name; got {!r}\n{}"
                         .format(spec, _SPELLINGS))
    cls, rest = parts[0].strip().lower(), parts[1:]
    if cls in ("shell", "shells"):
        if not rest:
            return Shell()
        if len(rest) == 1:
            return Shell(atoms=_keys(rest[0]))
        if len(rest) == 2:
            return Shell(atoms=_keys(rest[0]), l=_ell(rest[1]))
        raise ValueError('("shell", atoms[, l]); got {!r}'.format(spec))
    if cls == "frontier":
        if len(rest) == 1:
            return Frontier(atoms=_keys(rest[0]))
        if len(rest) == 3:
            return Frontier(atoms=_keys(rest[0]), n_occ=_count(rest[1], "n_occ"),
                            n_vir=_count(rest[2], "n_vir"))
        raise ValueError('("frontier", atoms) or ("frontier", atoms, n_occ, n_vir); got {!r}'
                         .format(spec))
    if cls == "bridge":
        if not 1 <= len(rest) <= 3:
            raise ValueError('("bridge", (site_a, site_b)[, atoms[, n_pairs]]); got {!r}'
                             .format(spec))
        sites = tuple(rest[0])
        if len(sites) != 2:
            raise ValueError(
                "a bridge sits between exactly two sites; got {!r}. A pathway between three "
                "centres is three bridges, and which pair each one connects is the statement"
                .format(rest[0]))
        atoms = None if len(rest) < 2 or rest[1] is None else _keys(rest[1])
        n_pairs = DEFAULT_BRIDGE_PAIRS if len(rest) < 3 else _count(rest[2], "n_pairs")
        return Bridge(sites=(_keys(sites[0]), _keys(sites[1])), atoms=atoms,
                      n_pairs=n_pairs)
    if cls == "bonding":
        if len(rest) == 1:
            return Bonding(atoms=_keys(rest[0]))
        if len(rest) == 2:
            return Bonding(atoms=_keys(rest[0]), n_pairs=_count(rest[1], "n_pairs"))
        raise ValueError('("bonding", atoms[, n_pairs]); got {!r}'.format(spec))
    if cls in ("double", "double-shell", "doubleshell"):
        if len(rest) == 1:
            return DoubleShell(atoms=_keys(rest[0]))
        if len(rest) == 2:
            return DoubleShell(atoms=_keys(rest[0]), l=_ell(rest[1]))
        raise ValueError('("double", atoms[, l]); got {!r}'.format(spec))
    raise ValueError(
        "unknown target class {!r}; the five classes are shell, frontier, bridge, bonding "
        "and double:\n{}".format(parts[0], _SPELLINGS))


def parse_targets(specs) -> Tuple[object, ...]:
    """A user's ``targets=`` into the fixed priority order (:data:`PRIORITY`).

    ``None`` is the default target set -- one :class:`Shell` over whatever centre detection
    finds. A single target may be given unwrapped; the ordering inside a class is the order
    it was written in, so two bridges stay in the order they were asked for.
    """
    if specs is None:
        return (Shell(),)
    if isinstance(specs, str) or isinstance(specs, (Shell, Frontier, Bridge, Bonding,
                                                    DoubleShell)):
        specs = [specs]
    else:
        try:
            specs = list(specs)
        except TypeError:
            raise ValueError("targets= is a target or a list of them; got {!r}".format(specs))
        # A single tuple spelling, e.g. ("shell", "Dy"), is one target and not a list of two.
        if specs and isinstance(specs[0], str) and specs[0].strip().lower() in PRIORITY:
            specs = [tuple(specs)]
    parsed = [parse(s) for s in specs]
    if not parsed:
        raise ValueError("targets= is empty: an active space is a physical statement and "
                         "there is no default that means nothing")
    order = sorted(range(len(parsed)), key=lambda i: (parsed[i].priority, i))
    return tuple(parsed[i] for i in order)


def resolve_atoms(layout, keys) -> Tuple[int, ...]:
    """Atom keys -> sorted 0-based atom indices. THE boundary between the two numberings.

    An element symbol resolves to **every** atom of that element (see the module docstring);
    a label or a number resolves to one atom. A key naming nothing in this molecule is
    refused with the elements and the atom range in the message -- a silently ignored target
    is an active space that is not the one that was asked for.
    """
    from ..basis.atommap import parse_atom_key
    from ..basis.ghosts import normalize_symbol

    symbols = [str(s) for s in layout.atom_symbols]
    caps = [normalize_symbol(s) for s in symbols]
    found = set()
    for key in _keys(keys):
        kind, target = parse_atom_key(key, symbols)
        if kind == "element":
            found.update(i for i, s in enumerate(caps) if s == target)
        else:
            found.add(int(target))
    if not found:
        raise ValueError("the key {!r} names no atom of this molecule".format(keys))
    return tuple(sorted(found))


def atom_labels(layout, atoms: Sequence[int]) -> Tuple[str, ...]:
    """``("1 Ti", "2 Ti")`` -- the output-side labels of resolved atom indices."""
    return tuple(layout.atom_label(int(a)) for a in atoms)
