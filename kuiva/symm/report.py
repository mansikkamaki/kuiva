"""The symmetry output block, and the lab-frame character table that makes labels mean something.

⚠ **A symmetry label is only as meaningful as the convention behind it, and the classic
failure is silent**: two programs agree on "C2h, Bu" and disagree on which Cartesian axis is
the two-fold one, so every number they exchange is subtly about a different calculation and
nothing raises. A run with symmetry on therefore prints the character table of the group
**actually used in the math**, with every operation named by its lab-frame geometry
(``C2(z)``, ``sigma(xy)``, ``i``) rather than by a Schoenflies label whose orientation the
reader has to guess, so the axis convention is established by inspection.

Three properties of the table are load-bearing and are asserted by the suite rather than
described here:

* it carries **both** the boson and the fermion (double-valued) irreps, because the spinor
  labels live in the fermion rows and a table that stopped at the boson ones would not
  contain the vocabulary the selection surfaces use;
* the irrep names in it are **byte-identical** to the names used in every selection request,
  refusal message and state table;
* the characters printed are reproduced by ``tr U(g)`` computed from the run's own AO
  operator matrices on the run's own labelled orbitals, so the printed conventions and the
  arithmetic cannot drift apart.

Characters are ASCII, because the whole output stream is: for the groups here they are always
fourth roots of unity and print as ``1``, ``-1``, ``i``, ``-i``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..util import output as out
from ..util.logging import get_logger
from .groups import Group

log = get_logger(__name__)

#: ASCII rendering of the purely imaginary units, which have no numeric spelling that reads
#: like a character. Everything else falls out of the rules in :func:`character_text`.
_CHARACTER_TEXT = {(0, 1): "i", (0, -1): "-i"}

#: How far a character may be from a whole number (or from zero) before it is printed as a
#: decimal. Characters are algebraic integers, so a genuine value is either exact to roundoff
#: or genuinely irrational (``sqrt(3)`` on a threefold axis); nothing lands in between.
_CHARACTER_TOL = 1.0e-9


def character_text(chi: complex) -> str:
    """One character, in ASCII.

    Integers print as integers, the imaginary units as ``i`` and ``-i``, and anything else —
    a ``sqrt(3)`` on a threefold axis, say — as a signed decimal. ⚠ The whole output stream is
    ASCII: a character table full of Unicode is a table nobody can grep over ssh.
    """
    chi = complex(chi)
    real, imaginary = chi.real, chi.imag
    if abs(imaginary) < _CHARACTER_TOL:
        rounded = int(np.rint(real))
        return str(rounded) if abs(real - rounded) < _CHARACTER_TOL else "{:.3f}".format(real)
    if abs(real) < _CHARACTER_TOL:
        key = (0, int(np.rint(imaginary)))
        if abs(imaginary - key[1]) < _CHARACTER_TOL and key in _CHARACTER_TEXT:
            return _CHARACTER_TEXT[key]
        return "{:.3f}i".format(imaginary)
    return "{:.3f}{:+.3f}i".format(real, imaginary)


def character_table(group: Group, logger=None, *, title: Optional[str] = None) -> None:
    """Print ``group``'s double-group character table, boson rows then fermion rows."""
    logger = logger or log
    elements = group.elements()
    width = max(6, max(len(group.element_name(e)) for e in elements) + 1)
    columns = [out.Column("irrep", "{}", max(8, max(len(n) for n in group.names.values())),
                          align="<")]
    columns += [out.Column(group.element_name(e), "{}", width) for e in elements]
    table = out.Table(logger, columns)
    table.start(title or "character table of {} (lab frame; the group used in the math)"
                .format(group.name))
    for label in group.labels():
        table.row(group.irrep_name(label),
                  *[character_text(group.character(label, e)) for e in elements])
    table.end("rows above the fermion block are single-valued (boson) irreps; the rows with "
              "chi(Ebar) = -1 are the spinor irreps")


def double_character_table(group, logger=None, *, title: Optional[str] = None) -> None:
    """Print a **full** double group's character table, boson rows then fermion rows.

    The non-abelian sibling of :func:`character_table`, and the second of the three tables a
    classifying run prints. Its rows are classes rather than elements — a non-abelian group's
    characters are class functions and printing one column per element would print the same
    number several times and invite the reader to think the operations differ.

    ⚠ Class names are lab-frame geometry throughout, exactly as in the abelian table, and the
    names of the irreps are the same bytes used in every classification and every refusal.
    """
    logger = logger or log
    names = [group.class_name(k) for k in range(group.n_irreps)]
    # Each column sized to its own header rather than to the widest of them: a full double
    # group has up to eighteen classes and a uniform width would push the table far past the
    # width of the output stream for no reason.
    columns = [out.Column("irrep", "{}", max(8, max(len(n) for n in group.irrep_names)),
                          align="<")]
    columns += [out.Column(n, "{}", max(len(n), 6) + 1) for n in names]
    table = out.Table(logger, columns)
    table.start(title or "character table of the double group of {} (lab frame; the group "
                "used in the CLASSIFICATION)".format(group.name))
    for r in range(group.n_irreps):
        table.row(group.irrep_names[r],
                  *[character_text(complex(group.characters[r, k]))
                    for k in range(group.n_irreps)])
    table.end("rows with chi(Ebar) = -dim are the spinor (double-valued) irreps; the "
              "mathematics of the calculation runs in the abelian subgroup above, not here")


def correspondence_table(group, abelian, logger=None) -> None:
    """Print the computed subduction of each full-group irrep onto the abelian sectors.

    ⚠ **This is the row a user actually needs**, and it is computed rather than transcribed
    (:meth:`kuiva.symm.double.DoubleGroup.subduction`): it says which abelian sectors a
    physical multiplet is spread over, and therefore what a per-irrep ``n_states`` request
    selects. Where a multiplet subduces to *more than one* abelian sector, an abelian count
    can cut it in half and every abelian check still passes.
    """
    logger = logger or log
    sub = group.subduction(abelian)
    entries = []
    for r in range(group.n_irreps):
        entries.append(" + ".join(
            ("{} x ".format(m) if m > 1 else "") + abelian.irrep_name(t)
            for t, m in sub[r].items()) or "-")
    table = out.Table(logger, [
        out.Column("irrep of " + group.name, "{}", max(14, max(len(n) for n in
                                                               group.irrep_names) + 2),
                   align="<"),
        out.col_count("dim", 5),
        out.Column("sectors of " + abelian.name, "{}", max(20, max(map(len, entries)) + 1),
                   align="<")])
    table.start("correspondence between the classification group and the label group "
                "(computed subduction)")
    for r in range(group.n_irreps):
        table.row(group.irrep_names[r], int(group.dimensions[r]), entries[r])
    table.end("a multiplet spread over more than one sector is one a per-irrep state count "
              "can cut in half with every abelian check still passing")


def report(symmetry, logger=None, *, spinor_labels=None) -> None:
    """The symmetry section of the output: what group, from what, and how well it held."""
    logger = logger or log
    group = symmetry.group
    out.subsection(logger, "Molecular symmetry")
    rows = [
        ("point group requested", symmetry.requested),
        ("operations detected", ", ".join(symmetry.detected) or "none"),
        ("label group", group.name, "",
         "generators " + ", ".join("{}(order {})".format(g.name, g.modulus)
                                   for g in group.generators)),
        ("frame", "input geometry", "", "the molecule is never reoriented"),
    ]
    out.entries(logger, rows)
    if symmetry.reduced_from:
        logger.warning("the double group of %s is not abelian (its fermion irreps are "
                       "two-dimensional), so the labels come from its largest subgroup with "
                       "one-dimensional fermion irreps, %s; physically degenerate partners "
                       "may then carry different labels and a per-irrep state count can cut "
                       "a degenerate manifold in half",
                       symmetry.reduced_from, group.name)
    if symmetry.unused:
        out.entry(logger, "operations not used", ", ".join(symmetry.unused), "",
                  "present in the geometry, outside the label group")
    rep = symmetry.report
    out.entries(logger, [
        ("degenerate blocks adapted", rep.n_adapted, "",
         "rotated inside a degenerate block; no observable changes"),
        ("largest block mixing removed", rep.max_rotation, "", "", "{:.2e}"),
        ("worst off-diagonal residual", rep.max_residual, "", "of C^T S U(g) C", "{:.2e}"),
    ])
    character_table(group, logger)
    full = getattr(symmetry, "full_group", None)
    if full is not None:
        out.entries(logger, [
            ("classification group", full.name, "",
             "non-abelian labels on converged states; the math stays abelian"),
            ("classification group order", full.order),
        ])
        double_character_table(full, logger)
        correspondence_table(full, group, logger)
    if spinor_labels is not None:
        sector_table(spinor_labels, logger)


def sector_table(labels, logger=None, *, title: str = "spinors per fermion irrep") -> None:
    """How many orbitals carry each fermion label — the shape of every sector below."""
    logger = logger or log
    group = labels.group
    counts = labels.counts()
    table = out.Table(logger, [out.Column("irrep", "{}", 10, align="<"),
                               out.col_count("spinors", 8),
                               out.Column("conjugate", "{}", 10)])
    table.start(title)
    for label in group.labels(fermion=True):
        table.row(group.irrep_name(label), counts.get(label, 0),
                  group.irrep_name(group.conjugate(label)))
    table.end()


__all__ = ["character_table", "character_text", "correspondence_table",
           "double_character_table", "report", "sector_table"]
