"""Term and level assignment: an **offer**, with the evidence that produced it.

What this does, and what it deliberately does not
-------------------------------------------------
Reading a converged two-component spectrum means deciding which ``^{2S+1}L_J`` level each
degenerate block is. That decision is an *inference* from three measured quantities — the
block dimension, its ``<S^2>`` (:mod:`kuiva.props.spin`) and its isotropic g value
(:mod:`kuiva.props.multiplet`) — and inference is exactly the kind of thing this program does
not print as if it were a computed number.

So the assignment lives here, in its own report, and it never enters the per-state energy
table. Every row carries the evidence beside the label and a **fit residual** saying how well
the label actually explains the numbers; a block whose evidence does not add up is labelled
``?`` rather than given the closest plausible term. A reader can therefore disagree with any
row from what is printed next to it, which is the whole point.

How the inference runs
----------------------
**Spin-orbit coupling off.** Spin is a good quantum number, so a block of a free ion is a
``^{2S+1}L`` term of dimension ``(2S+1)(2L+1)``::

    S from <S^2> = S(S+1)         then      2L+1 = size / (2S+1)

Both must come out integral, which is a real constraint: two of the three numbers determine
the third and the residual is a genuine check.

**Spin-orbit coupling on.** ``S`` is no longer conserved, and the block dimension is ``2J+1``::

    J = (size - 1) / 2
    S from <S^2>                                        (now a *purity* reading)
    L from the Lande factor, inverted:
        g_J = 3/2 + [S(S+1) - L(L+1)] / [2 J(J+1)]
        =>  L(L+1) = S(S+1) - (g_J - 3/2) * 2 J(J+1)

⚠ **The inversion is only as good as the free-ion picture behind it.** A crystal-field-split
level of a complex is not a ``2J+1`` manifold at all, so ``J`` from its dimension is a fiction
and the inverted ``L`` comes out non-integral — which is the correct outcome, reported as
``?``, and is why the residual is printed rather than hidden. The triangle condition
``|L - S| <= J <= L + S`` is checked too, and a block failing it is refused a label.

⚠ The tolerances below say what may be *printed*, not what is physically true
-----------------------------------------------------------------------------
They are deliberately loose. A label is a suggestion a human confirms; a tight tolerance here
would suppress the offer on exactly the interesting cases (a strongly mixed level) while
adding no correctness. What must never happen is the opposite — a label printed with no way to
see how badly it fits — and that is what the evidence and residual columns are for.

References
----------
* Russell-Saunders terms, the Lande factor and the ``2J+1`` counting inverted here: standard
  atomic theory, e.g. R. D. Cowan, "The Theory of Atomic Structure and Spectra", Univ.
  California Press (1981), ch. 4 and 11.
* The g values the inversion consumes follow the Chibotaru-Ungur construction cited in
  :mod:`kuiva.props.multiplet`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util.logging import get_logger
from .multiplet import Multiplet, lande_g

log = get_logger(__name__)

#: Spectroscopic term letters, ``L = 0, 1, 2, ...``. ⚠ ``J`` is skipped, by convention, so
#: this may not be generated from the alphabet.
TERM_LETTERS = "SPDFGHIKLMNOQRTUV"

#: ⚠ **Advisory: what may be printed, not a physical tolerance** (module docstring). Largest
#: deviation of ``2S+1`` from an integer for a spin label to be offered.
SPIN_ROUND_TOL = 0.05

#: ⚠ Advisory. Largest deviation of the inferred ``L`` from an integer for a term letter to be
#: offered.
ORBITAL_ROUND_TOL = 0.10

#: ⚠ Advisory. Largest ``|g_measured - g_Lande(L, S, J)|`` for the assignment to be called a
#: fit. Generous on purpose: a real free-ion g carries basis and picture-change error of order
#: 1e-3 — the bare non-relativistic ``L`` and ``S`` used unchanged in the two-component basis
#: are measured at 1e-4 to 1e-3 relative on free-ion g factors — and a *mixed* level departs
#: much further while still being worth offering as the dominant term.
LANDE_FIT_TOL = 0.05


def term_letter(l: int) -> str:
    """``L`` as its spectroscopic letter. Raises above the tabulated range rather than guess."""
    l = int(l)
    if not 0 <= l < len(TERM_LETTERS):
        raise ValueError("no tabulated term letter for L = {}".format(l))
    return TERM_LETTERS[l]


def _fmt_half(x: float) -> str:
    """``5/2`` for a half-integer, ``2`` for an integer — how a J is written."""
    twice = int(round(2.0 * x))
    return "{:d}".format(twice // 2) if twice % 2 == 0 else "{:d}/2".format(twice)


@dataclass(frozen=True)
class TermAssignment:
    """One degenerate block's offered label and the evidence for it.

    Attributes
    ----------
    block, start, size : int — index of the block, of its first state, and its dimension.
    energy_cm : float — block energy above the ground state.
    s_squared : float — the measured block ``<S^2>``.
    spin, orbital, j : float / int / float or ``None``
        ``S``, ``L`` and ``J`` as inferred. ``None`` where the evidence did not support one.
    g_iso : float — the measured isotropic g, or ``nan`` (spin-free, or a block with no
        moment).
    label : str — ``"^3F_4"``-style, or ``"?"``. ⚠ **An inference.**
    residual : float
        How badly the label fits: the Lande residual with spin-orbit coupling on, the
        dimension mismatch ``|size - (2S+1)(2L+1)|`` with it off. ``nan`` for ``"?"``.
    evidence : tuple of str — one line per step of the inference, printed with the row.
    irrep : str — symmetry label of the block where one was carried, else ``""``.
    """

    block: int
    start: int
    size: int
    energy_cm: float
    s_squared: float
    spin: Optional[float] = None
    orbital: Optional[int] = None
    j: Optional[float] = None
    g_iso: float = float("nan")
    label: str = "?"
    residual: float = float("nan")
    evidence: Tuple[str, ...] = ()
    irrep: str = ""

    @property
    def assigned(self) -> bool:
        return self.label != "?"


@dataclass(frozen=True)
class Assignment:
    """The assignment offer for a whole spectrum. ⚠ Inference throughout; see :meth:`report`."""

    terms: Tuple[TermAssignment, ...]
    has_soc: bool = True

    def labels(self) -> Tuple[str, ...]:
        """One label per block, ``"?"`` where none was offered."""
        return tuple(t.label for t in self.terms)

    def report(self, logger=None, evidence: bool = True) -> None:
        """The assignment table — ⚠ **offers**, never the standard state table.

        ``evidence=False`` prints the table alone; by default each row is followed by the
        measurements the label was inferred from, because a label without them is folklore.
        """
        logger = logger or log
        out.subsection(logger, "Term assignment (inferred)")
        # ASCII only in the output stream: this file is tailed and diffed over ssh.
        out.note(logger, "every label below is an INFERENCE from the block dimension, "
                         "<S^2>{} -- not a computed quantum number"
                 .format(" and g" if self.has_soc else ""))
        if self.has_soc:
            out.note(logger, "levels ^(2S+1)L_J; S is not conserved with spin-orbit "
                             "coupling on, so 2S+1 names the dominant spin")
        else:
            out.note(logger, "terms ^(2S+1)L; spin-orbit coupling is off, so S is exact")
        # ⚠ The g column exists only where g is part of the inference. Spin-free, ``L`` comes
        # from the dimension and ``<S^2>`` alone, so a g column there would be a column of
        # ``nan`` under a heading suggesting a measurement that was never made.
        columns = [out.col_count("block", 7), out.Column("states", "{:d}", 8),
                   out.Column("E [cm^-1]", out.CM_FMT, 14),
                   out.Column("<S^2>", "{:.4f}", 10)]
        if self.has_soc:
            columns.append(out.Column("g_iso", "{:.4f}", 10))
        columns += [out.Column("assignment", "{:s}", 12), out.Column("fit", "{:.1e}", 9)]
        table = out.Table(logger, columns).start()
        for t in self.terms:
            row = [t.block + 1, t.size, t.energy_cm, t.s_squared]
            if self.has_soc:
                row.append(t.g_iso)
            row += [t.label + (" [{}]".format(t.irrep) if t.irrep else ""), t.residual]
            table.row(*row)
        table.end("{} of {} blocks assigned".format(
            sum(1 for t in self.terms if t.assigned), len(self.terms)))
        if evidence:
            for t in self.terms:
                for line in t.evidence:
                    out.note(logger, "block {}: {}".format(t.block + 1, line))

    def __repr__(self) -> str:
        return "Assignment({})".format(", ".join(self.labels()))


def _spin_from_block(s_squared: float) -> Tuple[float, float, str]:
    """``(S, |deviation of 2S+1 from integer|, evidence)`` for one block."""
    from .spin import spin_from_s_squared
    s = spin_from_s_squared(s_squared)
    mult = 2.0 * s + 1.0
    dev = abs(mult - round(mult))
    return (round(mult) - 1.0) / 2.0, dev, (
        "<S^2> = {:.6f} -> S = {:.4f}, 2S+1 = {:.4f} (off an integer by {:.1e})"
        .format(s_squared, s, mult, dev))


def _assign_spin_free(block: int, m: Multiplet, s_squared: float,
                      irrep: str) -> TermAssignment:
    """``^{2S+1}L`` from ``<S^2>`` and the block dimension — the two fix the third."""
    spin, dev, ev_spin = _spin_from_block(s_squared)
    evidence = [ev_spin]
    base = dict(block=block, start=m.start, size=m.size, energy_cm=m.energy_cm,
                s_squared=s_squared, irrep=irrep)
    if dev > SPIN_ROUND_TOL:
        evidence.append("2S+1 is not integral to {:.2g}, so no term is offered"
                        .format(SPIN_ROUND_TOL))
        return TermAssignment(spin=spin, evidence=tuple(evidence), **base)
    mult = int(round(2.0 * spin + 1.0))
    twice_l_plus = m.size / mult
    l = (twice_l_plus - 1.0) / 2.0
    evidence.append("dimension {} / (2S+1 = {}) = {:.4f} = 2L+1 -> L = {:.4f}"
                    .format(m.size, mult, twice_l_plus, l))
    if abs(l - round(l)) > ORBITAL_ROUND_TOL or l < 0:
        evidence.append("L is not a non-negative integer, so this block is not one free-ion "
                        "term (a split or accidentally degenerate group would look like this)")
        return TermAssignment(spin=spin, evidence=tuple(evidence), **base)
    l_int = int(round(l))
    residual = abs(m.size - mult * (2 * l_int + 1))
    return TermAssignment(spin=spin, orbital=l_int, residual=float(residual),
                          label="^{}{}".format(mult, term_letter(l_int)),
                          evidence=tuple(evidence), **base)


def _assign_soc(block: int, m: Multiplet, s_squared: float, irrep: str) -> TermAssignment:
    """``^{2S+1}L_J`` from the block dimension, ``<S^2>`` and the inverted Lande factor."""
    spin, dev, ev_spin = _spin_from_block(s_squared)
    j = m.j
    g_iso = m.g_iso
    evidence = ["dimension {} = 2J+1 -> J = {}".format(m.size, _fmt_half(j)), ev_spin]
    base = dict(block=block, start=m.start, size=m.size, energy_cm=m.energy_cm,
                s_squared=s_squared, g_iso=g_iso, j=j, irrep=irrep)
    if dev > SPIN_ROUND_TOL:
        evidence.append("2S+1 is not integral to {:.2g} -- a strongly spin-mixed level; no "
                        "term is offered".format(SPIN_ROUND_TOL))
        return TermAssignment(spin=spin, evidence=tuple(evidence), **base)
    spin = (round(2.0 * spin + 1.0) - 1.0) / 2.0
    if m.non_kramers:
        # ⚠ A tunnelling-split pair of singlets grouped by an explicit request. Its g is the
        # Griffith / Abragam-Bleaney axial one with the transverse components zero by
        # convention, so the isotropic average is not a Lande factor and inverting it would
        # produce a number out of a convention rather than out of the states.
        evidence.append("this block is a non-Kramers pseudo-doublet (Delta = {:.4g} cm^-1): "
                        "its g_z is an effective-spin quantity, not a Lande factor, so no "
                        "free-ion level is offered"
                        .format(m.tunnelling_gap_cm or float("nan")))
        return TermAssignment(spin=spin, evidence=tuple(evidence), **base)
    if j <= 0.0 or not np.isfinite(g_iso):
        evidence.append("J = 0 carries no magnetic moment, so L cannot be inverted from g")
        return TermAssignment(spin=spin, evidence=tuple(evidence), **base)
    # g_J = 3/2 + [S(S+1) - L(L+1)] / [2 J(J+1)]   ->   L(L+1)
    ll = spin * (spin + 1.0) - (g_iso - 1.5) * 2.0 * j * (j + 1.0)
    if ll < -0.25:
        evidence.append("the Lande inversion gives L(L+1) = {:.4f} < 0: this block is not a "
                        "free-ion J multiplet".format(ll))
        return TermAssignment(spin=spin, evidence=tuple(evidence), **base)
    l = 0.5 * (np.sqrt(max(0.0, 1.0 + 4.0 * ll)) - 1.0)
    evidence.append("g_iso = {:.6f} -> L(L+1) = {:.4f} -> L = {:.4f} (off an integer by "
                    "{:.1e})".format(g_iso, ll, l, abs(l - round(l))))
    if abs(l - round(l)) > ORBITAL_ROUND_TOL:
        evidence.append("L is not integral to {:.2g}, so no term is offered"
                        .format(ORBITAL_ROUND_TOL))
        return TermAssignment(spin=spin, evidence=tuple(evidence), **base)
    l_int = int(round(l))
    if not (abs(l_int - spin) - 1e-9 <= j <= l_int + spin + 1e-9):
        evidence.append("the triangle condition |L-S| <= J <= L+S fails for L = {}, S = {}, "
                        "J = {}".format(l_int, _fmt_half(spin), _fmt_half(j)))
        return TermAssignment(spin=spin, orbital=l_int, evidence=tuple(evidence), **base)
    residual = abs(g_iso - lande_g(float(l_int), spin, j))
    evidence.append("Lande g({}{}_{}) = {:.6f} vs measured {:.6f}: residual {:.2e}"
                    .format(int(round(2 * spin + 1)), term_letter(l_int), _fmt_half(j),
                            lande_g(float(l_int), spin, j), g_iso, residual))
    if residual > LANDE_FIT_TOL:
        evidence.append("residual above {:.2g}, so the label is withheld"
                        .format(LANDE_FIT_TOL))
        return TermAssignment(spin=spin, orbital=l_int, residual=float(residual),
                              evidence=tuple(evidence), **base)
    return TermAssignment(
        spin=spin, orbital=l_int, residual=float(residual),
        label="^{}{}_{}".format(int(round(2 * spin + 1)), term_letter(l_int), _fmt_half(j)),
        evidence=tuple(evidence), **base)


def assign_terms(multiplets: Sequence[Multiplet], spin, *,
                 irreps: Optional[Sequence[str]] = None) -> Assignment:
    """Offer a term (or level) label for each degenerate block.

    Parameters
    ----------
    multiplets : from :func:`kuiva.props.multiplet.analyse_spectrum` — the degeneracy pattern
        and the g values.
    spin : :class:`kuiva.props.spin.SpinAnalysis` — the block ``<S^2>``. Its blocking must be
        the *same* blocking, which is checked rather than assumed: the two are computed from
        the same energies at the same tolerance, and a mismatch means one of them was not.
    irreps : optional, one label per **state** (``CASCIResult.multiplets`` or
        ``CASCIResult.irreps``). A block gets a label only where its members agree, since a
        symmetry label is a property of the block (:mod:`kuiva.symm.classify`).
    """
    starts = [m.start for m in multiplets]
    if starts != [s for s, _ in spin.blocks]:
        raise ValueError(
            "the multiplet blocking {} and the <S^2> blocking {} differ; they must be the "
            "same grouping of the same spectrum. Use one tol_cm for both, and note that "
            "pseudo_doublet_tol_cm regroups the multiplets and not the spin blocks"
            .format(starts, [s for s, _ in spin.blocks]))
    labels: List[str] = []
    for m in multiplets:
        if irreps is None:
            labels.append("")
        else:
            names = {str(irreps[i]) for i in range(m.start, m.start + m.size)}
            labels.append(names.pop() if len(names) == 1 else "")
    terms = []
    for b, m in enumerate(multiplets):
        s2 = float(spin.block_s_squared[b])
        terms.append(_assign_soc(b, m, s2, labels[b]) if spin.has_soc
                     else _assign_spin_free(b, m, s2, labels[b]))
    log.debug("assignment: %s", [t.label for t in terms])
    return Assignment(terms=tuple(terms), has_soc=bool(spin.has_soc))


__all__ = ["Assignment", "LANDE_FIT_TOL", "ORBITAL_ROUND_TOL", "SPIN_ROUND_TOL",
           "TERM_LETTERS", "TermAssignment", "assign_terms", "term_letter"]
