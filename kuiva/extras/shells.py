"""Shell-resolved atomic configurations: ``(n, l, q)``, with the principal quantum number kept.

Why this exists beside :class:`kuiva.amf.configuration.AtomicConfiguration`
---------------------------------------------------------------------------
That class canonicalizes a configuration to **electrons per ``l`` channel** and discards the
principal quantum numbers deliberately: an atomic mean field depends on nothing more, and the
per-``l`` tuple is what makes two spellings of one reference share a cache entry.

A method that computes quantities **of individual shells** cannot use that form. ``4f9 5d1
6s1`` and ``4f9 5d1 6s1`` differ from ``4f10 6s1`` in the per-``l`` count, but ``5d1 6s1``
against ``4d1 6s1`` does not tell them apart at all once ``n`` is gone — and a Slater-Condon
parameter such as ``F^2(4f, 5d)`` names its shells. So this class keeps ``(n, l, q)`` and
converts *to* the per-``l`` form when it needs an atomic mean field, never back.

⚠ **One grammar, one aufbau rule.** The string is read by
:func:`kuiva.amf.configuration.parse_shell_terms`, shared with ``AtomicConfiguration.parse``,
so the two classes can never disagree about what ``"[Xe] 4f9 5d1 6s1"`` means. And what this
class accepts is exactly what survives the round trip through the per-``l`` form: every shell
below an occupied one of the same ``l`` present and full. A configuration outside that — a
core hole such as ``3d9 4d10`` — has no single-ensemble average-of-configuration
representation at all (you cannot spread a hole over a shell and stay spherical *and* keep the
shell below it full), so it is refused here rather than silently re-ordered downstream. The
refusal and the representability statement are therefore the same check, and
:meth:`to_atomic` is guaranteed lossless by construction.

Occupations are electron counts and integers. A *fractional* occupation is what an
average-of-configuration SCF produces from this object; it is not a way of stating one.

The other half: shells of a *converged* solution
------------------------------------------------
:func:`extract_shells` turns an average-of-configuration ``ScalarX2CData`` into one
:class:`ShellOrbitals` per named shell — a radial function, the ``2l+1`` orbitals built from
it, and the diagnostics that say whether the solution really has shells. The construction and
the trap it avoids are in that function's docstring; the short version is that the radial
function is read off the **Fock operator**, never off the degenerate MOs, so the ``2l+1``
orbitals of a shell are aligned to pure ``m`` by construction rather than by a post-hoc
rotation of an arbitrary basis of a degenerate manifold.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from kuiva.amf.configuration import (SHELL_LETTERS, AtomicConfiguration,
                                     angular_channel_groups, parse_shell_terms)
from kuiva.util import output as out
from kuiva.util.degeneracy import DEFAULT_GROUP_RTOL, group_bounds
from kuiva.util.logging import get_logger

log = get_logger(__name__)


def shell_capacity(l: int) -> int:
    """Electrons a shell of angular momentum ``l`` holds: ``2(2l+1)``."""
    return 2 * (2 * int(l) + 1)


def shell_label(n: int, l: int) -> str:
    """``(4, 3) -> "4f"``."""
    return "{}{}".format(int(n), SHELL_LETTERS[int(l)])


@dataclass(frozen=True)
class ShellConfiguration:
    """An atomic configuration as an ordered tuple of ``(n, l, q)`` shells.

    Canonical and hashable: shells are sorted by ``(n, l)``, repeated shells are summed, and
    empty ones are dropped, so ``"[Xe] 6s1 4f9 5d1"`` and ``"[Xe] 4f9 5d1 6s1 5f0"`` are the
    same object. ``label`` is provenance and is **excluded from equality and hashing**, as it
    is in :class:`kuiva.amf.configuration.AtomicConfiguration` and for the same reason: two
    descriptions of one electron distribution are one configuration.

    The charge is not stored. It is derived from the element (:meth:`charge`), which keeps the
    configuration the single source of truth about how many electrons are being solved for —
    the two can then never disagree.
    """

    shells: Tuple[Tuple[int, int, int], ...]

    def __init__(self, shells: Iterable[Sequence[int]], label: str = "") -> None:
        merged: Dict[Tuple[int, int], int] = {}
        for item in shells:
            n, l, q = (int(v) for v in item)
            if l < 0:
                raise ValueError("negative angular momentum in {!r}".format(item))
            if l >= len(SHELL_LETTERS):
                raise ValueError(
                    "angular momentum {} is past the {} shells this notation covers"
                    .format(l, SHELL_LETTERS))
            if n <= l:
                raise ValueError(
                    "there is no {}{} shell: the lowest shell of angular momentum {} is "
                    "{}.".format(n, SHELL_LETTERS[l], SHELL_LETTERS[l],
                                 shell_label(l + 1, l)))
            if q < 0:
                raise ValueError("negative occupation in {!r}".format(item))
            merged[(n, l)] = merged.get((n, l), 0) + q
        for (n, l), q in sorted(merged.items()):
            if q > shell_capacity(l):
                raise ValueError(
                    "{} holds at most {} electrons, not {}.".format(
                        shell_label(n, l), shell_capacity(l), q))
        occupied = {key: q for key, q in merged.items() if q}
        _check_aufbau(occupied)
        object.__setattr__(self, "shells",
                           tuple((n, l, q) for (n, l), q in sorted(occupied.items())))
        object.__setattr__(self, "_label", label)
        # ⚠ The backstop for the class contract, and cheap: the per-``l`` form must resolve
        # back to exactly these shells, or :meth:`to_atomic` would be lossy and every mean
        # field taken through it would be of a different state than the one named. The rule
        # above is what makes this hold; this is what makes "the rule above is right" a
        # tested statement rather than an argument.
        if self.to_atomic().shells() != self.shells:
            raise ValueError(
                "{} cannot be represented as electrons per angular-momentum channel, so no "
                "average-of-configuration reference exists for it. Every shell below an "
                "occupied shell of the same l must be present and full."
                .format(self.canonical))

    # -- basic properties ------------------------------------------------------------------

    @property
    def n_electrons(self) -> int:
        return int(sum(q for _, _, q in self.shells))

    @property
    def label(self) -> str:
        """Human-readable provenance. Not part of the identity — see the class docstring."""
        return getattr(self, "_label", "") or self.canonical

    @property
    def canonical(self) -> str:
        """``"1s2 2s2 ... 4f9 5d1 6s1"`` — the identity, written out."""
        return " ".join("{}{}".format(shell_label(n, l), q) for n, l, q in self.shells)

    @property
    def labels(self) -> Tuple[str, ...]:
        """``("1s", "2s", ..., "4f", "5d", "6s")`` — the occupied shells, in ``(n, l)`` order."""
        return tuple(shell_label(n, l) for n, l, _ in self.shells)

    def occupation(self, n: int, l: int) -> int:
        """Electrons in shell ``(n, l)``; ``0`` if the configuration does not occupy it."""
        for n_i, l_i, q in self.shells:
            if (n_i, l_i) == (int(n), int(l)):
                return q
        return 0

    def open_shells(self) -> Tuple[Tuple[int, int, int], ...]:
        """The partially filled shells, ``(n, l, q)`` each.

        These are the shells an average-of-configuration SCF occupies fractionally and the
        ones whose two-electron coupling coefficient is not 1
        (:attr:`kuiva.amf.configuration.OpenShell.coupling`).
        """
        return tuple((n, l, q) for n, l, q in self.shells if q < shell_capacity(l))

    @property
    def is_closed_shell(self) -> bool:
        return not self.open_shells()

    def charge(self, element: Union[str, int]) -> int:
        """``Z - N`` for this configuration on ``element``.

        Derived rather than stated, as it is in :func:`kuiva.amf.atomic.make_request`: the
        configuration says how many electrons there are, so a separately supplied charge could
        only ever contradict it.
        """
        from pyscf import gto

        z = int(gto.charge(element)) if isinstance(element, str) else int(element)
        return z - self.n_electrons

    # -- construction ----------------------------------------------------------------------

    @classmethod
    def parse(cls, text: str) -> "ShellConfiguration":
        """``"[Xe] 4f9 5d1 6s1"``, ``"1s2 2s1"``, ``"[Ar] 3d2 4s1"`` — all of these.

        The same grammar :meth:`kuiva.amf.configuration.AtomicConfiguration.parse` reads
        (:func:`kuiva.amf.configuration.parse_shell_terms`), with the principal quantum
        numbers kept instead of summed away.
        """
        return cls(parse_shell_terms(text), label=text.strip())

    @classmethod
    def coerce(cls, value: Union[str, "ShellConfiguration", Iterable[Sequence[int]]],
               element: Optional[str] = None) -> "ShellConfiguration":
        """Accept a string or a sequence of ``(n, l, q)`` and return the canonical object.

        ⚠ **An oxidation state is deliberately not accepted**, unlike
        :meth:`kuiva.amf.configuration.AtomicConfiguration.coerce`. Which shells an ion's
        electrons sit in is exactly the judgement the per-``l`` form does not have to make —
        Dy(I) is ``4f9 5d1 6s1`` in one place and ``4f10 6s1`` in another, and a method that
        computes shell-resolved quantities would be answering about a different state than
        the user meant. It is stated, never guessed.
        """
        if isinstance(value, ShellConfiguration):
            return value
        if isinstance(value, str):
            from kuiva.amf.configuration import _OXIDATION

            if _OXIDATION.match(value.strip()):
                raise ValueError(
                    "{!r} is an oxidation state, and a shell-resolved configuration is never "
                    "derived from one: which shells the electrons of, say, Dy(1+) occupy "
                    "(4f9 5d1 6s1 or 4f10 6s1) is chemistry, and the answer changes every "
                    "shell-resolved quantity. State the configuration, e.g. "
                    "'[Xe] 4f9 5d1 6s1'.".format(value))
            return cls.parse(value)
        return cls(value)

    def to_atomic(self) -> AtomicConfiguration:
        """The per-``l`` configuration, for the atomic mean field and its cache key.

        Lossless in this direction and checked to be so at construction (class docstring), so
        an AMF solve keyed on it is a solve of *this* configuration. The inverse,
        :meth:`AtomicConfiguration.shells`, is a derivation under the aufbau assumption and is
        not part of any contract outside this class.
        """
        occ: List[int] = []
        for _, l, q in self.shells:
            while len(occ) <= l:
                occ.append(0)
            occ[l] += q
        return AtomicConfiguration(occ, label=self.label)

    # -- reporting -------------------------------------------------------------------------

    def as_dict(self) -> Dict[str, int]:
        """``{"4f": 9, "5d": 1, "6s": 1}`` — for reference records and JSON provenance."""
        return {shell_label(n, l): q for n, l, q in self.shells}

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return "ShellConfiguration({}, {} electrons{})".format(
            self.canonical, self.n_electrons,
            "" if self.is_closed_shell else ", open: " + ", ".join(
                "{}^{}".format(shell_label(n, l), q) for n, l, q in self.open_shells()))


def _check_aufbau(occupied: Dict[Tuple[int, int], int]) -> None:
    """Refuse a hole below an occupied shell of the same ``l``, naming the channel.

    Two ways it can happen and both are refused: a **missing** lower shell (``1s2 3s1`` — no
    2s) and a **partially filled** one (``3d9 4d1``). The physical statement is the same
    either way — the configuration is not an aufbau filling of its channels, so there is no
    single spherical ensemble averaging it, and the per-``l`` electron count every atomic
    mean field in this project is built from would put the electrons somewhere else.
    """
    channels: Dict[int, List[int]] = {}
    for (n, l) in occupied:
        channels.setdefault(l, []).append(n)
    for l, ns in sorted(channels.items()):
        ns.sort()
        expected = list(range(l + 1, l + 1 + len(ns)))
        if ns != expected:
            raise ValueError(
                "the occupied {} shells are {} but an aufbau filling of that channel is {}: a "
                "hole below an occupied shell of the same l has no average-of-configuration "
                "representation.".format(
                    SHELL_LETTERS[l], ", ".join(shell_label(n, l) for n in ns),
                    ", ".join(shell_label(n, l) for n in expected)))
        for n in ns[:-1]:
            if occupied[(n, l)] != shell_capacity(l):
                raise ValueError(
                    "{}^{} is below the occupied {} shell and is not full ({} electrons of "
                    "{}): a hole below an occupied shell of the same l has no "
                    "average-of-configuration representation.".format(
                        shell_label(n, l), occupied[(n, l)], shell_label(ns[-1], l),
                        occupied[(n, l)], shell_capacity(l)))


#: Largest relative variation the Fock blocks of one ``l`` may show across its ``m`` channels
#: before :func:`extract_shells` warns. An atom's Fock operator in real spherical harmonics is
#: block diagonal in ``(l, m)`` with **m-independent** blocks — that is symmetry, not
#: convergence — so this is how spherical the converged solution actually is, measured on the
#: operator the shells are read off rather than on the density.
#:
#: ⚠ **It does not go to zero, and the floor is the Fock reconstruction rather than the
#: physics**: ``F = S C diag(eps) C^T S`` inherits the conditioning of the overlap, so a heavy
#: atom sits higher than a light one at the same convergence. Measured on converged
#: average-of-configuration solutions in ``x2c-SVPall-2c``: **2e-11** (O 2p4), **5e-12** (Ti
#: 3d1), **7e-9** (Ce(3+) 4f1, cond(S) = 6e4), unchanged by tightening ``conv_tol`` by two
#: decades. Against **0.27** for a symmetry-broken ROHF oxygen — nine orders of separation, so
#: the bound is not delicate and is set well above the floor.
SHELL_ANISOTROPY_TOLERANCE = 1e-6

#: Agreement required between a shell's radial eigenvalue and the mean orbital energy of the
#: MOs that span it, in Eh. They are the same number computed two ways — through the
#: m-averaged Fock block and through the SCF's own diagonalization — so a deviation is the
#: m-averaging error, not a physical quantity. Measured: 1e-16 to 1e-14.
SHELL_ENERGY_TOLERANCE = 1e-8

#: Electrons. How far a shell's projected occupation may sit from what the configuration says
#: before :func:`extract_shells` refuses.
#:
#: ⚠ **This is a check that the solution has the configuration's electrons in its shells, and
#: deliberately not a second sphericity check** — a solution that settled into a neighbouring
#: configuration is off by a **whole electron**, which is three orders above this bound, while
#: sphericity separates by nine orders on the anisotropy above. Measured: exact to 1e-15 on a
#: converged average of configuration, and 1.3e-3 on a symmetry-broken ROHF oxygen, which this
#: bound therefore also refuses.
SHELL_OCCUPATION_TOLERANCE = 1e-3

_LABEL = re.compile(r"^(\d+)([{}])$".format(SHELL_LETTERS))


def parse_shell_label(text: str) -> Tuple[int, int]:
    """``"4f" -> (4, 3)``. A shell **name**, which is not a configuration term (no occupation).

    Deliberately a separate, smaller grammar from
    :func:`kuiva.amf.configuration.parse_shell_terms`: ``"4f"`` is not a shell occupation and
    reading it as one would make ``"4f"`` and ``"4f0"`` the same thing, which is the
    difference between naming a shell and stating that it is empty.
    """
    m = _LABEL.match(str(text).strip())
    if not m:
        raise ValueError(
            "cannot read {!r} as a shell label; expected a principal quantum number and an "
            "angular-momentum letter, e.g. '4f' or '6s'.".format(text))
    n, l = int(m.group(1)), SHELL_LETTERS.index(m.group(2))
    if n <= l:
        raise ValueError("there is no {}: the lowest shell of angular momentum {} is {}."
                         .format(text, SHELL_LETTERS[l], shell_label(l + 1, l)))
    return n, l


@dataclass(frozen=True)
class ShellOrbitals:
    """One atomic shell of a converged average-of-configuration solution.

    Attributes
    ----------
    n, l : int
    energy : float
        The shell's orbital energy [Eh], the eigenvalue of the m-averaged Fock block.
    occupation : float
        Electrons in the shell, **projected out of the converged density** rather than read
        off an MO list, so it does not depend on which basis of a degenerate manifold the
        eigensolver happened to return.
    coefficients : ndarray (nao, 2l+1)
        The shell's orbitals in the AO basis, ⚠ **columns ordered by ascending ``m``, from
        ``-l`` to ``+l``** — this convention, not the integral library's AO order (where a p
        shell runs ``px, py, pz`` = ``m = +1, -1, 0``). Anything that contracts these against
        angular coefficients must use the same order.
    radial : ndarray (n_radial,)
        The shell's coefficients over the radial (contracted) AO functions of this ``l``, in
        AO-shell order. The same numbers as a column of :attr:`coefficients`, without the
        embedding — this is the object a radial integral is defined by.
    m_values : tuple of int
        ``(-l, ..., +l)``, restating the column order of :attr:`coefficients` in the data.
    energy_spread : float
        ``|energy - mean orbital energy of the MOs spanning this shell|`` [Eh].
    degenerate_block : int
        Size of the degenerate group the shell's MOs form. Equal to ``2l+1`` in a clean
        solution; a different value means the SCF spectrum does not have this shell as one
        whole degenerate manifold.
    """

    n: int
    l: int
    energy: float
    occupation: float
    coefficients: np.ndarray
    radial: np.ndarray
    m_values: Tuple[int, ...]
    energy_spread: float
    degenerate_block: int

    @property
    def label(self) -> str:
        return shell_label(self.n, self.l)

    @property
    def size(self) -> int:
        """Orbitals in the shell, ``2l+1``."""
        return 2 * self.l + 1

    def __repr__(self) -> str:
        return "ShellOrbitals({}, eps = {:.6f} Eh, {:.4f} electrons)".format(
            self.label, self.energy, self.occupation)


@dataclass(frozen=True)
class AtomicShells:
    """The shells extracted from one converged solution, plus what they were checked against.

    ``anisotropy`` is the largest relative deviation of any ``l`` channel's Fock blocks across
    its ``m`` channels — the sphericity of the solution measured on the operator the radial
    functions come from. It is the diagnostic that says whether "shell" means anything here at
    all, and it is reported with every set of parameters derived from these orbitals.
    """

    shells: Tuple[ShellOrbitals, ...]
    anisotropy: float
    configuration: ShellConfiguration

    def __len__(self) -> int:
        return len(self.shells)

    def __iter__(self):
        return iter(self.shells)

    def __getitem__(self, key) -> ShellOrbitals:
        """By label (``shells["4f"]``), by ``(n, l)``, or by position."""
        if isinstance(key, int):
            return self.shells[key]
        n, l = parse_shell_label(key) if isinstance(key, str) else (int(key[0]), int(key[1]))
        for shell in self.shells:
            if (shell.n, shell.l) == (n, l):
                return shell
        raise KeyError("{} was not extracted; this set holds {}".format(
            shell_label(n, l), ", ".join(self.labels) or "nothing"))

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(shell.label for shell in self.shells)

    def report(self, logger=None) -> None:
        """The shell table, through the output grammar."""
        logger = logger or log
        table = out.Table(logger, [out.Column("shell", "{:s}", 6),
                                   out.Column("l", "{:d}", 3),
                                   out.Column("orbitals", "{:d}", 9),
                                   out.Column("electrons", "{:.6f}", 10),
                                   out.Column("eps [Eh]", out.E_FMT, 21),
                                   out.col_sci("spread [Eh]")])
        table.start("atomic shells, average of configuration")
        for shell in self.shells:
            table.row(shell.label, shell.l, shell.size, shell.occupation, shell.energy,
                      shell.energy_spread)
        table.end("Fock anisotropy across m channels: {:.2e}".format(self.anisotropy))


def _channel_blocks(layout, l: int) -> "List[Tuple[int, np.ndarray]]":
    """``[(m, ao indices of that (l, m), in radial order), ...]`` for angular momentum ``l``.

    A thin view of :func:`kuiva.amf.configuration.angular_channel_groups`, which is the one
    implementation of this grouping in the project — the same one the
    average-of-configuration SCF projects its Fock with, so the channels a shell is read out
    of are by construction the channels the solution was constrained to be symmetric over.
    Columns are in **ascending m**, which is not the integral library's within-shell order (a
    p shell is stored ``px, py, pz`` = ``m = +1, -1, 0``); the reordering lives there.
    """
    groups = angular_channel_groups(layout.ao_l, layout.ao_m, layout.ao_shell)
    if l not in groups:
        raise RuntimeError(
            "this basis has no functions of angular momentum {} ({}), so no shell of that "
            "channel can be extracted".format(l, SHELL_LETTERS[l]))
    return [(m, np.ascontiguousarray(column))
            for m, column in zip(range(-l, l + 1), groups[l].T)]


def extract_shells(data, configuration, *, shells: Optional[Sequence[str]] = None,
                   group_rtol: float = DEFAULT_GROUP_RTOL,
                   anisotropy_tol: float = SHELL_ANISOTROPY_TOLERANCE,
                   occupation_tol: float = SHELL_OCCUPATION_TOLERANCE) -> AtomicShells:
    """Radial functions and ``m``-aligned orbitals of the shells of an atomic solution.

    ``data`` is a converged **average-of-configuration** :class:`ScalarX2CData`
    (:func:`kuiva.interface.pyscf_bridge.run_scalar_aoc`) and ``configuration`` the
    :class:`ShellConfiguration` it was run with. ``shells`` names the shells to extract
    (``("4f", "5d", "6s")``); the default is every occupied shell.

    ⚠ **The radial function is read off the Fock operator, never off the orbitals**, and that
    is the whole design. An SCF returns each degenerate shell in an **arbitrary rotation** of
    its ``2l+1`` orbitals — a perfectly valid basis of the manifold that mixes ``m`` values
    with no way to tell from the coefficients — so extracting a radial function from an MO
    means first undoing a rotation nothing recorded. The Fock operator has no such freedom:
    on a spherical atom in real spherical harmonics it is **block diagonal in ``(l, m)`` with
    m-independent blocks**, by symmetry. So the blocks are averaged over the ``m`` channels of
    each ``l``, the small generalized eigenproblem ``F v = eps S v`` is solved once per
    channel, and the ``2l+1`` orbitals of a shell are then *built* from one radial function —
    pure ``m`` by construction, aligned with nothing to align.

    The Fock operator itself is reconstructed as ``F = S C diag(eps) C^T S``, which is exactly
    the operator the SCF last diagonalized (it inverts ``C^T F C = diag(eps)`` using
    ``C^T S C = 1``), so nothing has to be stored or recomputed and the DF and conventional
    routes are treated identically.

    Three checks run, and their asymmetry is deliberate:

    * the **m-channel deviation** of the Fock blocks — the sphericity of the solution, on the
      operator rather than on the density — warns above ``anisotropy_tol``;
    * each shell's radial eigenvalue against the mean orbital energy of the MOs spanning it,
      and whether those MOs form **one whole degenerate group** (:mod:`kuiva.util.degeneracy`,
      the project's group-completeness rule in its atomic instance) — both warn;
    * the shell's **projected occupation** against the configuration, which **raises**,
      because a mismatch there means the solution being analysed is not the one the
      configuration describes — the classic silent failure of an atomic average, where an SCF
      settles into a neighbouring configuration and every number after it is for another
      state.
    """
    configuration = ShellConfiguration.coerce(configuration)
    if getattr(data, "unrestricted", False):
        raise ValueError(
            "shell extraction needs a spin-restricted solution; this one has two MO sets, "
            "which have no common radial function to extract")
    layout = getattr(data, "ao_layout", None)
    if layout is None:
        raise ValueError(
            "shell extraction needs the AO layout (which atom, shell and m each AO belongs "
            "to) and this container carries none")
    if layout.natm != 1:
        raise ValueError(
            "shell extraction is defined for a single atom; this solution has {} of them, "
            "and 'the 4f shell' of a molecule is not a well-defined object"
            .format(layout.natm))

    s = np.asarray(data.s_ao, dtype=float)
    c = np.asarray(data.mo_coeff, dtype=float)
    e = np.asarray(data.mo_energy, dtype=float)
    occ = np.asarray(data.mo_occ, dtype=float)
    if c.shape[1] < c.shape[0]:
        log.warning("the solution carries %d orbitals in a %d-function basis; the Fock "
                    "operator reconstructed from them is blind to the %d directions the SCF "
                    "dropped, so a radial function here is one of the retained space.",
                    c.shape[1], c.shape[0], c.shape[0] - c.shape[1])
    sc = s @ c
    fock = (sc * e) @ sc.T

    requested = ([parse_shell_label(name) for name in shells] if shells is not None
                 else [(n, l) for n, l, _ in configuration.shells])
    wanted_l = sorted({l for _, l in requested})

    # One generalized eigenproblem per l channel: the m-averaged radial block.
    import scipy.linalg

    radial: Dict[int, Tuple[np.ndarray, np.ndarray, List[Tuple[int, np.ndarray]]]] = {}
    anisotropy = 0.0
    for l in wanted_l:
        blocks = _channel_blocks(layout, l)
        f_blocks = np.stack([fock[np.ix_(index, index)] for _, index in blocks])
        s_blocks = np.stack([s[np.ix_(index, index)] for _, index in blocks])
        f_mean, s_mean = f_blocks.mean(axis=0), s_blocks.mean(axis=0)
        scale = max(float(np.max(np.abs(f_mean))), 1e-300)
        deviation = float(np.max(np.abs(f_blocks - f_mean))) / scale
        anisotropy = max(anisotropy, deviation)
        # scipy's symmetric-definite driver: the blocks are tiny (one contraction count
        # square), so this is free, and it is the same generalized problem the SCF solves.
        eps, vectors = scipy.linalg.eigh(f_mean, s_mean)
        radial[l] = (eps, vectors, blocks)

    if anisotropy > anisotropy_tol:
        log.warning("the Fock operator varies by %.2e across the m channels of one angular "
                    "momentum, above the %.0e a spherical solution gives. Its blocks are "
                    "m-independent by symmetry, so the shells extracted here are averages "
                    "over orbitals that are not degenerate.", anisotropy, anisotropy_tol)

    extracted = []
    for n, l in requested:
        eps, vectors, blocks = radial[l]
        k = n - l - 1                      # the k-th radial function of the channel is n = l+1+k
        if not 0 <= k < eps.size:
            raise ValueError(
                "{} is the {} radial function of the {} channel and this basis offers only "
                "{}".format(shell_label(n, l), k + 1, SHELL_LETTERS[l], eps.size))
        v = vectors[:, k]
        coefficients = np.zeros((s.shape[0], 2 * l + 1))
        for column, (_, index) in enumerate(blocks):
            coefficients[index, column] = v

        # Weight of every MO on the shell's subspace: 1 for an MO of this shell, 0 otherwise,
        # summing to 2l+1. It replaces an MO *assignment* — no orbital has to be recognized
        # as belonging to a shell, and the answer is invariant to how the eigensolver
        # oriented a degenerate manifold.
        weights = np.sum((coefficients.T @ sc) ** 2, axis=0)
        occupation = float(weights @ occ)
        members = np.where(weights > 0.5)[0]
        energy = float(np.sum(weights * e) / max(np.sum(weights), 1e-300))
        spread = abs(energy - float(eps[k]))
        block = 0
        if members.size:
            # -e is descending where e ascends, which is the convention the grouping takes.
            start, stop = group_bounds(-e, int(members[0]), rtol=group_rtol)
            block = stop - start
            if block != 2 * l + 1 or members.size != 2 * l + 1:
                log.warning("the %s shell spans %d orbitals and sits in a degenerate group of "
                            "%d, where a shell of angular momentum %d is %d degenerate "
                            "orbitals. The SCF spectrum does not have this shell as one whole "
                            "manifold.", shell_label(n, l), members.size, block, l, 2 * l + 1)
        if spread > SHELL_ENERGY_TOLERANCE:
            log.warning("the %s radial eigenvalue is %.3e Eh from the mean orbital energy of "
                        "the orbitals that span it; the two are the same quantity computed "
                        "through the m-averaged Fock block and through the SCF's own "
                        "diagonalization.", shell_label(n, l), spread)
        wanted = configuration.occupation(n, l)
        if abs(occupation - wanted) > occupation_tol:
            raise RuntimeError(
                "the converged density puts {:.6f} electrons in {} where the configuration "
                "{} says {}. The solution being analysed is not the one this configuration "
                "describes — an atomic SCF that settles into a neighbouring configuration "
                "converges cleanly and gives entirely plausible numbers for the wrong state."
                .format(occupation, shell_label(n, l), configuration.canonical, wanted))

        extracted.append(ShellOrbitals(
            n=n, l=l, energy=float(eps[k]), occupation=occupation,
            coefficients=np.ascontiguousarray(coefficients),
            radial=np.ascontiguousarray(v), m_values=tuple(range(-l, l + 1)),
            energy_spread=spread, degenerate_block=block))

    return AtomicShells(shells=tuple(extracted), anisotropy=anisotropy,
                        configuration=configuration)


__all__ = ["AtomicShells", "SHELL_ANISOTROPY_TOLERANCE", "SHELL_ENERGY_TOLERANCE",
           "SHELL_OCCUPATION_TOLERANCE", "ShellConfiguration", "ShellOrbitals",
           "extract_shells", "parse_shell_label", "shell_capacity", "shell_label"]
