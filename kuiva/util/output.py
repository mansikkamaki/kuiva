"""The formatted-output grammar.

Why this module exists
----------------------
The INFO stream must "read like a conventional QC output" and that *every*
iterative procedure logs a consistent header/row/footer table. Consistency cannot be achieved
by asking each module to format its own lines nicely: it has to be a shared vocabulary. This
module is that vocabulary, and it is the **only** place where output layout is decided. A new
module writes output by calling :func:`section`, :func:`entry` and :class:`Table` — never by
composing its own rules, banners or column widths.

The grammar (fixed; change it here or nowhere)
----------------------------------------------
Five constructs, and nothing else::

    ================================================================================
     Integral transformation                                            t = 12.3 s     <- section()
    ================================================================================

     -- Cholesky decomposition of the AO ERIs -------------------------------------    <- subsection()

       AO basis functions                                    116                       <- entry()
       Cholesky vectors                                      612
       largest neglected diagonal                       8.31e-09  Eh

       iter            energy [Eh]      dE [Eh]      |grad|   wall [s]                 <- Table
       ----  ---------------------  -----------  ----------  ---------
          1     -1234.567890123456     1.23e-02    4.56e-03      12.34
        ...
       ----  ---------------------  -----------  ----------  ---------
       converged in 7 iterations                                                       <- Table.end()

     *** WARNING [orth.canonical] 2 near-linearly-dependent vectors dropped             <- log.warning

Rules that follow from it, and that callers must respect:

1. **INFO is the output.** The stream handler prints INFO records verbatim, so an INFO line
   *is* a line of the output file. Do not log unstructured prose at INFO.
2. **WARNING/ERROR interrupt the output** with a marked, module-attributed line — they are
   meant to be greppable (``grep '\\*\\*\\*'``) and to break the visual flow on purpose.
3. **DEBUG/TRACE are diagnostics**, not output: they keep a level+module prefix and may be
   as verbose as they like.
4. **ASCII only.** Log files are read over ssh, tailed, diffed and parsed by scripts on
   machines with unknown locales; a multi-byte character in a column header is a portability
   risk for no benefit. Unicode belongs in docstrings, not in the output stream. Use
   ``dE``, ``|grad|``, ``mu_B``, ``Eh``, ``cm^-1``.
5. **Units are always in the label or the column header**, never implied, and numbers are
   printed to the meaningful tolerance (energies 1e-8 Eh -> ``E_FMT``; moments
   1e-5 mu_B -> ``MOMENT_FMT``). Use the ``*_FMT`` constants rather than inventing formats,
   so a given physical quantity looks the same everywhere in the output.

There is deliberately no dependency on :mod:`kuiva.util.timing` here (timing imports output,
not the other way round), so the section clock is kept locally.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Optional, Sequence, Tuple

# --- Layout constants (the single source of truth for output geometry) -----------------

#: Total line width. 80 columns: the width every terminal, pager and printer agrees on.
WIDTH = 80
#: Indentation of everything below a section header.
INDENT = "   "
#: Width of the label column in :func:`entry`.
LABEL_W = 46
#: Width of the value column in :func:`entry`.
VALUE_W = 18

# --- Standard numeric formats (precision matched to the meaningful tolerance) ------

#: Total energies [Eh]. 12 decimals covers the 1e-8 Eh tolerance with margin, and stays
#: meaningful for heavy-element totals of order -1e4 Eh (16 significant digits).
E_FMT = "{:.12f}"
#: Energy differences / gradients / residuals, where only the order of magnitude matters.
SCI_FMT = "{:.3e}"
#: Magnetic moments and g values [mu_B] (1e-5 mu_B).
MOMENT_FMT = "{:.6f}"
#: Excitation energies [cm^-1].
CM_FMT = "{:.2f}"
#: Wall/CPU seconds.
TIME_FMT = "{:.2f}"

_T0 = time.perf_counter()


def elapsed() -> float:
    """Wall-clock seconds since this module was imported (i.e. since program start)."""
    return time.perf_counter() - _T0


# --- Constructs -------------------------------------------------------------------------


def banner(log: logging.Logger, version: Optional[str] = None, subtitle: str = "") -> None:
    """Print the program banner. Emitted once, by the top-level driver.

    ``version`` defaults to :data:`kuiva.__version__` — the single source of truth of
    by design — so an output file always states the code that produced it and no
    driver can print a version that has fallen behind. It is still accepted explicitly,
    for a caller reporting on a run other than this process.

    The banner is where a run states *how* it computes, in two lines that each come from
    the one module owning the state:

    * :func:`kuiva.util.native.banner_entry` — the kernel backend, printed only when it
      differs from the pure-NumPy default (compiled backend active, or NumPy forced
      through ``KUIVA_KERNELS``). An unmarked banner *is* a pure-NumPy run, which is what
      keeps every committed example reference valid on a build-less clone.
    * :func:`kuiva.util.threads.banner_entry` — the thread budget, where it came from,
      and the **measured** verdict on whether the loaded BLAS actually threads. Printed
      unconditionally, because unlike the backend there is no state here that can be
      inferred from silence, and because a budget nobody verifies is exactly how a run
      spends four CPU-hours to do one hour of work.
    """
    log.info("=" * WIDTH)
    log.info("{:^{w}}".format("K U I V A", w=WIDTH))
    log.info("{:^{w}}".format("relativistic multireference quantum chemistry", w=WIDTH))
    if subtitle:
        log.info("{:^{w}}".format(subtitle, w=WIDTH))
    if version is None:
        from .. import __version__ as version
    log.info("{:^{w}}".format("version " + version, w=WIDTH))
    from . import native
    from . import threads

    backend_line = native.banner_entry()
    if backend_line:
        log.info("{:^{w}}".format(backend_line, w=WIDTH))
    log.info("{:^{w}}".format(threads.banner_entry(), w=WIDTH))
    log.info("=" * WIDTH)


def section(log: logging.Logger, title: str, *, clock: bool = True) -> None:
    """A major section header: the top-level unit of the output.

    One per phase of the calculation (ingestion, orthogonalization, integral transformation,
    CASSCF, NEVPT2, properties). ``clock`` stamps the elapsed wall time, which is what makes
    a long output file navigable in time as well as in structure.
    """
    stamp = "t = " + TIME_FMT.format(elapsed()) + " s" if clock else ""
    head = " " + title
    pad = WIDTH - len(head) - len(stamp)
    log.info("")
    log.info("=" * WIDTH)
    log.info("%s%s%s", head, " " * max(1, pad), stamp)
    log.info("=" * WIDTH)


def subsection(log: logging.Logger, title: str) -> None:
    """A subsection rule inside a section (one step of a phase)."""
    head = " -- " + title + " "
    log.info("")
    log.info("%s%s", head, "-" * max(0, WIDTH - len(head)))


def entry(log: logging.Logger, label: str, value: Any, unit: str = "",
          note: str = "", fmt: Optional[str] = None,
          level: int = logging.INFO) -> None:
    """One ``label ... value [unit] (note)`` line — the workhorse of the output.

    ``fmt`` is a :meth:`str.format` spec applied to numeric values (use the module's
    ``*_FMT`` constants). Without it, floats fall back to ``{:.6g}``, which is fine for
    dimensionless diagnostics but *not* for physical quantities — pass the format.
    """
    text = _format_value(value, fmt)
    line = INDENT + label.ljust(LABEL_W) + text.rjust(VALUE_W)
    if unit:
        line += "  " + unit
    if note:
        line += "   (" + note + ")"
    log.log(level, "%s", line)


def entries(log: logging.Logger, pairs: Iterable[Sequence[Any]],
            level: int = logging.INFO) -> None:
    """Several :func:`entry` lines from ``(label, value[, unit[, note[, fmt]]])`` tuples."""
    for p in pairs:
        entry(log, p[0], p[1], *tuple(p[2:]), level=level)


def blank(log: logging.Logger, level: int = logging.INFO) -> None:
    """A single empty output line."""
    log.log(level, "")


def note(log: logging.Logger, text: str, level: int = logging.INFO) -> None:
    """A free-standing indented line (a short statement of fact, not prose)."""
    log.log(level, "%s%s", INDENT, text)


def _format_value(value: Any, fmt: Optional[str] = None) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):                      # before int: bool is an int
        return "yes" if value else "no"
    if isinstance(value, float):
        return (fmt or "{:.6g}").format(value)
    if isinstance(value, int):
        return (fmt or "{:d}").format(value)
    return str(value)


# --- Tables (every iterative procedure logs header/row/footer) ---------------------


class Column:
    """One table column: header text, a format spec, and a width.

    The width is ``max(len(header), width)``, so a column can never be narrower than its own
    header; a value wider than the column overflows rather than being truncated (silently
    losing digits in an output file is worse than a ragged line).
    """

    __slots__ = ("header", "fmt", "width", "align")

    def __init__(self, header: str, fmt: str = "{}", width: int = 0, align: str = ">"):
        self.header = header
        self.fmt = fmt
        self.width = max(len(header), width)
        self.align = align

    def render(self, value: Any) -> str:
        if value is None:
            text = ""
        else:
            try:
                text = self.fmt.format(value)
            except (ValueError, TypeError):
                text = str(value)
        return "{:{a}{w}}".format(text, a=self.align, w=self.width)

    def render_header(self) -> str:
        return "{:{a}{w}}".format(self.header, a=self.align, w=self.width)

    def rule(self) -> str:
        return "-" * self.width


#: Ready-made columns for the quantities that recur in every iterative procedure. Use these
#: rather than redefining equivalents, so an SCF table, a Davidson table, a DMRG sweep table
#: and a CASSCF macro-iteration table line up column-for-column in the output.
def col_iter(header: str = "iter") -> Column:
    return Column(header, "{:d}", 4)


def col_energy(header: str = "energy [Eh]") -> Column:
    return Column(header, E_FMT, 21)


def col_delta(header: str = "dE [Eh]") -> Column:
    return Column(header, SCI_FMT, 11)


def col_resid(header: str = "|grad|") -> Column:
    return Column(header, SCI_FMT, 10)


def col_time(header: str = "wall [s]") -> Column:
    return Column(header, TIME_FMT, 9)


def col_count(header: str, width: int = 8) -> Column:
    return Column(header, "{:d}", width)


def col_sci(header: str, width: int = 11) -> Column:
    return Column(header, SCI_FMT, width)


class Table:
    """A header/row/footer table logged through ``log``.

    ::

        tab = Table(log, [col_iter(), col_energy(), col_delta(), col_time()])
        tab.start()
        for it in ...:
            tab.row(it, e, de, dt)
        tab.end("converged in {} iterations".format(it))

    The table writes at ``level`` (INFO by default). Rows logged at DEBUG belong to
    micro-iterations; the macro-iteration table stays at INFO.
    """

    def __init__(self, log: logging.Logger, columns: Sequence[Column],
                 *, level: int = logging.INFO, gap: str = "  "):
        self.log = log
        self.columns = list(columns)
        self.level = level
        self.gap = gap
        self._rows = 0

    def _line(self, cells: Sequence[str]) -> str:
        return INDENT + self.gap.join(cells)

    def start(self, title: str = "") -> "Table":
        if title:
            note(self.log, title, level=self.level)
        self.log.log(self.level, "%s", self._line([c.render_header() for c in self.columns]))
        self.log.log(self.level, "%s", self._line([c.rule() for c in self.columns]))
        return self

    def row(self, *values: Any) -> None:
        if len(values) != len(self.columns):
            raise ValueError("table row has {} values for {} columns".format(
                len(values), len(self.columns)))
        self._rows += 1
        self.log.log(self.level, "%s",
                     self._line([c.render(v) for c, v in zip(self.columns, values)]))

    def end(self, footer: str = "") -> None:
        self.log.log(self.level, "%s", self._line([c.rule() for c in self.columns]))
        if footer:
            note(self.log, footer, level=self.level)

    @property
    def n_rows(self) -> int:
        return self._rows


# --- Diagnostic helpers (DEBUG/TRACE only) ---------------------------------------------


def matrix(log: logging.Logger, name: str, a, *, level: int = 5,
           max_dim: int = 12, fmt: str = "{:12.6f}") -> None:
    """Dump a small matrix at TRACE (level 5) for debugging.

    Never used at INFO: matrices do not belong in the output file (machine-readable
    matrices go to the property dump, not the log). Large arrays print shape and norms only.
    """
    if not log.isEnabledFor(level):
        return
    import numpy as np                       # local: keep the import off the INFO path

    a = np.asarray(a)
    log.log(level, "%s%s: shape=%s dtype=%s |A|_F=%.6e", INDENT, name, a.shape, a.dtype,
            float(np.linalg.norm(a)))
    if a.ndim != 2 or max(a.shape) > max_dim:
        return
    for i in range(a.shape[0]):
        cells = "".join((fmt.format(v.real) if not np.iscomplexobj(a) else
                         "{:11.5f}{:+.5f}i".format(v.real, v.imag)) for v in a[i])
        log.log(level, "%s  %s", INDENT, cells)


__all__ = [
    "WIDTH", "INDENT", "LABEL_W", "VALUE_W",
    "E_FMT", "SCI_FMT", "MOMENT_FMT", "CM_FMT", "TIME_FMT",
    "elapsed", "banner", "section", "subsection", "entry", "entries", "blank", "note",
    "Column", "Table", "matrix",
    "col_iter", "col_energy", "col_delta", "col_resid", "col_time", "col_count", "col_sci",
]
