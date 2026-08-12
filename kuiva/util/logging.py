"""Project logging wrapper.

Policy enforced here:
- No bare ``print`` in the library; everything goes through ``logging``.
- One logger per module, named by module (e.g. ``kuiva.basis.registry``), so verbosity
  is controllable per-subsystem.
- Levels mapped to scientific meaning: ERROR (cannot proceed), WARNING (proceeds but the
  user should know), INFO (one line per macro-iteration; default), DEBUG (per-micro-
  iteration detail), TRACE (tensor shapes / contraction paths; off unless requested).

TRACE is a custom level below DEBUG. Physical quantities should be logged with units and a
fixed precision matching the meaningful tolerance (energies 1e-8 Eh, moments 1e-5 μ_B).

Formatting policy (decided with the output grammar; see :mod:`kuiva.util.output`)
---------------------------------------------------------------------------------
The INFO stream **is** the calculation's output file, so INFO records are printed *verbatim*:
a timestamp/level/module prefix on every line would make the fixed-width tables and
label/value blocks of the output grammar unreadable. The other levels are diagnostics and keep a prefix:

* ``INFO``            — printed as-is (the output; produced through ``kuiva.util.output``).
* ``WARNING``/``ERROR`` — ``*** WARNING [subsystem] message``: deliberately breaks the visual
  flow of the output and is greppable (``grep '\\*\\*\\*' out``), and names the subsystem so
  the origin of a warning buried in a long run is unambiguous.
* ``DEBUG``/``TRACE``  — ``[DEBUG   subsystem] message``, with a timestamp only for DEBUG-level
  work where "when" matters.

Consequence for callers: **never log unstructured prose at INFO** — it lands in the output
file with no marker. Diagnostics go to DEBUG; things the user must notice go to WARNING.
"""
import logging as _logging

# --- Custom TRACE level (below DEBUG=10) -----------------------------------------------
TRACE = 5
_logging.addLevelName(TRACE, "TRACE")


def _trace(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)


# Attach .trace() to the Logger class once.
if not hasattr(_logging.Logger, "trace"):
    _logging.Logger.trace = _trace  # type: ignore[attr-defined]

_ROOT_NAME = "kuiva"
_CONFIGURED = False


def _subsystem(name: str) -> str:
    """``kuiva.dmrg.sweep`` -> ``dmrg.sweep`` (the ``kuiva.`` prefix is noise in our own log)."""
    return name[len(_ROOT_NAME) + 1:] if name.startswith(_ROOT_NAME + ".") else name


class KuivaFormatter(_logging.Formatter):
    """Level-dependent formatting, per the policy in this module's docstring."""

    def format(self, record: _logging.LogRecord) -> str:
        msg = record.getMessage()
        level = record.levelno
        if level == _logging.INFO:
            return msg                                    # the output stream, verbatim
        if level >= _logging.WARNING:
            tag = "ERROR" if level >= _logging.ERROR else "WARNING"
            head = " *** {} [{}] ".format(tag, _subsystem(record.name))
            out = [head + msg.splitlines()[0]] if msg else [head]
            out += ["     " + line for line in msg.splitlines()[1:]]
            if record.exc_info:
                out.append(self.formatException(record.exc_info))
            return "\n".join(out)
        stamp = _logging.Formatter("%(asctime)s", datefmt="%H:%M:%S").format(record)
        return "[{} {:<5} {}] {}".format(stamp, record.levelname, _subsystem(record.name), msg)


def _ensure_root_handler() -> None:
    """Attach a single stream handler to the ``kuiva`` root logger (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = _logging.getLogger(_ROOT_NAME)
    if not root.handlers:
        handler = _logging.StreamHandler()
        handler.setFormatter(KuivaFormatter())
        root.addHandler(handler)
    root.setLevel(_logging.INFO) # INFO is the default level
    root.propagate = False
    _CONFIGURED = True


def add_file_handler(path, level=None) -> _logging.Handler:
    """Also write the log (output included) to ``path``. Returns the handler.

    The formatted property matrices are a *separate* file written by ``props/dump.py``;
    this is the human-readable output stream only, and the two must never be mixed.
    """
    _ensure_root_handler()
    handler = _logging.FileHandler(str(path), mode="w")
    handler.setFormatter(KuivaFormatter())
    if level is not None:
        handler.setLevel(level)
    _logging.getLogger(_ROOT_NAME).addHandler(handler)
    return handler


def get_logger(name: str) -> _logging.Logger:
    """Return the module logger for ``name`` (typically ``__name__``).

    Names are namespaced under ``kuiva`` so per-subsystem verbosity control works.
    """
    _ensure_root_handler()
    if not name.startswith(_ROOT_NAME):
        # Map bare module names into the kuiva namespace.
        name = f"{_ROOT_NAME}.{name}"
    return _logging.getLogger(name)


def set_verbosity(level) -> None:
    """Set the verbosity of the whole ``kuiva`` logger tree.

    ``level`` may be a stdlib level int, the custom ``TRACE``, or a name
    ("ERROR"/"WARNING"/"INFO"/"DEBUG"/"TRACE").
    """
    _ensure_root_handler()
    if isinstance(level, str):
        level = TRACE if level.upper() == "TRACE" else _logging.getLevelName(level.upper())
    _logging.getLogger(_ROOT_NAME).setLevel(level)


__all__ = ["get_logger", "set_verbosity", "add_file_handler", "KuivaFormatter", "TRACE"]
