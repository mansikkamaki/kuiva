"""The version number and the machinery that has to agree with it.

`kuiva.__version__` is the single source of truth: `pyproject.toml` derives its version from
it, the run banner prints it, and every stored product — checkpoints, the property dump, the
pseudospin export — records it, where it is the only statement of which code produced a file
that outlives the session.

These tests are pure text and cost nothing, which is deliberate: the check has to run in the
default laptop suite or it will not run at the commit that breaks it.
"""
import re
from pathlib import Path

import kuiva

ROOT = Path(__file__).resolve().parent.parent

#: `MAJOR.MINOR.PATCH`, digits only. The versioning scheme defines no pre-release or build
#: suffix, so the test refuses one rather than let an unversioned "0.1.0-dev" reach a stored
#: file header.
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def test_version_is_a_plain_major_minor_patch():
    assert SEMVER.match(kuiva.__version__), (
        "{!r} is not MAJOR.MINOR.PATCH".format(kuiva.__version__))


def test_pyproject_takes_the_version_from_the_package():
    """No literal version in `pyproject.toml`: it reads the package attribute.

    Asserted as text rather than through `importlib.metadata`, which would report whatever is
    *installed* — and the working tree is what is under test.
    """
    text = (ROOT / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in text
    assert 'version = { attr = "kuiva.__version__" }' in text
    assert not re.search(r'(?m)^version\s*=\s*"', text), (
        "pyproject.toml declares a literal version; there is exactly one source of truth")


def test_the_banner_prints_the_current_version_without_being_told():
    """`out.banner()` defaults to the source of truth, so no driver can print a stale number."""
    import logging

    from kuiva.util import output as out

    log = logging.getLogger("kuiva.tests.version-banner")
    log.setLevel(logging.INFO)
    lines = []

    class Sink(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    handler = Sink()
    log.addHandler(handler)
    try:
        out.banner(log)
    finally:
        log.removeHandler(handler)

    assert any("version " + kuiva.__version__ in line for line in lines), lines


def test_stored_products_record_the_code_version():
    """A file that outlives the session says which code wrote it."""
    from kuiva.io import checkpoint
    from kuiva.props import dump, pseudospin

    assert checkpoint._kuiva_version() == kuiva.__version__
    assert dump._kuiva_version() == kuiva.__version__
    assert pseudospin._kuiva_version() == kuiva.__version__
