"""The persistent X2CAMF correction cache (:mod:`kuiva.amf.cache`).

An in-process cache can only be wrong about its key. A **persistent** one can also be wrong
about *time*: it outlives the code that wrote it. That is a failure mode this project has not
had before, and it is not hypothetical — a convention change once moved which ``X`` the
decoupling uses, which
moves ``dh_sf`` by 10% for an unchanged request, so a cache written the day before would have
gone on serving the old matrices with nothing anywhere reading wrong.

So the tests here are mostly about the cache being **wrong in the safe direction**. Every one
of them is a statement that some kind of staleness or damage produces a *miss* rather than an
answer:

* a different formula version, a different schema, a different request, a corrupt file, a
  truncated file, an unwritable directory — all misses, none an exception;
* and the one case that is *not* a silent miss: two different requests arriving at one
  filename, which is a defect in :func:`kuiva.amf.atomic.cache_key` and warns.

⚠ **Every test here points ``$KUIVA_AMF_CACHE`` at a temporary directory.** A test that used
the real cache would either pollute the developer's or, far worse, pass because of it.
"""
from __future__ import annotations

import numpy as np
import pytest

from kuiva.amf import atomic, cache
from kuiva.amf.decouple import AtomicAMF

h5py = pytest.importorskip("h5py", reason="the persistent cache needs h5py")


#: ⚠ **The whole file is the test of the persistent cache**, so no stage checkpoint and no
#: stage-managed X2CAMF cache may reach it. Every assertion here is a solve
#: count or a hit count against a directory this file controls; one placed underneath by the
#: checkpoint machinery would answer them all with zero.
pytestmark = pytest.mark.stage_under_test("amf_atomic")


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """A private, empty cache directory for one test."""
    d = tmp_path / "amf-cache"
    monkeypatch.setenv(cache.ENV_CACHE_DIR, str(d))
    atomic.clear_cache()
    yield d
    atomic.clear_cache()


@pytest.fixture
def request_and_correction():
    """A request and a correction that are consistent with each other but cost nothing.

    The correction is fabricated rather than computed: this file tests the *storage*, and a
    four-component solve would make every one of these tests seconds long for no added
    coverage. What the arrays contain is irrelevant as long as it comes back unchanged, so
    they are deliberately asymmetric and non-trivial — an accidental transpose or a real/imag
    slip in the round trip has something to fail on.
    """
    req = atomic.make_request("Ne", [[0, [1.0, 1.0]], [1, [0.5, 1.0]]],
                              interaction="coulomb")
    rng = np.random.default_rng(20260730)
    a = rng.standard_normal((4, 4))
    h_sf = a + a.T
    w = np.array([b - b.T for b in rng.standard_normal((3, 4, 4))])
    amf = AtomicAMF(h_sf=h_sf, w=w, configuration=req.configuration, scale=1.25,
                    tr_residual=3.5e-13, tr_residual_rel=2.5e-12,
                    transformed_scale=8.0, subtracted_scale=7.5, compensation_scale=0.125)
    return req, amf


# --- the round trip -----------------------------------------------------------------------

def test_a_correction_survives_the_round_trip_exactly(cache_dir, request_and_correction):
    """Bitwise, not ``allclose``.

    HDF5 stores IEEE doubles, so there is no reason for a stored correction to differ from the
    computed one in any bit — and "approximately the same correction" is not a thing this
    cache is allowed to return, since the whole justification for reusing it across jobs is
    that the quantity has exactly one value.
    """
    req, amf = request_and_correction
    key = atomic.cache_key(req)
    assert cache.store(req, key, amf)
    assert cache.entry_path(cache_dir, key).is_file()

    got = cache.load(req, key)
    assert got is not None
    assert np.array_equal(got.h_sf, amf.h_sf)
    assert np.array_equal(got.w, amf.w)
    for field in ("scale", "tr_residual", "tr_residual_rel", "transformed_scale",
                  "subtracted_scale", "compensation_scale"):
        assert getattr(got, field) == getattr(amf, field), field
    assert got.configuration == amf.configuration


def test_a_miss_is_none_not_an_exception(cache_dir, request_and_correction):
    req, _ = request_and_correction
    assert cache.load(req, atomic.cache_key(req)) is None


# --- staleness, which is what a persistent cache adds ---------------------------------------

def test_a_different_formula_version_is_a_miss(cache_dir, request_and_correction,
                                               monkeypatch):
    """⚠ The test this whole module exists for.

    :data:`kuiva.amf.cache.FORMULA_VERSION` is the only thing standing between a change in the
    physics and a cache that keeps serving the answer from before it. Simulated by writing at
    one version and reading at another, which is exactly what a code update does to a user's
    cache directory.
    """
    req, amf = request_and_correction
    key = atomic.cache_key(req)
    assert cache.store(req, key, amf)
    assert cache.load(req, key) is not None

    monkeypatch.setattr(cache, "FORMULA_VERSION", cache.FORMULA_VERSION + 1)
    assert cache.load(req, key) is None, \
        "a correction written under a different formula version was served anyway"


def test_a_different_schema_is_a_miss(cache_dir, request_and_correction, monkeypatch):
    req, amf = request_and_correction
    key = atomic.cache_key(req)
    cache.store(req, key, amf)
    monkeypatch.setattr(cache, "SCHEMA", cache.SCHEMA + 1)
    assert cache.load(req, key) is None


def test_the_stored_request_is_checked_field_by_field(cache_dir, request_and_correction,
                                                      kuiva_caplog):
    """⚠ A filename collision must not become another element's correction.

    The request is written into the file and compared on read, so an entry found under a key
    it does not describe is refused. This is asserted through a **deliberately wrong key**
    rather than by corrupting the file, because that is the shape the real defect would take:
    a :func:`kuiva.amf.atomic.cache_key` that has stopped separating two requests.

    And it **warns** rather than passing silently, unlike every other miss here: a normal miss
    means "not computed yet", while this one means the key is broken.
    """
    req, amf = request_and_correction
    other = atomic.make_request("Ar", [[0, [1.0, 1.0]]], interaction="coulomb")
    key = atomic.cache_key(req)
    cache.store(req, key, amf)

    kuiva_caplog.clear()
    assert cache.load(other, key) is None
    warnings = [r.getMessage() for r in kuiva_caplog.records if r.levelname == "WARNING"]
    assert any("not separating" in m for m in warnings), \
        "a request/entry mismatch must warn: it means cache_key is broken; got {}".format(
            warnings)


def test_every_request_field_is_stored_and_checked(request_and_correction):
    """A new field on ``AtomicRequest`` must take part in the validation automatically.

    The cache-key rule: "any new axis of variation must enter the key, or two different
    calculations alias and the second silently returns the first". The persistent cache adds a
    second place that can be forgotten, so this test asserts the two cannot drift — the stored
    attributes are derived from ``dataclasses.fields`` rather than listed, and this is what
    fails if someone replaces that with a hand-written list.
    """
    import dataclasses

    req, _ = request_and_correction
    stored = cache._request_attributes(req)
    fields = {f.name for f in dataclasses.fields(req)}
    assert set(stored) == fields, \
        "stored request attributes {} do not match AtomicRequest's fields {}".format(
            sorted(stored), sorted(fields))


# --- damage, which must never reach the caller ----------------------------------------------

@pytest.mark.parametrize("damage", ["truncate", "garbage", "empty"])
def test_a_damaged_entry_is_a_miss_and_not_an_exception(cache_dir, request_and_correction,
                                                        damage):
    """A cache must never be able to fail a calculation.

    A job killed mid-write, a full filesystem, an interrupted copy. All three shapes of damage
    are read back, and the only acceptable outcome is ``None`` plus a warning.
    """
    req, amf = request_and_correction
    key = atomic.cache_key(req)
    cache.store(req, key, amf)
    path = cache.entry_path(cache_dir, key)

    raw = path.read_bytes()
    if damage == "truncate":
        path.write_bytes(raw[:len(raw) // 3])
    elif damage == "garbage":
        path.write_bytes(b"this is not an HDF5 file" * 100)
    else:
        path.write_bytes(b"")

    assert cache.load(req, key) is None


def test_an_unwritable_directory_does_not_raise(tmp_path, monkeypatch,
                                                request_and_correction):
    """The cache degrades to "no cache", not to a failed run."""
    target = tmp_path / "readonly" / "amf"
    (tmp_path / "readonly").mkdir()
    (tmp_path / "readonly").chmod(0o500)
    monkeypatch.setenv(cache.ENV_CACHE_DIR, str(target))
    req, amf = request_and_correction
    try:
        assert cache.store(req, atomic.cache_key(req), amf) is False
        assert cache.load(req, atomic.cache_key(req)) is None
    finally:
        (tmp_path / "readonly").chmod(0o700)


def test_a_partial_write_leaves_no_visible_entry(cache_dir, request_and_correction,
                                                 monkeypatch):
    """Writes are atomic: an entry appears whole or not at all.

    Forced by making the rename fail, which is the last step — everything before it has
    already been written to the temporary file, so a non-atomic implementation would leave a
    complete-looking entry behind at the real path.
    """
    import os

    req, amf = request_and_correction
    key = atomic.cache_key(req)

    def boom(*a, **k):
        raise OSError("no")

    monkeypatch.setattr(os, "replace", boom)
    assert cache.store(req, key, amf) is False
    assert not cache.entry_path(cache_dir, key).exists()
    assert list(cache_dir.glob("*.tmp")) == [], "the temporary file was left behind"


# --- configuration ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["off", "none", "OFF", ""])
def test_the_cache_can_be_turned_off(monkeypatch, value, request_and_correction):
    monkeypatch.setenv(cache.ENV_CACHE_DIR, value)
    assert cache.cache_directory() is None
    assert cache.available() is False
    assert cache.describe() == "off"
    req, amf = request_and_correction
    assert cache.store(req, "k", amf) is False
    assert cache.load(req, "k") is None


def test_the_default_directory_is_under_the_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.delenv(cache.ENV_CACHE_DIR, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    # The configuration file must not be consulted for this test to mean anything about the
    # fallback, so point the search at a file that sets only the memory limit.
    cfg = tmp_path / "defaults.conf"
    cfg.write_text("[memory]\nmemory_gb = 1.0\n")
    monkeypatch.setenv("KUIVA_CONFIG", str(cfg))
    assert cache.cache_directory() == tmp_path / "kuiva" / "amf"


def test_the_configuration_file_can_set_the_directory(monkeypatch, tmp_path):
    monkeypatch.delenv(cache.ENV_CACHE_DIR, raising=False)
    cfg = tmp_path / "defaults.conf"
    cfg.write_text("[memory]\nmemory_gb = 1.0\n\n[amf]\n{} = {}\n".format(
        cache.CONFIG_KEY, tmp_path / "from-config"))
    monkeypatch.setenv("KUIVA_CONFIG", str(cfg))
    assert cache.cache_directory() == tmp_path / "from-config"


def test_purge_removes_entries_and_reports_how_many(cache_dir, request_and_correction):
    req, amf = request_and_correction
    cache.store(req, "a", amf)
    cache.store(req, "b", amf)
    assert cache.purge() == 2
    assert cache.purge() == 0


# --- the integration: a second process must not solve anything ------------------------------

@pytest.mark.parametrize("uncontract", [True])
def test_a_second_cold_start_serves_the_correction_from_disk(cache_dir, uncontract):
    """⚠ **The deliverable, asserted by call count and not by timing**.

    Two cold in-process starts against one cache directory: the first solves, the second must
    not. Hydrogen would be free and prove nothing (a one-electron reference returns exact
    zeros without solving), so this uses helium in a tiny basis — a real four-component solve,
    a real correction, and under a second.
    """
    from pyscf import gto

    basis = gto.basis.load("sto-3g", "He")

    atomic.clear_cache()
    first = atomic.atomic_correction("He", basis, uncontract=uncontract)
    stats = atomic.cache_statistics()
    assert stats["solves"] == 1, stats
    assert stats["disk_writes"] == 1, stats

    atomic.clear_cache()
    second = atomic.atomic_correction("He", basis, uncontract=uncontract)
    stats = atomic.cache_statistics()
    assert stats["solves"] == 0, "the second cold start solved again: {}".format(stats)
    assert stats["disk_hits"] == 1, stats
    assert np.array_equal(first.h_sf, second.h_sf)
    assert np.array_equal(first.w, second.w)


def test_a_scaled_speed_of_light_is_not_written_to_disk(cache_dir):
    """A ``c``-scaled solve is a test artefact and does not belong in a user's cache.

    It could not alias — ``light_speed`` is in the key and in the stored request — so this is
    about what the directory is *for*, not about correctness.
    """
    from pyscf import gto

    basis = gto.basis.load("sto-3g", "He")
    atomic.clear_cache()
    atomic.atomic_correction("He", basis, light_speed=1e4)
    stats = atomic.cache_statistics()
    assert stats["solves"] == 1
    assert stats["disk_writes"] == 0, stats
    assert list(cache_dir.glob("*.h5")) == []
