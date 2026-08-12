"""The external ``x2camf`` plugin, as a second implementation of the same correction.

This module is the whole of Kuiva's contact with
https://github.com/Warlocat/x2camf — the reference implementation of X2CAMF by the group that
published the method (J. Liu, L. Cheng, *J. Chem. Phys.* **148**, 144108 (2018)), built by
``scripts/bootstrap/80_x2camf.sh``.

Why a second implementation is worth a dependency
-------------------------------------------------
``tests/reference/x2camf_dirac.json`` already says, from outside, that Kuiva's
*answer* is right: the corrected one-particle spectrum reproduces DIRAC's four-component atoms
to 0.003-0.005% on a j-splitting. What it cannot say is **which term** a disagreement would
live in, because a four-component calculation performs no picture change and so has no term
that corresponds to any of Kuiva's. A same-method implementation does, and that is the only
thing this dependency buys — see :mod:`tests.generate.x2camf_plugin` for what it bought.

⚠ Import-gated, never default, not a runtime dependency. :func:`available` is the only
honest way to ask whether it is here; the committed reference file means the comparison still
runs without it.

Three things about the plugin that are not obvious and cost measurements to establish
--------------------------------------------------------------------------------------

1. ⚠ **Its default entry point returns only the spin-dependent half of the correction.**
   ``x2camf.amfi(...)`` is built from the *spin-dependent* parts of the two-electron integrals
   (``dhf_sph.cpp::get_amfi_unc``, which contracts ``h2eSSLL_SD`` / ``h2eSSSS_SD``), so the
   two-electron **scalar** picture change — the larger half of Kuiva's correction, and the
   part Breit-Pauli AMFI and SNSO do not describe at all — is simply absent
   from it. Measured for Ne in the primitive ``x2c-SVPall-2c`` basis: ``max |dh_sf|``
   **1.01e-05** against Kuiva's **2.72e-02**. That is not a disagreement, it is a different
   quantity, and taking it for the whole correction is how one would silently lose the scalar
   term. The counterpart of Kuiva's ``dG`` is the ``pcc=True`` variant
   (``dhf_sph_pcc.cpp::x2c2ePCC``), and that is what :data:`VARIANTS` defaults to.

2. **It decouples with the X of the converged four-component Fock — and Kuiva now does too,
   because of this comparison.** The plugin builds ``X`` from the SCF coefficients and
   compensates the resulting change in the one-electron Hamiltonian by adding
   ``h1e(X_2e) - h1e(X_1e)`` to the correction (``dhf_sph_pcc.cpp::x2c2ePCC``). Through Stages
   2-7 Kuiva derived ``X`` and ``R`` from the **one-electron** blocks instead, on the argument
   that "the same X as the one-electron Hamiltonian the molecule carries" is then structural.

   ⚠ **That was the entire difference between the two implementations, and it is worth keeping
   the number that closed it.** The difference was localized rather than argued about:
   rebuilding Kuiva's correction the plugin's way reproduced its matrix to **2.0e-10 Eh on
   3.0e-02 Eh**, while the two conventions differed by 5-10% in ``dh_sf`` in the primitive
   basis, by 0.011% in ``dw`` (so no splitting moved by more than 0.42 cm^-1), and by **5-35x
   on the energy functional** against four-component Dirac-Coulomb — the one in-house check
   that discriminates the subtraction. Kuiva adopted the plugin's convention on that
   evidence (:func:`kuiva.amf.decouple.x2c_decoupling`), so the two now
   agree to **1e-08 relative on both parts** and this module compares two implementations of
   one convention rather than two conventions.

3. ⚠ **It always uses the neutral atom.** Its interface takes an atomic number, a shell list
   and exponents, and nothing else — there is no way to ask it for an ion or for a chosen
   reference configuration, which is a capability Kuiva has and it does not. So a like-for-like comparison must be run against Kuiva's **neutral** reference,
   not against the f-block M(3+) default. :func:`plugin_correction` refuses any other
   configuration rather than quietly comparing two different states. Open shells are handled
   by the plugin's own average-of-configuration solver (``aoc``), which this module turns on
   exactly when the neutral atom has one.

Molecules — and why that is worth having
--------------------------------------------------------
``x2camf.amfi`` does its own **molecular assembly**: one atomic block per unique element,
placed with ``construct_molecular_matrix`` over ``xmol.aoslice_2c_by_atom()``. That makes it a
second, independent implementation of exactly the block placement
:func:`kuiva.amf.correction.amf_correction` performs — written by a different group, in a
different basis convention, from the same paper. So the molecular path is **not** restricted
to a single atom here: "off-atom blocks are exactly zero" and "the diagonal blocks landed on
the right AOs" stop being assertions about Kuiva's own code and become a cross-code
comparison, at the cost of a few lines (``tests/test_amf_molecular.py``).

⚠ Two asymmetries survive it and are refused rather than absorbed: the plugin's ``aoc`` is one
switch for the whole molecule where Kuiva decides per element, and it has no configuration
input at all, so a molecule containing a lanthanide cannot be compared against Kuiva's M(3+)
default without pinning both sides to neutral.

Basis
-----
The plugin evaluates its atomic integrals over the fully **uncontracted** primitive set
(``pyscf.gto.mole.uncontracted_basis``) and returns the correction over PySCF's ``xmol``.
Kuiva decontracts with ``Mole.decontract_basis(aggregate=True)``. The two agree — verified,
not assumed: :func:`_decontracted` checks the overlap matrices element by element and refuses
on any mismatch, because an **ordering** difference between them would be silent and would
produce a Hermitian correction of plausible magnitude over the wrong functions (the failure
failure mode the X2CAMF design exists to prevent).

References
----------
* X2CAMF: J. Liu, L. Cheng, J. Chem. Phys. 148, 144108 (2018), doi:10.1063/1.5023750.
* The plugin and its spherical-symmetry atomic solver: C. Zhang, L. Cheng, J. Phys. Chem. A
  126, 4537 (2022), doi:10.1021/acs.jpca.2c02181; https://github.com/Warlocat/x2camf.
* pybind11, which the plugin's Python interface is built on: W. Jakob, J. Rhinelander,
  D. Moldovan, https://github.com/pybind/pybind11 (2017).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..spinor.expand import decompose_two_component
from ..util.logging import get_logger
from .backend import INTERACTIONS
from .configuration import AtomicConfiguration

log = get_logger(__name__)

#: What the plugin can be asked for. ``"pcc"`` is the two-electron picture-change correction
#: in full — the counterpart of Kuiva's ``dG`` — and the default. ``"soc"`` is the plugin's own
#: default entry point and carries **only the spin-dependent part**; it is offered for the
#: term-by-term comparison, where seeing the scalar term go missing is the point, and it is
#: rejected as a Hamiltonian correction by :func:`plugin_correction` unless asked for by name.
VARIANTS = ("pcc", "soc")

#: ``interaction`` mapped onto the plugin's two independent flags. The plugin
#: defaults *both* to ``True``, i.e. to full Breit — so every call here passes them explicitly.
_INTERACTION_FLAGS = {
    "coulomb": dict(with_gaunt=False, with_gauge=False),
    "gaunt": dict(with_gaunt=True, with_gauge=False),
    "breit": dict(with_gaunt=True, with_gauge=True),
}


def available() -> bool:
    """Whether the ``x2camf`` plugin can be imported. The only honest way to ask."""
    try:
        import x2camf                                                       # noqa: F401
    except Exception:                                                       # noqa: BLE001
        return False
    return True


def version() -> str:
    """The plugin's declared version, or ``""`` if it is not installed.

    ⚠ The upstream repository publishes no releases and hard-codes ``0.1``, so this string
    does **not** identify the code. The commit is what pins it
    (``scripts/bootstrap/versions.env::X2CAMF_COMMIT``), and that is what a reference record
    must carry.
    """
    try:
        import x2camf
    except Exception:                                                       # noqa: BLE001
        return ""
    return str(getattr(x2camf, "__version__", "unknown"))


def _raise_stack_limit() -> None:
    """Lift ``RLIMIT_STACK`` to its hard limit, because the plugin **segfaults** without it.

    ⚠ This is not defensive hygiene, it is a fix for an observed crash. The plugin's own
    README says "the default stack size might not be enough for larger systems due to the
    implementation. Use `ulimit -s unlimited`", and the failure mode is exactly as bad as that
    sounds: Ne (24 primitives) is fine and **Ar (51) dumps core**, taking the whole Python
    process with it — no exception, no traceback, and a reference generator that had already
    written four good records simply dies. Doing it here rather than in a shell wrapper means
    every entry point is covered, including an interactive ``method="x2camf-external"``.

    Linux grows the main thread's stack on demand and checks it against the *current* soft
    limit, so raising it at runtime is effective and needs no re-exec. Failure to raise it is
    a warning rather than an error: the limit may already be adequate, and refusing to run
    would be worse than crashing informatively.
    """
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        if hard != resource.RLIM_INFINITY and soft >= hard:
            return
        if soft != resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
            log.debug("raised RLIMIT_STACK from %s to %s for the x2camf plugin", soft, hard)
    except Exception as exc:                                                # noqa: BLE001
        log.warning("could not raise the stack limit for the x2camf plugin (%s: %s). Its "
                    "atomic solver can exhaust the default stack and abort the process "
                    "without an exception; run under `ulimit -s unlimited` if it does.",
                    type(exc).__name__, exc)


def _require():
    try:
        import x2camf
    except Exception as exc:                                                # noqa: BLE001
        raise ImportError(
            "the x2camf plugin is not installed ({}: {}). It is an optional, "
            "reference-only dependency; build it with "
            "`bash scripts/bootstrap/80_x2camf.sh`. Kuiva's own X2CAMF "
            "(method=\"x2camf\") needs nothing external.".format(type(exc).__name__, exc))
    _raise_stack_limit()
    return x2camf


def _decontracted(mol):
    """``(xmol, contraction)`` for the atom, with the plugin's basis checked against Kuiva's.

    Returns the molecule the plugin's correction comes back over and the
    ``(nao_primitive, nao_molecular)`` matrix that takes it to the molecular basis.

    ⚠ The check is on the **overlap matrices**, not on shapes or shell counts. Two
    decontraction routines can agree on how many primitives there are and disagree on their
    order, and that difference is invisible in every norm-based test while being fatal.
    """
    from pyscf.x2c import x2c

    xmol, contraction = mol.decontract_basis(aggregate=True)
    plugin_xmol, _ = x2c.X2C(mol).get_xmol()
    if int(plugin_xmol.nao) != int(xmol.nao):
        raise NotImplementedError(
            "the plugin works over {} primitive functions for this basis where Kuiva's "
            "decontraction gives {}. The correction would be expressed over a different set "
            "of functions from the one it has to be added to.".format(plugin_xmol.nao,
                                                                      xmol.nao))
    diff = float(np.max(np.abs(plugin_xmol.intor("int1e_ovlp")
                               - xmol.intor("int1e_ovlp"))))
    if diff > 1e-10:
        raise NotImplementedError(
            "the plugin's primitive basis and Kuiva's differ (max |dS| = {:.2e}) even though "
            "both have {} functions — an AO ordering or normalization mismatch. Comparing "
            "the two corrections would compare different functions.".format(diff, xmol.nao))
    return xmol, np.ascontiguousarray(np.asarray(contraction, dtype=float))


def _contract(a: np.ndarray, contraction: Optional[np.ndarray]) -> np.ndarray:
    """Take a ``(2n, 2n)`` two-component operator to the molecular basis.

    The same block-diagonal congruence :meth:`kuiva.amf.backend.AtomicDiracSolution.contract`
    applies, repeated here rather than shared because there is no solution object in this path
    — the plugin returns a matrix and nothing else.
    """
    if contraction is None:
        return np.ascontiguousarray(a, dtype=np.complex128)
    n, m = contraction.shape
    if a.shape != (2 * n, 2 * n):
        raise ValueError("expected a ({0}, {0}) operator, got {1}".format(2 * n, a.shape))
    big = np.zeros((2 * n, 2 * m), dtype=float)
    big[:n, :m], big[n:, m:] = contraction, contraction
    return np.ascontiguousarray(big.T @ a @ big)


#: Strings the plugin writes to **stdout** when something went wrong, with what they mean.
#: Scanned for by :func:`_captured`, because none of them raises anything a caller can see.
_DIAGNOSTICS = (
    ("SCF did not converge",
     "the plugin's four-component atomic SCF did not reach its 1e-9 convergence threshold"),
    ("Something went wrong", "the plugin reported an internal inconsistency"),
    ("ERROR", "the plugin reported an error"),
)


class _captured:
    """Capture the plugin's C-level stdout so its diagnostics stop being invisible.

    ⚠ **The plugin reports failure by printing and continuing.** Its ``x2c2ePCC`` prints
    ``"SCF did not converge. x2c2ePCC cannot be used!"`` and then returns the matrix anyway
    (the ``exit(99)`` beside that message is commented out upstream), so from Python the call
    is indistinguishable from a clean one. Measured on neutral Ti and Bi in
    ``x2c-SVPall-2c``: the message appears **and** the resulting ``max |dw|`` agrees with
    Kuiva's independently converged value to 0.3% — so the right response is a loud
    warning, not a refusal. Refusing would discard a working bisection tool on the strength of
    a message whose threshold Kuiva does not control; ignoring it would be the "silently poor
    correction" this project keeps finding.

    ⚠ **What this cannot catch is the other failure mode.** On neutral Ce the plugin prints
    ``"ERROR: Matrix has negative eigenvalues!"`` and calls ``exit(99)`` — the Python process
    dies with no exception and no traceback. Nothing in-process can intercept that, which is
    why :mod:`tests.generate.x2camf_plugin` writes every record as soon as it exists.

    Redirection is at the file-descriptor level (``os.dup2`` on fd 1) because the output comes
    from C++ ``std::cout`` and never passes through :data:`sys.stdout`. Any failure to
    redirect degrades to running uncaptured rather than to not running.
    """

    def __init__(self, what: str) -> None:
        self.what = what
        self._fd = None
        self._saved = None
        self._file = None

    def __enter__(self) -> "_captured":
        import os
        import tempfile
        try:
            self._file = tempfile.TemporaryFile(mode="w+b")
            self._saved = os.dup(1)
            self._fd = self._file.fileno()
            import sys as _sys
            _sys.stdout.flush()
            os.dup2(self._fd, 1)
        except Exception as exc:                                            # noqa: BLE001
            log.debug("could not capture the x2camf plugin's stdout (%s: %s); its "
                      "diagnostics will only appear on the terminal",
                      type(exc).__name__, exc)
            self._saved = None
        return self

    def __exit__(self, *exc) -> bool:
        import os
        import sys as _sys
        if self._saved is not None:
            try:
                _sys.stdout.flush()
                os.dup2(self._saved, 1)
                os.close(self._saved)
                self._file.seek(0)
                text = self._file.read().decode("utf-8", "replace")
            except Exception:                                               # noqa: BLE001
                text = ""
            finally:
                self._file.close()
            log.debug("x2camf plugin output for %s:\n%s", self.what, text.strip())
            for marker, meaning in _DIAGNOSTICS:
                if marker in text:
                    log.warning("the x2camf plugin reported %r while computing %s: %s. The "
                                "correction is returned so it can still be compared, but "
                                "nothing built on it is trustworthy without checking it "
                                "against Kuiva's own method=\"x2camf\".",
                                marker, self.what, meaning)
                    break
        return False


def guard_arguments(mol, *, configuration=None, uncontract: bool = True) -> None:
    """Refuse the two things :func:`kuiva.amf.correction.amf_correction` can express and the
    plugin cannot, rather than silently comparing two different calculations.

    Shared by both entry points so that ``method="x2camf-external"`` and a direct call refuse
    identically — a guard that only one path applies is worse than none, because the reference
    generator and the production path would then disagree about what was computed.

    ⚠ ``configuration`` is checked for **every** element of the molecule, and a mapping is
    accepted for the same reason :func:`kuiva.amf.correction.amf_correction` takes one: a
    molecule containing a lanthanide takes Kuiva's M(3+) default per element, and the plugin
    has no way to express it. That is the difference this guard exists to make loud, and it
    is what forces the molecular cross-check onto neutral references on both sides.
    """
    if configuration is not None:
        symbols = {mol.atom_pure_symbol(ia) for ia in range(mol.natm)}
        if isinstance(configuration, dict):
            wanted_by_symbol = {}
            for ia in range(mol.natm):
                pure = mol.atom_pure_symbol(ia)
                spec = configuration.get(mol.atom_symbol(ia), configuration.get(pure))
                if spec is not None:
                    wanted_by_symbol[pure] = spec
        else:
            wanted_by_symbol = {s: configuration for s in symbols}
        for symbol, spec in sorted(wanted_by_symbol.items()):
            wanted = AtomicConfiguration.coerce(spec, symbol)
            if wanted != AtomicConfiguration.ground(symbol):
                raise NotImplementedError(
                    "the x2camf plugin computes the mean field of the neutral atom and takes "
                    "no configuration input, so it cannot be asked for {} ({}). Kuiva's own "
                    "method=\"x2camf\" can; comparing the two therefore has to use the "
                    "neutral reference on both sides.".format(wanted.label, wanted.canonical))
    if not uncontract:
        raise NotImplementedError(
            "the x2camf plugin always decouples in the primitive basis; uncontract=False has "
            "no counterpart in it. Compare against Kuiva's uncontract=True, which is the "
            "default and the physically correct choice.")


def plugin_hamiltonian(mol, *, interaction: str = "coulomb", variant: str = "pcc",
                       aoc: Optional[bool] = None) -> np.ndarray:
    """The plugin's correction as a ``(2*nao_primitive, 2*nao_primitive)`` operator.

    In the spin-blocked ``[alpha; beta]`` basis of the **primitive** functions — the
    basis the plugin works in. :func:`plugin_correction` is the version a caller wants;
    this one exists for the term-by-term comparison, which must not go through a contraction.
    """
    x2camf = _require()
    from pyscf.x2c import x2c

    from .pyscf_dhf import spinor_to_spin_orbital

    if interaction not in INTERACTIONS:
        raise ValueError("unknown two-electron interaction {!r}; expected one of {}".format(
            interaction, INTERACTIONS))
    if variant not in VARIANTS:
        raise ValueError("unknown plugin variant {!r}; expected one of {}".format(
            variant, VARIANTS))
    if aoc is None:
        # ⚠ The plugin's ``aoc`` flag is one switch for the whole molecule, not one per
        # element, so it is turned on as soon as **any** element has an open-shell ground
        # configuration. That is safe rather than a compromise: the average of configuration
        # over a *closed* shell is the closed shell, so the flag changes nothing for the
        # elements that do not need it. Kuiva's own path decides this per element, which is
        # one more reason the two are compared and not swapped.
        aoc = any(not AtomicConfiguration.ground(mol.atom_pure_symbol(ia)).is_closed_shell
                  for ia in range(mol.natm))
    flags = dict(_INTERACTION_FLAGS[interaction])
    flags["aoc"] = bool(aoc)
    flags["pcc"] = variant == "pcc"

    xo = x2c.X2C(mol)
    xmol, _ = xo.get_xmol()
    formula = "".join(sorted({mol.atom_pure_symbol(ia) for ia in range(mol.natm)}))
    with _captured("{} ({}, {})".format(formula, interaction, variant)):
        matrix = np.asarray(x2camf.amfi(xo, **flags))
    if matrix.shape != (xmol.nao_2c(), xmol.nao_2c()):
        raise RuntimeError(
            "the plugin returned a {} matrix where ({}, {}) was expected for this "
            "basis".format(matrix.shape, xmol.nao_2c(), xmol.nao_2c()))
    u = spinor_to_spin_orbital(xmol)
    return np.ascontiguousarray(u @ matrix.astype(np.complex128) @ u.conj().T)


def plugin_correction_matrix(mol, *, interaction: str = "coulomb", variant: str = "pcc",
                             configuration=None, uncontract: bool = True,
                             aoc: Optional[bool] = None) -> np.ndarray:
    """The plugin's correction as a ``(2*nao, 2*nao)`` operator in ``mol``'s **own** AO basis.

    :func:`plugin_correction` decomposed from this, and
    :func:`kuiva.amf.correction.amf_correction` needs the assembled matrix rather than the
    decomposition — it measures the time-reversal residual before the projection throws it
    away. One function so the two cannot drift apart.
    """
    guard_arguments(mol, configuration=configuration, uncontract=uncontract)
    dg = plugin_hamiltonian(mol, interaction=interaction, variant=variant, aoc=aoc)
    xmol, contraction = _decontracted(mol)
    if dg.shape[0] != 2 * int(xmol.nao):
        raise RuntimeError(
            "the plugin's correction spans {} spin-orbitals where the primitive basis has "
            "{}".format(dg.shape[0], 2 * xmol.nao))
    return _contract(dg, contraction)


def plugin_correction(mol, *, interaction: str = "coulomb", variant: str = "pcc",
                      configuration=None, uncontract: bool = True,
                      aoc: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
    """``(delta h_sf, delta w)`` from the plugin, in ``mol``'s AO basis and spin-blocked conventions.

    The signature deliberately mirrors :func:`kuiva.amf.correction.amf_correction`'s so the
    two are interchangeable in a comparison, but three of its arguments cannot be honoured and
    are refused rather than ignored:

    * ``configuration`` — the plugin has no configuration input at all (point 3 of the module
      docstring). Only the neutral atom's ground configuration is accepted.
    * ``uncontract=False`` — the plugin decontracts unconditionally, so there is no way to ask
      it to decouple in a contracted basis.
    * a modified speed of light is *not* refused but is not a parameter either: the plugin
      reads ``pyscf.lib.param.LIGHT_SPEED`` at call time, so
      :func:`kuiva.amf.pyscf_dhf.light_speed` controls both codes at once.
    """
    return decompose_two_component(plugin_correction_matrix(
        mol, interaction=interaction, variant=variant, configuration=configuration,
        uncontract=uncontract, aoc=aoc))


__all__ = ["VARIANTS", "available", "guard_arguments", "plugin_correction",
           "plugin_correction_matrix", "plugin_hamiltonian", "version"]
