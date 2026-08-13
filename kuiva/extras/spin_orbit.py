"""One-electron spin-orbit constants ``zeta_nl`` of an atom's shells.

What is extracted, and why it is one number per shell
-----------------------------------------------------
Restricted to a single shell — one radial function, one angular momentum ``l``, spin
``1/2`` — the spin-dependent part of *any* spherically symmetric one-electron operator is
proportional to ``l . s`` and to nothing else. There is no freedom: the shell's ``2(2l+1)``
spinors carry the two irreducible representations ``j = l +- 1/2`` once each, so a scalar
operator odd in the spin has exactly one independent matrix element. That number is the
spin-orbit constant,

.. math::

    \\langle n l m_l \\sigma | H^{SO} | n l m_l' \\sigma' \\rangle
        = \\zeta_{nl}\\, (\\mathbf{l} \\cdot \\mathbf{s})_{m_l \\sigma,\\, m_l' \\sigma'} ,

and the level splitting it produces is ``E(l + 1/2) - E(l - 1/2) = (2l + 1) \\zeta_{nl} / 2``.

The operator taken apart is the two-component X2C one-electron Hamiltonian as the front end
already stores it, ``H = h_{sf} \\otimes 1_2 + \\sigma \\cdot W`` with ``W_k = i w_k`` and
``w_k`` real antisymmetric (:class:`kuiva.interface.pyscf_bridge.SpinOrbitX2C`). The
spin-free half is untouched; the three ``w_k`` are transformed to the shell's orbitals, the
Pauli structure is put back, and the result is fitted to ``l . s``.

⚠ **What the residual of that fit does and does not see, measured rather than assumed.** It
sees the conventions — the ``m`` ordering of the shell orbitals, the spin-blocked row layout,
the Pauli assembly — and it sees shell orbitals whose ``m`` channels have been mixed or
scaled unequally: a permutation of two ``m`` channels puts it at 1.0 and turns ``zeta``
negative, a rotation of two channels at 0.4, against 6e-8 for a clean shell. It does **not**
see two other things, and both are worth stating because the obvious reading is that it
would:

* **another shell's character leaking into one channel.** The atomic spin-orbit operator is
  block diagonal in ``l`` — for a spherical atom it is ``f(r) l``, which has no matrix element
  between different ``l`` — so mixing a little ``2s`` into a ``2p`` channel changes the block
  by nothing at all. The radial-parameter extraction *does* catch that case, because the
  Coulomb interaction couples different ``l`` freely. The two diagnostics are complementary
  and neither subsumes the other;
* **a non-spherical solution.** The one-electron operator is spherical whatever the SCF did
  with the density, and the shell orbitals are built pure-``m`` by construction. Measured: a
  symmetry-broken ROHF oxygen (Fock anisotropy 0.27) and a converged average of configuration
  (2e-11) give residuals of 4.7e-9 and 4.7e-9 — indistinguishable — while their constants
  differ by 1.7%. :attr:`kuiva.extras.shells.AtomicShells.anisotropy` remains the **only**
  diagnostic of a non-spherical solution, which is the same conclusion the radial-parameter
  extraction reached for a different reason, and the two are reported together.

**Only shell-diagonal constants are extracted.** Two shells of the *same* ``l`` also have an
off-diagonal constant ``zeta(nl, n'l)`` — the same operator taken between two different radial
functions — and nothing here computes it. It is not needed by the case this feature was built
for (the shells ``4f``, ``5d``, ``6s`` share no ``l``), and adding it is a widening of
:class:`SpinOrbitConstant` rather than a change of method: the block would be rectangular and
the model the same ``l . s``.

⚠ **An s shell has no spin-orbit constant, and is absent rather than reported as zero.**
``l . s`` vanishes identically for ``l = 0``, so no multiple of it is determined by anything —
"the fit returned 0" and "there is nothing to fit" are different statements. There is
deliberately **no check that its operator block vanishes**, tempting as one looks: a shell of
``l = 0`` is a single real orbital and the ``w_k`` are real antisymmetric, so ``C^T w_k C = 0``
is an algebraic identity that holds for any vector whatsoever, contaminated or not. It would
be a check that cannot fail.

What the constants contain
--------------------------
The one-electron X2C spin-orbit operator, **plus the two-electron screening if and only if
the screening record says so** — the record travels with the Hamiltonian
(:meth:`kuiva.interface.pyscf_bridge.SpinOrbitX2C.provenance`) and is the authority, since
the difference between a screened and an unscreened constant is 5-30%.

⚠ **These are frozen-orbital constants of one fixed average-of-configuration reference.**
They are not self-consistent two-component splittings, and the two constructions differ by
tens of per cent on the absolute value of a splitting. Any number quoted from them states
which of the two produced it, and is never compared against a value obtained in a different
basis set.

References
----------
* E. U. Condon, G. H. Shortley, *The Theory of Atomic Spectra*, Cambridge University Press
  (1935), Chapter XI — ``zeta_{nl}``, the ``l . s`` form of the one-electron spin-orbit
  operator within a shell, and the ``(2l+1) zeta / 2`` level splitting.
* R. D. Cowan, *The Theory of Atomic Structure and Spectra*, University of California Press
  (1981), Chapter 10 — spin-orbit parameters in the same convention as the ``F^k`` and
  ``G^k`` this module's constants are reported beside.
* J. Liu, L. Cheng, "An atomic mean-field spin-orbit approach within exact two-component
  theory for a non-perturbative treatment of spin-orbit coupling", J. Chem. Phys. **148**,
  144108 (2018), doi:10.1063/1.5023750 — the two-electron screening these constants contain
  when the screening record says they do.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from kuiva.extras.angular import spin_orbit_matrix
from kuiva.extras.shells import AtomicShells, ShellOrbitals
from kuiva.util import output as out
from kuiva.util.logging import get_logger

log = get_logger(__name__)

#: Largest relative deviation of a shell's spin-orbit operator from ``zeta l . s`` before the
#: extraction warns, scaled by the largest element of the operator block itself.
#:
#: ⚠ **It is a consistency diagnostic and not a sphericity one** — the module docstring says
#: what it can and cannot fail on. Measured on converged average-of-configuration atoms in
#: ``x2c-SVPall-2c``, from boron to cerium: **1e-10 to 1e-6** relative, against **0.4 to 1.0**
#: for shell orbitals mixed across ``m`` channels.
#:
#: ⚠ **It is meaningful only above the floor of the second gate below**, and on a light atom it
#: is not: boron's 2p block is reproduced to 3e-12 **Eh**, which is 6e-8 of a block whose
#: largest element is 5e-5 Eh. A relative bound alone would fire on every light element and on
#: nothing else.
ZETA_RESIDUAL_TOLERANCE = 1e-8

#: The floor under the residual, and it is **measured per run rather than chosen**: the X2C
#: decoupling's own discarded time-reversal-odd part
#: (:attr:`kuiva.interface.pyscf_bridge.SpinOrbitX2C.tr_residual`, in Eh), which is the front
#: end's measure of how exactly the matrix square root in the transformation preserved the
#: symmetries this fit relies on. A deviation below it is that rounding and says nothing about
#: the shell.
#:
#: It spans six orders across the periodic table, which is exactly why it cannot be a
#: constant: 1.8e-11 Eh (B), 6.8e-11 (Xe), **1.6e-4** (Ce). The residuals measured beside them
#: were 3.2e-12, 3.5e-11 and 3.5e-8 Eh — below the floor in every case, and by a factor that
#: is not the same twice. This value is only the backstop for an operator that carries no such
#: measurement at all (one assembled by hand in a test).
ZETA_DECOUPLING_FLOOR = 1e-12

#: The Pauli matrices, ``sigma_k = 2 s_k``. The factor of two against the spin operator is
#: **not** applied anywhere here: the operator is assembled with ``sigma``, exactly as
#: :func:`kuiva.spinor.expand.two_component_operator` assembles it, and the model it is fitted
#: to is ``l . s``, so the factor lands inside ``zeta`` where the convention puts it.
_PAULI = np.array([[[0.0, 1.0], [1.0, 0.0]],
                   [[0.0, -1j], [1j, 0.0]],
                   [[1.0, 0.0], [0.0, -1.0]]], dtype=np.complex128)


def shell_spin_orbit_block(shell: ShellOrbitals, soc) -> np.ndarray:
    """The spin-orbit operator inside one shell: ``(2(2l+1), 2(2l+1))`` complex Hermitian.

    ``soc`` is a :class:`kuiva.interface.pyscf_bridge.SpinOrbitX2C` in the **AO basis** the
    shell orbitals are expressed in. The three real antisymmetric factors ``w_k`` are each
    transformed by the shell's ``2l+1`` orbitals and the Pauli structure is reassembled,

    .. math::

        H^{SO}_{\\text{shell}} = \\sum_k \\sigma_k \\otimes (i\\, C^T w_k C) .

    The row layout is **spin-blocked**: the first ``2l+1`` rows are the alpha component and
    the last ``2l+1`` the beta component, each in ascending ``m``. That is the layout of
    :func:`kuiva.extras.angular.spin_orbit_matrix` and of the two-component code throughout,
    so the two can be contracted with no reordering — and it is the one convention here that
    a mistake in would produce a Hermitian, plausible, wrong constant.
    """
    c = np.asarray(shell.coefficients, dtype=float)
    w = np.asarray(soc.w)
    if w.shape[-1] != c.shape[0]:
        raise ValueError(
            "the shell orbitals span {} AO functions and the spin-orbit operator {}; they are "
            "not from the same calculation".format(c.shape[0], w.shape[-1]))
    # Three congruences on (nao, 2l+1) at most once per shell per run: einsum is orchestration
    # here, not a kernel, and nothing in this module belongs on a hot path.
    w_shell = np.einsum("mp,kmn,nq->kpq", c, w, c, optimize=True)
    block = np.zeros((2 * c.shape[1], 2 * c.shape[1]), dtype=np.complex128)
    for k in range(3):
        block += np.kron(_PAULI[k], 1j * w_shell[k])
    return block


@dataclass(frozen=True)
class SpinOrbitConstant:
    """One shell's spin-orbit constant, with the quality of the fit it came from.

    Attributes
    ----------
    n, l : int
    shell : str
        The shell label, ``"4f"``.
    zeta : float
        The constant [Eh], in the convention ``H^{SO} = zeta l . s``. Positive for a
        less-than-half-filled shell, where ``j = l - 1/2`` lies lowest.
    rms_residual, max_residual : float
        Deviation of the operator block from ``zeta l . s`` [Eh]; ``relative_residual``
        scales the larger of them by the largest element of the block.
    operator_scale : float
        ``max |H^{SO}|`` over the block [Eh] — the size of what was fitted.
    splitting : float
        ``(2l+1) zeta / 2`` [Eh], the ``j = l + 1/2`` against ``j = l - 1/2`` separation this
        constant implies for a single electron in the shell. ⚠ A **frozen-orbital,
        one-electron** splitting of one fixed average-of-configuration reference — not the
        splitting of a term, and not a self-consistent two-component one.
    """

    n: int
    l: int
    shell: str
    zeta: float
    rms_residual: float
    max_residual: float
    relative_residual: float
    operator_scale: float

    @property
    def zeta_cm(self) -> float:
        """The constant in wavenumbers, the unit spin-orbit constants are quoted in."""
        from kuiva.props.multiplet import HARTREE_TO_CM

        return self.zeta * HARTREE_TO_CM

    @property
    def splitting(self) -> float:
        """``(2l+1) zeta / 2`` [Eh] — see the class docstring for what it is and is not."""
        return 0.5 * (2 * self.l + 1) * self.zeta

    @property
    def splitting_cm(self) -> float:
        from kuiva.props.multiplet import HARTREE_TO_CM

        return self.splitting * HARTREE_TO_CM

    def __repr__(self) -> str:
        return "SpinOrbitConstant(zeta({}) = {:.2f} cm^-1)".format(self.shell, self.zeta_cm)


@dataclass(frozen=True)
class SpinOrbitConstants:
    """The constants extracted from one solution, and the Hamiltonian they came from.

    ``provenance`` is :meth:`kuiva.interface.pyscf_bridge.SpinOrbitX2C.provenance` — which
    decoupling, and **whether the two-electron screening is already in these numbers**. It
    travels with them into every stored product, because a constant that does not say whether
    it was screened is not interpretable.
    """

    constants: Tuple[SpinOrbitConstant, ...]
    provenance: Dict[str, object]
    decoupling_floor: float = 0.0

    def __len__(self) -> int:
        return len(self.constants)

    def __iter__(self):
        return iter(self.constants)

    def __getitem__(self, key) -> SpinOrbitConstant:
        """By shell label (``zeta["4f"]``) or by position."""
        if isinstance(key, int):
            return self.constants[key]
        for constant in self.constants:
            if constant.shell == key:
                return constant
        raise KeyError(
            "no spin-orbit constant for {}; this set holds {}. An s shell has none — l . s "
            "vanishes identically — and is absent rather than listed as zero.".format(
                key, ", ".join(c.shell for c in self.constants) or "nothing"))

    def as_dict(self) -> Dict[str, float]:
        """``{shell: zeta in Eh}`` — for reference records and JSON provenance."""
        return {c.shell: c.zeta for c in self.constants}

    @property
    def max_relative_residual(self) -> float:
        return max((c.relative_residual for c in self.constants), default=0.0)

    def report(self, logger=None) -> None:
        """The constants table, through the output grammar."""
        logger = logger or log
        if not self.constants:
            out.note(logger, "no shell of nonzero angular momentum: no spin-orbit constant "
                             "is defined (l . s vanishes for an s shell)")
            return
        table = out.Table(logger, [out.Column("shell", "{:s}", 6),
                                   out.Column("l", "{:d}", 3),
                                   out.Column("zeta [Eh]", out.E_FMT, 21),
                                   out.Column("zeta [cm^-1]", out.CM_FMT, 14),
                                   out.Column("(2l+1)zeta/2 [cm^-1]", out.CM_FMT, 21),
                                   out.col_sci("residual")])
        table.start("one-electron spin-orbit constants, H_SO = zeta l.s")
        for c in self.constants:
            table.row(c.shell, c.l, c.zeta, c.zeta_cm, c.splitting_cm, c.relative_residual)
        table.end("residual is relative to the shell's own operator block; the X2C decoupling "
                  "floor under it is {:.1e} Eh".format(self.decoupling_floor))

    def __repr__(self) -> str:
        return "SpinOrbitConstants({}, worst residual {:.1e})".format(
            ", ".join("{}={:.1f} cm^-1".format(c.shell, c.zeta_cm)
                      for c in self.constants) or "none", self.max_relative_residual)


def extract_spin_orbit(shells: AtomicShells, data=None, *, soc=None,
                       residual_tol: float = ZETA_RESIDUAL_TOLERANCE,
                       decoupling_floor: float = ZETA_DECOUPLING_FLOOR) -> SpinOrbitConstants:
    """Fit ``zeta_nl`` to the two-component one-electron Hamiltonian over ``shells``.

    Parameters
    ----------
    shells : AtomicShells
        The ``m``-aligned shell orbitals of a converged atomic solution
        (:func:`kuiva.extras.shells.extract_shells`).
    data
        The ingested solution the shells came from; its :attr:`soc` supplies the operator.
        ⚠ A solution ingested with ``with_soc=False`` carries none, and the extraction
        **refuses** rather than return constants of an operator that is not there.
    soc
        A :class:`kuiva.interface.pyscf_bridge.SpinOrbitX2C` given directly, in place of
        ``data`` — for comparing two Hamiltonians over one set of shells.
    residual_tol : float
        Relative residual above which a shell warns — **and only if the deviation also
        exceeds the absolute floor below**, since one of the two gates alone is wrong at one
        end of the periodic table. See :data:`ZETA_RESIDUAL_TOLERANCE`.
    decoupling_floor : float
        Backstop [Eh] for the absolute gate, used only when the operator carries no
        ``tr_residual`` of its own. See :data:`ZETA_DECOUPLING_FLOOR`.

    Returns
    -------
    SpinOrbitConstants
        One entry per shell with ``l > 0``, in the order the shells were extracted. ``s``
        shells are absent by construction — the class docstring says why.

    Notes
    -----
    The fit is the projection ``zeta = <l.s, H> / <l.s, l.s>`` with the real inner product
    ``Re Tr(A^dag B)``, i.e. ordinary least squares over every element of the block at once.
    It is a one-parameter fit to ``4(2l+1)^2`` numbers that symmetry makes exact, so — as in
    the radial-parameter extraction — the least squares is a diagnostic device rather than a
    statistical one, and the residual is the quantity of interest rather than a by-product.
    """
    if soc is None:
        soc = getattr(data, "soc", None)
        if soc is None:
            raise ValueError(
                "spin-orbit constants need the two-component one-electron Hamiltonian and "
                "this solution carries none: it was ingested with with_soc=False (or "
                "screening was never resolved). Re-run the reference with spin-orbit "
                "ingestion on, or ask only for the radial parameters.")

    floor = max(float(getattr(soc, "tr_residual", 0.0)), float(decoupling_floor))
    constants = []
    for shell in shells:
        if shell.l == 0:
            # ⚠ Nothing to fit, and deliberately nothing checked either: ``l . s`` vanishes
            # for l = 0, and the block does too as an algebraic identity rather than as a
            # physical statement (module docstring). A check here could not fail.
            continue

        block = shell_spin_orbit_block(shell, soc)
        scale = float(np.max(np.abs(block))) if block.size else 0.0
        model = spin_orbit_matrix(shell.l)
        overlap = float(np.real(np.vdot(model, model)))
        zeta = float(np.real(np.vdot(model, block))) / overlap
        deviation = block - zeta * model
        rms = float(np.sqrt(np.mean(np.abs(deviation) ** 2)))
        worst = float(np.max(np.abs(deviation))) if deviation.size else 0.0
        relative = worst / max(scale, 1e-300)
        if relative > residual_tol and worst > floor:
            log.warning(
                "the %s spin-orbit operator is reproduced by zeta * l.s only to a relative "
                "%.2e (bound %.0e, and %.2e Eh absolute against a decoupling floor of %.1e), "
                "where symmetry makes that form exact for one shell of a spherical atom. The "
                "shell's orbitals are then not one radial function per m channel, or the "
                "solution they came from is not spherical; zeta describes the operator only "
                "as well as this number.",
                shell.label, relative, residual_tol, worst, floor)
        constants.append(SpinOrbitConstant(
            n=shell.n, l=shell.l, shell=shell.label, zeta=zeta, rms_residual=rms,
            max_residual=worst, relative_residual=relative, operator_scale=scale))

    provenance = soc.provenance() if hasattr(soc, "provenance") else {}
    return SpinOrbitConstants(constants=tuple(constants), provenance=provenance,
                              decoupling_floor=floor)


__all__ = ["SpinOrbitConstant", "SpinOrbitConstants", "ZETA_DECOUPLING_FLOOR",
           "ZETA_RESIDUAL_TOLERANCE", "extract_spin_orbit", "shell_spin_orbit_block"]
