"""AO basis layout: the per-function metadata that crosses the ingestion boundary.

What this is for
----------------
Everything downstream of the front-end works in the orthonormal working basis and then in the
spinor-MO basis, where — by design — contraction type and AO identity are
deliberately invisible. Two things need them back:

* **Loewdin population analysis** (:mod:`kuiva.props.population`) needs to know which atom and
  which angular-momentum function each AO belongs to;
* **the molden dump** (:mod:`kuiva.props.molden`) needs the primitives themselves, plus the
  geometry, to write a file another program can evaluate on a grid.

Both are *analysis* of the result, not steps of the calculation, so this record is carried
along rather than consumed: it is built once in the front-end
(:func:`kuiva.interface.pyscf_bridge.ao_layout`), stored on
:class:`kuiva.interface.pyscf_bridge.ScalarX2CData`, and is plain NumPy arrays and strings —
no ``Mole``, nothing that would put PySCF back on the multireference path.

Conventions fixed here
----------------------
**Angular momentum.** ``spdfghi`` with ``l = 0..6``. Bases are **spherical** throughout —
Cartesian ones are refused at ingestion — so a shell of angular momentum ``l`` holds
exactly ``2l+1`` functions and :attr:`AOLayout.ao_m` is meaningful for every AO.

**AO ordering within a shell** is the integral library's, and ⚠ **it is not one rule**:
``libcint`` orders ``l >= 2`` as ``m = -l, -l+1, ..., +l`` but orders ``p`` shells as
``px, py, pz``, i.e. ``m = +1, -1, 0``. :func:`shell_m_values` is the single definition;
deriving ``m`` from the position in the shell by the ``l >= 2`` rule alone mislabels every
p function, and a mislabelled p function is invisible in any total population (the sum over
the shell is unchanged) while being wrong in every per-AO row.

Molden's ordering is ``m = 0, +1, -1, +2, -2, ...`` for ``l >= 2`` and ``px, py, pz`` for
``p`` — so the permutation is the identity for ``l <= 1`` and nontrivial above.
:func:`molden_ao_order` is the single definition of it, generated from ``l`` rather than
tabulated, which is what lets it extend past ``l = 4`` (see :mod:`kuiva.props.molden` for why
that is wanted and why it is not standard). It is asserted **equal to PySCF's own**
``molden.order_ao_index`` wherever that function is defined.

**General contractions are split.** A ``libcint`` shell may carry several contracted
functions over one set of primitives (``nctr > 1``); a :class:`Shell` here is always **one**
contracted function, because that is the granularity molden's ``[GTO]`` section works at and
because it makes the shell-to-AO map a simple run of ``2l+1``. The split preserves the AO
order exactly: the library lays out a general contraction contraction-major, m-minor.

.. warning::
   **The principal quantum number in :attr:`AOLayout.ao_labels` counts shells within the
   basis, not physical shells.** This is PySCF's convention and it is basis-dependent: the
   same physical orbital is ``"6p"`` in a segmented (Karlsruhe) set and ``"3p"`` in a general
   (ANO-RCC) one, because the latter contracts the core away. It is a live instance of the
   contraction-type trap, and it has already produced one wrong active-space selection
. **Group and select on ``(atom, l, m)``, which is unambiguous; treat the label as
   a human-readable annotation only.**

References
----------
* The molden file format and its ordering/normalization conventions: G. Schaftenaar,
  J. H. Noordik, "Molden: a pre- and post-processing program for molecular and electronic
  structures", J. Comput.-Aided Mol. Design 14, 123 (2000), doi:10.1023/A:1008193805436.
* Real solid-harmonic Gaussian conventions: H. B. Schlegel, M. J. Frisch,
  Int. J. Quantum Chem. 54, 83 (1995), doi:10.1002/qua.560540202.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Angular-momentum letters, indexed by ``l``.
ANGULAR = "spdfghi"

#: Highest angular momentum the molden format defines (``[9g]``). Above this a writer must
#: either drop the functions or invent a convention — see :mod:`kuiva.props.molden`.
MOLDEN_MAX_L = 4


@dataclass(frozen=True)
class Shell:
    """One **contracted** basis function's radial part, on one atom.

    ``exponents`` and ``coefficients`` are over normalized primitive Gaussians, which is both
    the integral library's storage convention and molden's file convention, so they are
    written out unchanged.
    """

    atom: int
    l: int
    exponents: np.ndarray            # (nprim,)
    coefficients: np.ndarray         # (nprim,)

    @property
    def nprim(self) -> int:
        return int(self.exponents.size)

    @property
    def nao(self) -> int:
        """Functions this shell contributes: ``2l+1`` (spherical)."""
        return 2 * self.l + 1

    def __repr__(self) -> str:
        return "Shell(atom={}, {}, nprim={})".format(self.atom, ANGULAR[self.l], self.nprim)


def shell_m_values(l: int) -> List[int]:
    """The ``m`` value of each function of a shell, **in integral-library order**.

    ``l = 1`` is the special case that catches people: ``libcint`` (and so PySCF) lays a p
    shell out as ``px, py, pz``, which is ``m = +1, -1, 0`` — not the ``-l..+l`` run that
    every other ``l`` follows.
    """
    if l == 1:
        return [1, -1, 0]
    return list(range(-l, l + 1))


def molden_m_order(l: int) -> List[int]:
    """Offsets into a shell's block, in molden's order.

    Molden orders the real solid harmonics ``m = 0, +1, -1, +2, -2, ...`` for ``l >= 2`` and
    ``px, py, pz`` for p. Since the library's own order already *is* molden's for ``l <= 1``,
    this is the identity there and a genuine permutation above: for ``l = 2`` it returns
    ``[2, 3, 1, 4, 0]``.

    Generated from ``l`` rather than tabulated, so it extends past ``l = 4`` by the obvious
    continuation of the same rule. That extension is **not** part of the molden standard and
    is only ever used when a caller asks for it explicitly (:mod:`kuiva.props.molden`).
    """
    if l <= 1:
        return list(range(2 * l + 1))
    order = [l]                                   # m = 0
    for k in range(1, l + 1):
        order.extend([l + k, l - k])              # m = +k, -k
    return order


def molden_ao_order(shells: Sequence[Shell], *, max_l: Optional[int] = MOLDEN_MAX_L
                    ) -> np.ndarray:
    """Permutation taking library AO order to molden AO order.

    ``coeff[molden_ao_order(shells)]`` is the coefficient vector as molden wants it. Shells
    with ``l > max_l`` are **omitted** from the result, so the permutation is in general an
    injection rather than a bijection and its length is the number of AOs actually written.
    ``max_l=None`` keeps everything.
    """
    order: List[int] = []
    offset = 0
    for sh in shells:
        if max_l is None or sh.l <= max_l:
            order.extend(offset + k for k in molden_m_order(sh.l))
        offset += sh.nao
    return np.asarray(order, dtype=int)


@dataclass(frozen=True)
class AOLayout:
    """Which atom, shell and angular-momentum function each AO belongs to, plus the geometry.

    Attributes
    ----------
    atom_symbols : tuple of str
        Element symbols, one per atom, in input order.
    atom_charges : ndarray (natm,)
        Nuclear charges. ECP bases are refused at ingestion, so these are the full ``Z``.
    coords_bohr : ndarray (natm, 3)
        Nuclear coordinates in **bohr** — molden's ``[Atoms] (AU)`` unit, and the one the
        integral library works in.
    shells : tuple of Shell
        One entry per contracted function, in AO order.
    ao_atom, ao_l, ao_m, ao_shell : ndarray (nao,), int
        Per-AO atom index, angular momentum, magnetic quantum number (``-l..+l``) and index
        into :attr:`shells`.
    ao_labels : tuple of str
        Human-readable ``"5g+2"``-style labels, one per AO. ⚠ Basis-dependent principal
        quantum number — see the module docstring.
    """

    atom_symbols: Tuple[str, ...]
    atom_charges: np.ndarray
    coords_bohr: np.ndarray
    shells: Tuple[Shell, ...]
    ao_atom: np.ndarray
    ao_l: np.ndarray
    ao_m: np.ndarray
    ao_shell: np.ndarray
    ao_labels: Tuple[str, ...]

    @property
    def nao(self) -> int:
        return int(self.ao_atom.size)

    @property
    def natm(self) -> int:
        return len(self.atom_symbols)

    @property
    def max_l(self) -> int:
        return int(self.ao_l.max()) if self.nao else 0

    def atom_indices(self, atom: int) -> np.ndarray:
        """AO indices belonging to ``atom``."""
        return np.nonzero(self.ao_atom == atom)[0]

    def atom_label(self, atom: int) -> str:
        """``"2 Cl"`` — index first, so a table sorts in input order and two Cl are distinct.

        ⚠ **1-based in output** (user decision: the quantum-chemistry convention, matching
        the ``"Cl2"`` / atom-number addressing of per-atom bases and configurations); the
        ``atom`` argument stays the internal 0-based index.
        """
        return "{} {}".format(atom + 1, self.atom_symbols[atom])

    def ao_full_label(self, mu: int) -> str:
        """``"2 Cl 3px"`` — the atom and the AO label together."""
        return "{} {}".format(self.atom_label(int(self.ao_atom[mu])), self.ao_labels[mu])

    def group_by_ao_type(self) -> Dict[Tuple[int, str], np.ndarray]:
        """``{(atom, label): ao indices}`` in AO order.

        The grouping key for the reduced-AO population table: every AO with the same
        label on the same atom is one row. Distinct contractions of the same shell carry
        distinct labels (``"2s"``, ``"3s"``, ...), so this does **not** merge them.
        """
        groups: Dict[Tuple[int, str], List[int]] = {}
        for mu in range(self.nao):
            groups.setdefault((int(self.ao_atom[mu]), self.ao_labels[mu]), []).append(mu)
        return {k: np.asarray(v, dtype=int) for k, v in groups.items()}

    def high_l_mask(self, max_l: int = MOLDEN_MAX_L) -> np.ndarray:
        """Boolean mask of AOs above ``max_l`` — what a molden writer would have to drop."""
        return self.ao_l > max_l

    def __repr__(self) -> str:
        return "AOLayout(natm={}, nao={}, nshell={}, max_l={})".format(
            self.natm, self.nao, len(self.shells), ANGULAR[self.max_l])


def build_layout(atom_symbols: Sequence[str], atom_charges: Sequence[float],
                 coords_bohr: np.ndarray, shells: Sequence[Shell],
                 ao_labels: Sequence[str]) -> AOLayout:
    """Assemble an :class:`AOLayout` from shells, deriving the per-AO arrays.

    The AO arrays are *derived* rather than passed in, so they cannot disagree with the shell
    list — which is the thing the molden writer and the population analysis must agree on.
    """
    ao_atom: List[int] = []
    ao_l: List[int] = []
    ao_m: List[int] = []
    ao_shell: List[int] = []
    for ish, sh in enumerate(shells):
        ao_atom.extend([sh.atom] * sh.nao)
        ao_l.extend([sh.l] * sh.nao)
        ao_m.extend(shell_m_values(sh.l))
        ao_shell.extend([ish] * sh.nao)
    nao = len(ao_atom)
    if len(ao_labels) != nao:
        raise ValueError("{} AO labels for {} basis functions implied by the shells"
                         .format(len(ao_labels), nao))
    return AOLayout(
        atom_symbols=tuple(atom_symbols),
        atom_charges=np.asarray(atom_charges, dtype=float),
        coords_bohr=np.ascontiguousarray(coords_bohr, dtype=float),
        shells=tuple(shells),
        ao_atom=np.asarray(ao_atom, dtype=int),
        ao_l=np.asarray(ao_l, dtype=int),
        ao_m=np.asarray(ao_m, dtype=int),
        ao_shell=np.asarray(ao_shell, dtype=int),
        ao_labels=tuple(ao_labels),
    )


__all__ = ["ANGULAR", "MOLDEN_MAX_L", "AOLayout", "Shell", "build_layout",
           "molden_ao_order", "molden_m_order", "shell_m_values"]
