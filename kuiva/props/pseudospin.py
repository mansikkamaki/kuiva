"""Pseudospin assignment and the Ouluspin export file.

Kuiva's multi-site deliverable. The local-multiplet model (built by
:mod:`kuiva.dmrg.manifold`) gives, per magnetic site, a ``d_k``-dimensional multiplet
space; this module names it a **pseudospin** ``S_k = (d_k - 1)/2``, fixes a
self-consistent ``M`` labelling inside each site space, and writes the plain-text file the
external **Ouluspin** code consumes: the ordered pseudospin product basis
``|S,M> (x) |S',M'> (x) ...``, the operator matrices (``H_eff``, ``mu_x``, ``mu_y``,
``mu_z``) on the model space in that basis, and the unitary mapping the ab initio states
(eigenvectors of ``H_eff``) to it. Everything downstream — ITO decomposition, exchange
Hamiltonian fitting, crystal-field analysis — is Ouluspin's job, out of scope here.

The ``M`` convention (self-consistent and stated; phases stay arbitrary)
-------------------------------------------------------------------------------
Within one site space the basis is ordered by the eigenvalue of the site-projected moment
component along a **stated axis**: descending ``<mu . axis>`` is labelled ``M = -S`` up
to ``M = +S`` — the Abragam–Bleaney sign convention ``mu = -g mu_B S~`` (positive ``g``)
makes the largest-``M`` state the one with the most negative moment projection. ⚠ The
storage order — per site ``M = -S .. +S`` ascending, site 0 slowest — **is OuluSpin's
``PseudoSpinBasis`` lexicographic order**, deliberately: the file exists to be read by
OuluSpin, and matching its convention removes a permutation and the mistakes that ride on
one. The axis defaults to the site's **principal magnetic axis**: the eigenvector of the
largest principal value of ``M_ij = Tr(mu_i mu_j)`` over the site space
(Chibotaru–Ungur), with the sign fixed by making its largest component positive; a
``common_axis`` (one axis for every site — a vector, or ``"ground-doublet"`` for the
principal magnetic axis of the lowest doublet of ``H_eff``) makes the labelling globally
consistent, which is what a single-quantization-axis consumer wants, and
``rotate_frame=True`` additionally re-expresses every Cartesian moment component in the
principal triad of that doublet (z = quantization axis — OuluSpin's operator frame
convention), with the applied rotation recorded in the file. For an easy-plane site the
axis within the degenerate plane is arbitrary and a ``WARNING`` says so — the labelling
stays self-consistent, which is all the export requires. ⚠ Per-state **phases are never
canonicalized**: a phase convention is Ouluspin's job (its ab initio route already
owns a time-reversal-proper phase fixing, which applies to these matrices unchanged), and
every validation of this file's content goes through the phase-invariant reductions of
:mod:`kuiva.props.multiplet` — degeneracy patterns, relative energies, and
``Tr_block(mu_i mu_j)`` with its principal g values. No test may compare an element.

What this module refuses
------------------------
* A site multiplet space that mixes particle-number sectors: ``|S, M>`` presumes a
  multiplet, and a charge-mixed space is not one (the refusal names the knob — the
  multiplet rule that produced the space).
* Nothing else is second-guessed: ``H_eff`` and ``mu`` are taken as given, because the
  gap discipline that makes them trustworthy lives where they are built.

The file (the format is **confirmed against what OuluSpin reads**)
----------------------------------------------------------------------------------
Line oriented, ``#`` comments, ``[SECTION]`` markers, ``i j Re Im`` element records —
deliberately the same dull shape as the property dump, and versioned the same way:
``format_version`` is bumped when the *meaning* of a stored field changes, and
:func:`read_pseudospin` refuses an unknown version rather than guessing. ⚠ Unlike the property
dump, **``H`` here is NOT diagonal**: it is the effective Hamiltonian over the pseudospin
*product* basis; ``[ENERGIES]`` lists its eigenvalues and ``[MATRIX U]`` the diagonalizing
unitary (columns = ab initio states over product-basis rows). The header carries the
provenance dict passed in — once the ab initio route feeds this file, that is where the
screening and decoupling records land (the standing provenance obligation transfers
here too), and a write with *empty* provenance warns.

The hyperfine field, when nuclei were selected
----------------------------------------------
With ``hyperfine=`` at ingestion the file additionally carries, per treated nucleus, the three
components of the **hyperfine field operator** ``T_{k,u}`` over the same pseudospin product
basis as ``mu``, plus the ``[NUCLEI]`` table of what each nucleus is — the *same* section and
the *same* block names the property dump writes, read and written through
:mod:`kuiva.props.dump`'s one implementation of them, so a consumer meets one vocabulary
whichever file it opened. On the electron-nuclear product space the interaction is

    H_hf = sum_k g_N(k) sum_u T_{k,u} (x) I_{k,u}        [Eh]

and everything in that formula except the matrices of ``T`` is nuclear-spin algebra belonging
to OuluSpin. Four consequences, all decisions:

* ⚠ **Kuiva never forms the electron-nuclear product space.** Writing the electronic matrices
  and a nuclear table separately is what lets the isotope, or the subset of nuclei, be changed
  without re-running the electronic calculation. The product dimension
  ``D * prod_k (2 I_k + 1)`` is therefore **reported** — in the header and, above
  :data:`PRODUCT_DIM_WARN`, as a ``WARNING`` — and never refused: it is not Kuiva's allocation.
* **There are no site-projected ``T`` matrices, unlike ``mu``.** OuluSpin consumes none, and
  the rule for this file is not to widen a contract without a consumer; a nucleus feels every
  electronic site (transferred hyperfine), so a per-site table would also invite being read as
  "this site's coupling" when what the interaction is made of is the sum. The export therefore
  passes ``site_local=`` naming the moments alone, which is a saving of one model-space
  contraction per nucleus per site and — the site spaces here all being charge pure — loses no
  information: the per-site parts of a one-electron operator sum back to the whole.
* ⚠ **``T`` is time odd, exactly as ``L`` and ``S`` are**, so a Kramers-paired inactive set
  contributes exactly zero to it; the trace is computed and checked rather than assumed, and
  added to the model operator as a multiple of the identity, as ``mu``'s is.
* ⚠ **The isotropic part is only as good as the active space.** The contact mechanism comes
  from core-s spin polarization, which a valence CAS does not carry. The front end warns at
  the point of selection and the active space travels in this file's provenance.

**What OuluSpin consumes, and therefore what this file is a contract for** (user
decision): the Hamiltonian, the magnetic-moment operators, the pseudospin transformation and
the basis — ``[MATRIX H]``, ``[MATRIX mu_*]``, ``[SITE_MATRIX k mu_*]``, ``[MATRIX U]``,
``[SITES]`` and ``[BASIS]`` — and, when nuclei were selected, ``[NUCLEI]`` and
``[MATRIX T_<label>_*]``. Two things follow, and both are decisions rather than
oversights:

* ⚠ **Spin operator matrices are deliberately NOT written.** OuluSpin does not use them, so
  adding them would widen a contract for no consumer. Anything that needs `S` gets it from
  the moments and the pseudospin labelling.
* ⚠ **``energy_shift`` is Kuiva-side provenance, not part of the interface.** OuluSpin
  applies its own energy shift, so nothing downstream reads this field; it exists so a
  reader can reconstruct absolute totals (``e_core``) if it wants them, and
  ``[ENERGIES]`` states that it is *not* already included. Do not build a convention on it.

**Portability:** orchestration — formatting and a few small congruences. Never a
port candidate.

References
----------
* Pseudospin Hamiltonians and principal magnetic axes from ab initio states:
  L. F. Chibotaru, L. Ungur, J. Chem. Phys. 137, 064112 (2012), doi:10.1063/1.4739763.
* Pseudospin conventions: A. Abragam, B. Bleaney, "Electron Paramagnetic Resonance of
  Transition Ions", Clarendon Press, Oxford (1970).
* Effective Hamiltonians on model spaces (what ``H_eff``/``U`` realise): C. Bloch, Nucl.
  Phys. 6, 329 (1958), doi:10.1016/0029-5582(58)90116-0; J. des Cloizeaux, Nucl. Phys.
  20, 321 (1960), doi:10.1016/0029-5582(60)90177-2.
* Hyperfine coupling from ab initio spin-orbit states, and the first-order ``A A^T``
  construction the reported ``|A|`` values are: K. Sharkas, B. Pritchard, J. Autschbach,
  J. Chem. Theory Comput. 11, 538 (2015), doi:10.1021/ct500988h; L. Birnoschi, N. F. Chilton,
  J. Chem. Theory Comput. 18, 4719 (2022), doi:10.1021/acs.jctc.2c00257.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util.logging import get_logger
# ⚠ The [NUCLEI] table has ONE writer and ONE parser, and they live in the property dump
# because that is where the section was born. Both formatted products emit the same bytes for
# the same nucleus; a second implementation here would pass every test and still be a second
# vocabulary in a file format that is a contract with an external program.
from .dump import (HYPERFINE_UNIT, hyperfine_from_file, hyperfine_header, nucleus_line,
                   parse_nucleus_line)
from .multiplet import (HARTREE_TO_CM, Multiplet, analyse_spectrum,
                        block_collinearity, block_hyperfine_tensor, block_moment_tensor,
                        multiplet_g_values, multiplet_hyperfine_values)

log = get_logger(__name__)

#: Bumped when the *meaning* of anything already in the file changes (meaning changes bump it; additions do not).
FORMAT_VERSION = 1

#: Relative separation of the top two principal values of the site moment tensor below
#: which the principal axis is ambiguous (easy-plane site) and the labelling axis is an
#: arbitrary in-plane choice — warned about, never silently resolved.
AXIS_DEGENERACY_RTOL = 1.0e-6

#: Electron-nuclear product dimension above which the file *says so* in a ``WARNING``. ⚠ It is
#: a courtesy and never a refusal: Kuiva does not form that space, so the allocation is the
#: consumer's, and refusing on someone else's behalf would only teach a user to raise a limit
#: blindly. The number is where a consumer's own arithmetic starts to hurt — one dense
#: ``complex128`` operator of this dimension is 0.27 GB, and a hyperfine problem carries
#: ``1 + 3 + 3 N_nuclei`` of them.
PRODUCT_DIM_WARN = 4096


def _kuiva_version() -> str:
    """The running code's version, for the file header."""
    from .. import __version__
    return str(__version__)

_ELEMENT_FMT = "{:6d} {:6d}  {:+.16e} {:+.16e}\n"


# --- assignment -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PseudospinSite:
    """One site's pseudospin: dimension, labelling axis, and its moment in the M basis.

    ``moment`` is the site-projected moment ``(3, d, d)`` **in the pseudospin (M-ordered)
    basis**; ``g_values`` are the principal g values of the whole site multiplet — the
    phase-invariant fingerprint a reader compares, since the matrix elements themselves
    carry arbitrary phases.
    """

    index: int
    twice_s: int
    axis: np.ndarray
    axis_choice: str
    moment: np.ndarray
    n_electrons: Optional[int] = None
    orbitals: Tuple[int, ...] = ()
    g_values: Tuple[float, ...] = ()

    @property
    def dim(self) -> int:
        return self.twice_s + 1

    @property
    def twice_m(self) -> Tuple[int, ...]:
        """``2M`` per basis state, in storage order: ``-2S, -2S+2, ..., +2S``.

        Ascending ``M`` — OuluSpin's ``PseudoSpinBasis`` local order (module docstring).
        """
        return tuple(-self.twice_s + 2 * i for i in range(self.dim))


@dataclass(eq=False)
class PseudospinModel:
    """The export deliverable, in memory: everything the Ouluspin file contains.

    ``h`` and ``mu`` are over the pseudospin **product** basis (site 0 slowest, C order;
    within a site ``M = -S`` first, ascending); ``energies``/``unitary`` its eigen-decomposition,
    ``unitary[:, i]`` the i-th ab initio state over product-basis rows. Phases arbitrary
    throughout.

    ``hyperfine`` maps a nucleus's **atom label** to its ``(3, D, D)`` field operator in the
    same basis and the same units the property dump writes it in, with ``hyperfine_nuclei``
    the ``[NUCLEI]`` table saying what each nucleus is. ⚠ It is ``None`` and never ``{}``
    when no nuclei were selected: a missing hyperfine operator and a small one look identical
    in every number, and only the first of the two is a statement about the *calculation*.
    """

    sites: Tuple[PseudospinSite, ...]
    h: np.ndarray
    mu: np.ndarray
    energies: np.ndarray
    unitary: np.ndarray
    energy_shift: float = 0.0
    #: The frame the Cartesian moment components are expressed in, and the rotation FROM
    #: the input (ab initio) frame TO it — the identity unless ``rotate_frame`` was used.
    frame: str = "input frame"
    frame_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    provenance: Dict[str, object] = field(default_factory=dict)
    comments: Tuple[str, ...] = ()
    #: ``{atom label: (3, D, D)}`` [Eh per nuclear magneton], or ``None`` — never ``{}``.
    hyperfine: Optional[Dict[str, np.ndarray]] = None
    #: The ``[NUCLEI]`` rows, in the order the ``T`` matrices are written and in the order the
    #: consumer is to build its nuclear sites.
    hyperfine_nuclei: Tuple[Dict[str, object], ...] = ()
    #: The container half of the front end's hyperfine provenance (unit, treatment, decoupling,
    #: nuclear model, ``g_e``) — what the ``hyperfine_*`` header keys are made of.
    hyperfine_record: Dict[str, object] = field(default_factory=dict)

    @property
    def model_dim(self) -> int:
        return int(self.h.shape[0])

    @property
    def dims(self) -> Tuple[int, ...]:
        return tuple(s.dim for s in self.sites)

    @property
    def has_hyperfine(self) -> bool:
        return bool(self.hyperfine)

    @property
    def hyperfine_labels(self) -> Tuple[str, ...]:
        """The treated nuclei's atom labels, in ``[NUCLEI]`` order."""
        return tuple(self.hyperfine) if self.hyperfine else ()

    def nucleus(self, label: str) -> Dict[str, object]:
        """One nucleus's record, refusing rather than guessing."""
        for record in self.hyperfine_nuclei:
            if str(record.get("atom_label")) == str(label):
                return dict(record)
        raise KeyError("no nucleus labelled {!r} in this model; it carries {}"
                       .format(label, ", ".join(self.hyperfine_labels) or "none"))

    def product_dim(self) -> int:
        """``D * prod_k (2 I_k + 1)`` — the electron-nuclear space the *consumer* builds.

        ⚠ Reported, never refused: Kuiva does not form this space, which is the whole point
        of storing electronic matrices and a nuclear table separately.
        """
        dim = self.model_dim
        for record in self.hyperfine_nuclei:
            dim *= int(record.get("twice_spin", 0)) + 1
        return dim

    def basis_labels(self) -> List[Tuple[int, ...]]:
        """Per product state, the ``2M`` value at each site (site 0 slowest, C order)."""
        labels = [()]
        for s in self.sites:
            labels = [lab + (tm,) for lab in labels for tm in s.twice_m]
        return labels

    def unitarity_error(self) -> float:
        u = self.unitary
        return float(np.max(np.abs(u.conj().T @ u - np.eye(u.shape[1]))))

    def mu_in_eigenbasis(self) -> np.ndarray:
        u = self.unitary
        return np.stack([u.conj().T @ m @ u for m in self.mu])

    def hyperfine_in_eigenbasis(self) -> Optional[Dict[str, np.ndarray]]:
        """``T`` per nucleus in the basis of the effective eigenstates, or ``None``.

        The same congruence :meth:`mu_in_eigenbasis` applies, and for the same reason: the
        invariants compare blocks of the *spectrum*, so both operators have to be in the
        basis the spectrum is indexed in.
        """
        if not self.hyperfine:
            return None
        u = self.unitary
        return {label: np.stack([u.conj().T @ t @ u for t in op])
                for label, op in self.hyperfine.items()}

    def analyse(self, tol_cm: float = 1.0,
                pseudo_doublet_tol_cm: Optional[float] = None) -> List[Multiplet]:
        """The phase-invariant reduction of the effective spectrum + moments.

        Carries the hyperfine field along when there is one; it moves no existing number,
        the blocking being the energies' decision alone.
        """
        return analyse_spectrum(self.energies, self.mu_in_eigenbasis(), tol_cm=tol_cm,
                                pseudo_doublet_tol_cm=pseudo_doublet_tol_cm,
                                hyperfine=self.hyperfine_in_eigenbasis())

    @classmethod
    def from_file(cls, path) -> "PseudospinModel":
        """Rebuild this model from a file :func:`write_pseudospin` wrote.

        The sibling of :meth:`kuiva.props.dump.PropertyMatrices.from_dump`, and there for the
        same reason: the phases in the file are arbitrary, so two stored exports can only be
        compared through :meth:`analyse`, and that needs the object rather than
        :func:`read_pseudospin`'s dictionary.

        ⚠ **``g_values`` are recomputed from the site moments rather than read.** They are a
        *reduction* of what the file stores, and recomputing keeps them in step with the
        matrices by construction — a stored reduction is a second thing that can disagree with
        the first. Same reasoning as the axes not being written in the first place.

        ⚠ **The hyperfine half is rebuilt from the file's own two sections**, the ``[NUCLEI]``
        table and the ``hyperfine_*`` header keys, never from the provenance JSON: a consumer
        that parses no JSON must still get the nuclei, and this method reads what that consumer
        reads. A table naming a nucleus whose matrices are absent is refused.
        """
        raw = read_pseudospin(path)
        header, matrices, site_matrices = raw["header"], raw["matrices"], raw["site_matrices"]

        def stack(source, prefix, key=lambda a: a):
            return np.ascontiguousarray(np.stack([source[key("{}_{}".format(prefix, a))]
                                                  for a in "xyz"]))

        sites = []
        for entry in raw["sites"]:
            i = int(entry["index"])
            moment = np.ascontiguousarray(np.stack([site_matrices[(i, "mu_" + a)]
                                                    for a in "xyz"]))
            twice_s = int(entry["twice_s"])
            m_tensor = block_moment_tensor(moment, 0, twice_s + 1)
            sites.append(PseudospinSite(
                index=i, twice_s=twice_s,
                axis=np.asarray(entry["axis"], dtype=float),
                axis_choice=str(entry["axis_choice"]), moment=moment,
                n_electrons=entry["n_electrons"], orbitals=tuple(entry["orbitals"]),
                g_values=multiplet_g_values(m_tensor, twice_s + 1)))

        nuclei = tuple(raw.get("nuclei") or ())
        hyperfine, hf_record = hyperfine_from_file(path, header, matrices, nuclei)
        return cls(
            sites=tuple(sites), h=np.ascontiguousarray(matrices["H"]),
            mu=stack(matrices, "mu"),
            energies=np.asarray(raw["energies"], dtype=float),
            unitary=np.ascontiguousarray(matrices["U"]),
            energy_shift=float(header.get("energy_shift", 0.0)),
            frame=header.get("frame", "input frame"),
            frame_rotation=np.asarray(raw["frame_rotation"], dtype=float),
            provenance=dict(raw.get("provenance") or {}),
            hyperfine=hyperfine, hyperfine_nuclei=nuclei, hyperfine_record=hf_record)

    def report(self, logger=None) -> None:
        logger = logger or log
        out.subsection(logger, "pseudospin assignment")
        out.entries(logger, [
            ("sites", len(self.sites)),
            ("product basis", self.model_dim,
             "", " x ".join("2S+1={}".format(s.dim) for s in self.sites)),
            ("frame", self.frame),
            ("unitary error", self.unitarity_error(), "", "", "{:.2e}"),
            ("energy shift", self.energy_shift, "Eh", "added to stored energies",
             "{:+.10f}"),
        ])
        table = out.Table(logger, [
            out.Column("site", "{:d}", 5), out.Column("2S", "{:d}", 4),
            out.Column("axis", "{:s}", 24), out.Column("g_1", "{:.4f}", 9),
            out.Column("g_2", "{:.4f}", 9), out.Column("g_3", "{:.4f}", 9)])
        table.start()
        for s in self.sites:
            g = s.g_values if s.g_values else (float("nan"),) * 3
            table.row(s.index, s.twice_s,
                      "({:+.4f} {:+.4f} {:+.4f})".format(*s.axis), g[0], g[1], g[2])
        table.end("axis: the stated M-labelling axis ({})".format(
            ", ".join(sorted({s.axis_choice for s in self.sites}))))
        table = out.Table(logger, [
            out.col_count("block", 7), out.Column("states", "{:d}", 8),
            out.Column("E [cm^-1]", out.CM_FMT, 14),
            out.Column("g_1", "{:.4f}", 9), out.Column("g_2", "{:.4f}", 9),
            out.Column("g_3", "{:.4f}", 9)])
        table.start("effective spectrum (phase-invariant reduction)")
        blocks = self.analyse()
        for i, m in enumerate(blocks):
            g = m.g_values if m.g_values else (float("nan"),) * 3
            table.row(i, m.size, m.energy_cm, g[0], g[1], g[2])
        table.end("compare only through these invariants; phases are arbitrary "
                  "")
        self._report_hyperfine(logger, blocks)

    def _report_hyperfine(self, logger, blocks: List[Multiplet]) -> None:
        """``|A|`` per block per nucleus, and how it sits against ``g``.

        ⚠ **Everything here is a reduction of the stored operator, and none of it is stored.**
        The file carries ``T`` in Eh per nuclear magneton, isotope-independent; ``|A|`` below is
        that reduction evaluated at the isotope that was *requested*, a report quantity in
        exactly the sense the principal g values are — never a fitted tensor and never a spin
        Hamiltonian. The last column is the mixed invariant's one-number reading: ``+-1`` means
        ``T`` is proportional to ``mu`` on that block, and its **sign** is the relative sign of
        ``A`` and ``g``, which ``|A|`` throws away by being quadratic.
        """
        if not self.hyperfine:
            return
        out.subsection(logger, "hyperfine field operators")
        entries = [
            ("nuclei", len(self.hyperfine_nuclei),
             "", ", ".join("{} ({})".format(r.get("atom_label"), r.get("label"))
                           for r in self.hyperfine_nuclei)),
            ("operator unit", HYPERFINE_UNIT, "",
             str(self.hyperfine_record.get("operator", ""))),
            ("picture change",
             str(self.hyperfine_record.get("picture_change", "unrecorded")), "",
             "decoupling={}, nuclear model={}".format(
                 self.hyperfine_record.get("decoupling", "?"),
                 self.hyperfine_record.get("nuclear_model", "?"))),
            # ⚠ Reported, never refused: Kuiva does not form this space.
            ("electron-nuclear product dimension", self.product_dim(), "",
             "{} electronic x {}".format(
                 self.model_dim,
                 " x ".join("2I+1={}".format(int(r.get("twice_spin", 0)) + 1)
                            for r in self.hyperfine_nuclei))),
        ]
        out.entries(logger, entries)
        t_eigen = self.hyperfine_in_eigenbasis() or {}
        mu_eigen = self.mu_in_eigenbasis()
        table = out.Table(logger, [
            out.col_count("block", 7), out.Column("states", "{:d}", 8),
            out.Column("E [cm^-1]", out.CM_FMT, 14),
            out.Column("nucleus", "{}", 10, align="<"),
            out.Column("isotope", "{}", 9, align="<"),
            out.Column("|A_1| [MHz]", "{:.4g}", 13),
            out.Column("|A_2| [MHz]", "{:.4g}", 13),
            out.Column("|A_3| [MHz]", "{:.4g}", 13),
            out.Column("A.g", "{:+.4f}", 9)])
        table.start()
        for i, m in enumerate(blocks):
            for label in self.hyperfine_labels:
                record = self.nucleus(label)
                tensor = None if m.hyperfine is None else m.hyperfine.get(label)
                if tensor is None:
                    continue
                a = multiplet_hyperfine_values(tensor, m.size, float(record.get("g", 0.0)))
                # ⚠ `nan`, never 0, for a block that carries no moment: a size-1 block has no
                # A tensor for the same reason it has no g, and the two must not print alike.
                a3 = a if a else (float("nan"),) * 3
                table.row(i, m.size, m.energy_cm, label, str(record.get("label", "?")),
                          a3[0], a3[1], a3[2],
                          block_collinearity(mu_eigen, t_eigen[label], m.start, m.size))
        table.end("|A| are principal values of 3 g_N^2 Tr_block(T_i T_j)/[J(J+1)(2J+1)] at the "
                  "isotope named -- magnitudes only, since the reduction is quadratic. A.g is "
                  "Tr_block(mu.T) normalized: +-1 means T is proportional to mu on the block. "
                  "The stored operator is T itself, in " + HYPERFINE_UNIT + ", so the consumer "
                  "changes the isotope without re-running this calculation ")

    def write(self, path, **kwargs) -> Path:
        return write_pseudospin(path, self, **kwargs)


def _proper_triad(evecs: np.ndarray) -> np.ndarray:
    """Rows x, y, z from ascending-eigenvalue eigenvectors, sign-fixed and proper.

    Each axis gets the library sign convention (largest component positive); the x row is
    flipped if needed so the triad is a proper rotation — the same resolution OuluSpin's
    ab initio route applies, and for the same reason (principal axes are directionless).
    """
    rows = []
    for j in range(3):
        v = np.real(evecs[:, j]).copy()
        i = int(np.argmax(np.abs(v)))
        if v[i] < 0.0:
            v = -v
        rows.append(v / np.linalg.norm(v))
    triad = np.array(rows)
    if np.linalg.det(triad) < 0.0:
        triad[0] = -triad[0]
    return triad


def _ground_doublet_triad(h_eff: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """The principal magnetic triad (rows x, y, z; z = largest principal value) of the
    lowest doublet of ``H_eff`` — the quantization frame a single-axis consumer
    (OuluSpin) chooses by default."""
    if h_eff.shape[0] < 2:
        raise ValueError("common_axis='ground-doublet' needs at least two model states")
    _, vecs = np.linalg.eigh(0.5 * (h_eff + h_eff.conj().T))
    blk = vecs[:, :2]
    mu_blk = np.stack([blk.conj().T @ m @ blk for m in mu])
    m_tensor = block_moment_tensor(mu_blk, 0, 2)
    evals, evecs = np.linalg.eigh(m_tensor)
    scale = float(np.max(np.abs(evals)))
    if scale <= 0.0:
        raise ValueError("the ground doublet carries no magnetic moment; a common "
                         "quantization axis cannot be derived from it — give one")
    if evals[-1] - evals[-2] <= AXIS_DEGENERACY_RTOL * scale:
        log.warning("the ground doublet's top two principal moment values are "
                    "degenerate; the common quantization axis is an arbitrary choice "
                    "within the degenerate plane (self-consistent, and recorded)")
    return _proper_triad(evecs)


def _site_axis(moment: np.ndarray, given: Optional[Sequence[float]],
               index: int) -> Tuple[np.ndarray, str]:
    """The M-labelling axis: given, or the principal magnetic axis (module docstring)."""
    if given is not None:
        axis = np.asarray(given, dtype=float).ravel()
        n = float(np.linalg.norm(axis))
        if axis.shape != (3,) or n <= 0.0:
            raise ValueError("site {}: an M-labelling axis must be a nonzero 3-vector, "
                             "got {!r}".format(index, given))
        return axis / n, "given"
    d = moment.shape[1]
    m_tensor = block_moment_tensor(moment, 0, d)
    evals, evecs = np.linalg.eigh(m_tensor)
    scale = float(np.max(np.abs(evals)))
    if scale <= 0.0:
        log.warning("site %d carries no magnetic moment; the M-labelling axis defaults "
                    "to z and the labelling is arbitrary", index)
        return np.array([0.0, 0.0, 1.0]), "default z (zero moment)"
    if evals[-1] - evals[-2] <= AXIS_DEGENERACY_RTOL * scale:
        log.warning("site %d: the top two principal values of the moment tensor are "
                    "degenerate (an isotropic or easy-plane site); the M-labelling axis "
                    "is an arbitrary choice within the degenerate plane. The labelling "
                    "stays self-consistent, which is what the file requires ",
                    index)
    axis = evecs[:, -1]
    j = int(np.argmax(np.abs(axis)))
    if axis[j] < 0.0:
        axis = -axis
    return axis, "principal magnetic axis"


def _checked_hyperfine(hyperfine: Optional[Mapping[str, np.ndarray]],
                       nuclei: Sequence[Dict[str, object]],
                       d_model: int) -> Dict[str, np.ndarray]:
    """The hyperfine operators as ``complex128``, with the two halves checked against each other.

    ⚠ Matrices and table are refused apart rather than reconciled, in both directions: matrices
    without a row name no nucleus and state no ``I``, and a row without matrices describes a
    coupling the file cannot supply. A ``T`` matched to the wrong nucleus is Hermitian,
    plausible and wrong, which is the failure this exists to make impossible.
    """
    if not hyperfine:
        if nuclei:
            raise ValueError(
                "a nuclear table of {} row(s) was given with no hyperfine matrices; the table "
                "alone describes a coupling the file cannot supply"
                .format(len(nuclei)))
        return {}
    listed = [str(r.get("atom_label")) for r in nuclei]
    if sorted(listed) != sorted(str(k) for k in hyperfine):
        raise ValueError(
            "the hyperfine matrices are keyed {} and the nuclear table lists {}; the two "
            "halves of the contract must name the same nuclei, in the order the matrices are "
            "to be written".format(sorted(str(k) for k in hyperfine), listed))
    out_ops: Dict[str, np.ndarray] = {}
    for label in listed:                       # [NUCLEI] order decides the written order
        op = np.asarray(hyperfine[label], dtype=np.complex128)
        if op.shape != (3, d_model, d_model):
            raise ValueError("the hyperfine field of {0} must be (3, {1}, {1}), got {2}"
                             .format(label, d_model, op.shape))
        out_ops[label] = op
    return out_ops


def assign_pseudospin(h_eff: np.ndarray, mu: np.ndarray, site_dims: Sequence[int],
                      site_moments: Sequence[np.ndarray], *,
                      axes: Optional[Sequence[Optional[Sequence[float]]]] = None,
                      common_axis=None, rotate_frame: bool = False,
                      site_electrons: Optional[Sequence[Optional[int]]] = None,
                      orbitals: Optional[Sequence[Sequence[int]]] = None,
                      energy_shift: float = 0.0,
                      hyperfine: Optional[Mapping[str, np.ndarray]] = None,
                      hyperfine_nuclei: Sequence[Dict[str, object]] = (),
                      hyperfine_record: Optional[Dict[str, object]] = None,
                      provenance: Optional[Dict[str, object]] = None,
                      comments: Sequence[str] = ()) -> PseudospinModel:
    """Assign pseudospin labels and rotate the model into the pseudospin basis.

    Parameters
    ----------
    h_eff, mu : ``(D, D)`` and ``(3, D, D)`` over the model product basis, site 0 slowest
        (:mod:`kuiva.dmrg.manifold`'s convention — but only plain arrays cross this
        boundary, deliberately: this module imports nothing from ``kuiva.dmrg``).
    site_dims : the per-site multiplet dimensions ``d_k``; ``prod d_k`` must equal ``D``.
    site_moments : per site, the site-projected moment ``(3, d_k, d_k)`` in the *model*
        site basis — what orders the M labels.
    axes : optional per-site labelling axes; ``None`` entries take the site's principal
        magnetic axis.
    common_axis : one labelling axis for **every** site — a 3-vector, or
        ``"ground-doublet"`` for the principal magnetic axis of the lowest doublet of
        ``H_eff`` (the default quantization axis of a single-axis consumer such as
        OuluSpin). Mutually exclusive with ``axes``.
    rotate_frame : with ``common_axis``, additionally re-express every Cartesian moment
        component in the principal triad of that choice (z = the quantization axis) —
        the frame OuluSpin's pseudospin operators live in. The applied rotation is
        recorded on the model and in the file; the default leaves everything in the
        input (ab initio) frame.
    hyperfine : ``{atom label: (3, D, D)}`` over the same model product basis as ``mu``,
        in Eh per nuclear magneton. ⚠ It rides through **both** transformations ``mu``
        does — the frame rotation, as a Cartesian vector, and the ``t^dag O t``
        congruence into the M-ordered basis — because the file states one frame and one
        basis for every operator in it, and an operator that missed either would be
        Hermitian, plausible and expressed in a frame the header denies.
    hyperfine_nuclei, hyperfine_record : the ``[NUCLEI]`` rows and the container record,
        as :func:`kuiva.props.dump.hyperfine_table` splits them. The labels must be exactly
        the keys of ``hyperfine``: matrices with no table name no nucleus and state no ``I``,
        and a table with no matrices describes a coupling the file cannot supply.
    """
    dims = tuple(int(d) for d in site_dims)
    h = np.asarray(h_eff, dtype=np.complex128)
    mu = np.asarray(mu, dtype=np.complex128)
    d_model = int(np.prod(dims))
    if h.shape != (d_model, d_model):
        raise ValueError("H_eff is {} but the site dimensions {} give a model dimension "
                         "of {}".format(h.shape, dims, d_model))
    if mu.shape != (3, d_model, d_model):
        raise ValueError("mu must be (3, {0}, {0}), got {1}".format(d_model, mu.shape))
    if len(site_moments) != len(dims):
        raise ValueError("{} site moments for {} sites".format(len(site_moments),
                                                               len(dims)))
    site_moments = [np.asarray(m, dtype=np.complex128) for m in site_moments]
    hf = _checked_hyperfine(hyperfine, hyperfine_nuclei, d_model)
    if common_axis is not None and axes is not None:
        raise ValueError("give either per-site axes or one common_axis, not both")
    if rotate_frame and common_axis is None:
        raise ValueError("rotate_frame needs a common_axis: rotating the components "
                         "while labelling along per-site axes would record a frame the "
                         "labels do not use")

    frame = "input frame"
    frame_rotation = np.eye(3)
    choice_override: Optional[str] = None
    if common_axis is not None:
        if isinstance(common_axis, str):
            if common_axis != "ground-doublet":
                raise ValueError("common_axis must be a 3-vector or 'ground-doublet', "
                                 "got {!r}".format(common_axis))
            triad = _ground_doublet_triad(h, mu)
            choice_override = "common (ground doublet)"
        else:
            axis = np.asarray(common_axis, dtype=float).ravel()
            n = float(np.linalg.norm(axis))
            if axis.shape != (3,) or n <= 0.0:
                raise ValueError("common_axis must be a nonzero 3-vector, got {!r}"
                                 .format(common_axis))
            axis = axis / n
            # complete a deterministic proper triad with z = the given axis
            seed = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(seed, axis))) > 0.9:
                seed = np.array([0.0, 1.0, 0.0])
            x = seed - float(np.dot(seed, axis)) * axis
            x = x / np.linalg.norm(x)
            triad = np.array([x, np.cross(axis, x), axis])
            choice_override = "common (given)"
        if rotate_frame:
            mu = np.tensordot(triad, mu, axes=(1, 0))
            site_moments = [np.tensordot(triad, m, axes=(1, 0)) for m in site_moments]
            # ⚠ The hyperfine field is a Cartesian vector operator and rotates with the rest:
            # the header states ONE frame for the file, and a T left in the input frame would
            # still be Hermitian, still have the right invariant magnitudes, and sit at an
            # arbitrary angle to the mu the consumer pairs it with.
            hf = {label: np.tensordot(triad, op, axes=(1, 0)) for label, op in hf.items()}
            frame = "quantization-axis frame (z = common axis)"
            frame_rotation = triad
            axes = [(0.0, 0.0, 1.0)] * len(dims)
        else:
            axes = [triad[2]] * len(dims)
    axes = [None] * len(dims) if axes is None else list(axes)
    n_elec = [None] * len(dims) if site_electrons is None else list(site_electrons)
    orbs = [()] * len(dims) if orbitals is None else [tuple(o) for o in orbitals]

    sites: List[PseudospinSite] = []
    rotations: List[np.ndarray] = []
    for k, d in enumerate(dims):
        mo = site_moments[k]
        if mo.shape != (3, d, d):
            raise ValueError("site {} moment must be (3, {}, {}), got {}"
                             .format(k, d, d, mo.shape))
        axis, choice = _site_axis(mo, axes[k], k)
        if choice_override is not None:
            choice = choice_override
        if d == 1:
            r = np.eye(1, dtype=np.complex128)
        else:
            mu_ax = np.tensordot(axis, mo, axes=(0, 0))
            evals, r = np.linalg.eigh(0.5 * (mu_ax + mu_ax.conj().T))
            groups = np.nonzero(np.diff(evals) <= AXIS_DEGENERACY_RTOL
                                * max(float(np.max(np.abs(evals))), 1e-300))[0]
            if groups.size:
                log.warning("site %d: %d pairs of moment projections along the labelling "
                            "axis are degenerate; the M assignment inside them is an "
                            "arbitrary (self-consistent) choice", k, int(groups.size))
            # descending <mu . axis> = ascending M (module docstring: mu = -g mu_B S~,
            # and ascending M is OuluSpin's PseudoSpinBasis local order)
            r = np.ascontiguousarray(r[:, ::-1])
        rotated = np.stack([r.conj().T @ m @ r for m in mo])
        m_tensor = block_moment_tensor(rotated, 0, d)
        sites.append(PseudospinSite(index=k, twice_s=d - 1, axis=axis,
                                    axis_choice=choice, moment=rotated,
                                    n_electrons=n_elec[k], orbitals=orbs[k],
                                    g_values=multiplet_g_values(m_tensor, d)))
        rotations.append(r)

    t = np.eye(1, dtype=np.complex128)
    for r in rotations:
        t = np.kron(t, r)
    h_ps = t.conj().T @ h @ t
    mu_ps = np.stack([t.conj().T @ m @ t for m in mu])
    # the same congruence, because the file states one basis for every matrix in it
    hf_ps = {label: np.stack([t.conj().T @ op_u @ t for op_u in op])
             for label, op in hf.items()}
    energies, unitary = np.linalg.eigh(0.5 * (h_ps + h_ps.conj().T))

    return PseudospinModel(sites=tuple(sites), h=h_ps, mu=mu_ps, energies=energies,
                           unitary=unitary, energy_shift=float(energy_shift),
                           frame=frame, frame_rotation=frame_rotation,
                           provenance=dict(provenance or {}),
                           comments=tuple(comments),
                           # ⚠ `None`, never `{}`: "no nuclei were selected" and "the coupling
                           # is small" must not be the same object downstream.
                           hyperfine=hf_ps or None,
                           hyperfine_nuclei=tuple(dict(r) for r in hyperfine_nuclei),
                           hyperfine_record=dict(hyperfine_record or {}))


def pseudospin_from_model(model, *, moments: Sequence[str] = ("mu_x", "mu_y", "mu_z"),
                          axes=None, common_axis=None, rotate_frame: bool = False,
                          energy_shift: float = 0.0,
                          hyperfine: Optional[Mapping[str, Sequence[str]]] = None,
                          hyperfine_nuclei: Sequence[Dict[str, object]] = (),
                          hyperfine_record: Optional[Dict[str, object]] = None,
                          provenance: Optional[Dict[str, object]] = None,
                          comments: Sequence[str] = ()) -> PseudospinModel:
    """:func:`assign_pseudospin` from an ``EffectiveModel``-shaped object.

    Duck-typed on ``sites`` (each with ``dim``, ``charges``, ``orbitals``), ``operators``
    and ``site_operators`` — the same one-way-dependency idiom as
    ``ttno_from_cas_integrals``: :mod:`kuiva.props` never imports :mod:`kuiva.dmrg`.

    ``hyperfine`` maps a nucleus's atom label to the **three operator names** its Cartesian
    components were contracted under, so the naming convention stays with the caller that
    chose it and this function learns nothing about nuclei beyond the table it passes on.
    Only ``model.operators`` is read for them and never ``model.site_operators``: nothing
    consumes a site-projected hyperfine matrix, which is why the export tells
    :func:`kuiva.dmrg.manifold.effective_model` not to compute one at all.

    ⚠ Refuses a site whose multiplet space mixes particle-number sectors: ``|S, M>``
    presumes a multiplet, and a charge-mixed space is not one. The knob is the multiplet
    rule that produced the space.
    """
    for k, sp in enumerate(model.sites):
        counts = {qn.n if hasattr(qn, "n") else int(qn) for qn in sp.charges}
        if len(counts) != 1:
            raise ValueError(
                "site {} mixes particle-number sectors {} — a pseudospin |S, M> labels a "
                "multiplet, which a charge-mixed space is not. Tighten the multiplet "
                "rule so each site space sits in one N sector"
                .format(k, sorted(counts)))
    wanted = list(moments) + [n for names in (hyperfine or {}).values() for n in names]
    missing = [n for n in wanted if n not in model.operators]
    if missing:
        raise ValueError("the effective model carries no operator(s) {}; build it with "
                         "operators={{name: terms}} for the three moment components "
                         "(and for each nucleus's three hyperfine components)"
                         .format(missing))
    mu = np.stack([model.operators[n] for n in moments])
    site_moments = [np.stack([model.site_operators[n][k] for n in moments])
                    for k in range(len(model.sites))]
    hf = {label: np.stack([model.operators[n] for n in names])
          for label, names in (hyperfine or {}).items()}
    return assign_pseudospin(
        model.h_eff, mu, [sp.dim for sp in model.sites], site_moments, axes=axes,
        common_axis=common_axis, rotate_frame=rotate_frame,
        site_electrons=[sp.n_electrons for sp in model.sites],
        orbitals=[sp.orbitals for sp in model.sites], energy_shift=energy_shift,
        hyperfine=hf or None, hyperfine_nuclei=hyperfine_nuclei,
        hyperfine_record=hyperfine_record,
        provenance=provenance, comments=comments)


# --- the file -------------------------------------------------------------------------------

def write_pseudospin(path, model: PseudospinModel, *, title: str = "",
                     include_site_moments: bool = True) -> Path:
    """Write the OuluSpin file (module docstring: the format) and return its path."""
    path = Path(path)
    if not model.provenance:
        log.warning("the pseudospin file %s carries no Hamiltonian provenance; once the "
                    "ab initio route feeds this file, the screening and "
                    "decoupling records belong in it ", path.name)

    d = model.model_dim
    if model.has_hyperfine:
        # ⚠ The same standing obligation the property dump discharges, for the one operator
        # family whose treatment is not a choice: what was done (always the picture change),
        # and what no number in the file can show (the contact part needs a spin polarization
        # a valence active space does not carry).
        log.warning("%s carries HYPERFINE FIELD operators for %s. They are the "
                    "isotope-independent operator T in %s, always X2C picture-changed, with "
                    "the unperturbed transformation and no two-electron picture change; "
                    "H_hf = sum_k g_N(k) sum_u T_k_u (x) I_k_u over the product of this "
                    "model space with the nuclear spins, and the nuclear-spin algebra and "
                    "any A tensor belong to the consumer. The isotropic part comes from "
                    "core-s spin polarization, which a VALENCE active space does not carry",
                    path.name, ", ".join(model.hyperfine_labels), HYPERFINE_UNIT)
        product = model.product_dim()
        if product > PRODUCT_DIM_WARN:
            # Reported, never refused: Kuiva does not form this space (module docstring).
            log.warning("the electron-nuclear product space of %s is %d x %d for the listed "
                        "isotopes (%d electronic states x the nuclear multiplicities); one "
                        "dense complex operator of that size is %.1f GB and the consumer "
                        "builds %d of them. Kuiva does not form this space -- drop a nucleus, "
                        "or a lighter isotope, if that is too large",
                        path.name, product, product, d,
                        16.0 * product * product / 1024.0 ** 3,
                        4 + 3 * len(model.hyperfine_nuclei))

    header = [
        ("format", "KUIVA_PSEUDOSPIN"),
        ("format_version", str(FORMAT_VERSION)),
        # the provenance obligation: which Kuiva produced the numbers, beside which format they are in
        #. Adding a header key does not bump `format_version`.
        ("code_version", _kuiva_version()),
        ("n_sites", str(len(model.sites))),
        ("model_dim", str(d)),
        ("energy_unit", "Eh"),
        ("moment_unit", "mu_B"),
        ("energy_shift", "{:+.16e}".format(model.energy_shift)),
        ("hamiltonian_is_diagonal", "no"),
        ("basis_order", "site 0 slowest (C order); within a site M = -S .. +S ascending "
                        "(OuluSpin PseudoSpinBasis lexicographic order)"),
        ("m_convention", "descending <mu . axis> labelled M = -S .. +S"),
        ("frame", model.frame),
        ("phase_convention", "arbitrary (not canonicalized)"),
    ]
    if model.has_hyperfine:
        header.extend(hyperfine_header(
            model.hyperfine_record, len(model.hyperfine_nuclei),
            g_electron=float(model.hyperfine_record.get("g_electron", 0.0))))
        header.extend([
            ("nuclear_site_order",
             "nuclear sites follow the electronic sites, in [NUCLEI] order; within each, "
             "M_I = -I .. +I ascending (the [BASIS] order extended, not a second convention)"),
            # ⚠ Reported so the consumer knows what it is about to allocate, never refused:
            # Kuiva does not form the product space, which is why the isotope is the
            # consumer's to change.
            ("product_dim", str(model.product_dim())),
        ])

    lines: List[str] = []
    w = lines.append
    w("# Kuiva pseudospin model for OuluSpin.\n")
    if title:
        w("# {}\n".format(title))
    w("#\n")
    w("# H is the effective Hamiltonian over the pseudospin PRODUCT basis and is NOT\n"
      "# diagonal; [ENERGIES] lists its eigenvalues and [MATRIX U] the diagonalizing\n"
      "# unitary (columns = ab initio states over product-basis rows).\n")
    w("#\n")
    if model.has_hyperfine:
        w("# T_<k>_x/y/z are the HYPERFINE FIELD operator of nucleus <k> (its atom label),\n"
          "# in Eh per nuclear magneton, over the same product basis as mu. On the product\n"
          "# of this model space with the nuclear spins the interaction is\n"
          "#\n"
          "#     H_hf = sum_k g_N(k) sum_u T_<k>_u (x) I_<k>_u        [Eh]\n"
          "#\n"
          "# with g_N and I from the [NUCLEI] table below, whose order is the order the\n"
          "# nuclear sites are to be built in, after the electronic sites. The nuclear-spin\n"
          "# algebra, the Kronecker products and any A tensor belong to the consumer. The\n"
          "# operator is the ISOTOPE-INDEPENDENT field, so the isotope -- or the subset of\n"
          "# nuclei -- may be changed without re-running the electronic calculation, and the\n"
          "# product dimension in the header is reported for that reason rather than imposed.\n"
          "#\n"
          "# The X2C picture change IS applied to these operators, always, independently of\n"
          "# what the header says about mu: a bare hyperfine operator is wrong by a factor of\n"
          "# 4-10 wherever s character carries spin density. The transformation uses the\n"
          "# unperturbed X and R, no two-electron picture change is applied, and the nuclear\n"
          "# magnetization follows the nuclear charge model named in the header.\n"
          "#\n"
          "# WARNING: the isotropic (contact) part comes from core-s spin polarization, which\n"
          "# a VALENCE active space does not carry. For a 4f ion that is minor, the orbital\n"
          "# mechanism dominating; for s/d spin density, for ligand nuclei and for spin-only\n"
          "# ions it is qualitatively wrong. Judge these matrices against the active space in\n"
          "# the provenance below.\n")
        w("#\n")
    w("# WARNING: state phases are arbitrary and degenerate states mix arbitrarily.\n"
      "# Compare this file only through invariants: degeneracy patterns, relative\n"
      "# energies, and Tr_block(mu_i mu_j) with its principal g values.\n")
    w("#\n")
    for line in model.comments:
        w("# {}\n".format(line))
    if model.comments:
        w("#\n")

    w("[HEADER]\n")
    for key, value in header:
        w("{:32s} {}\n".format(key, value))
    w("[END]\n\n")

    w("[PROVENANCE]\n")
    w(json.dumps(model.provenance, sort_keys=True, indent=2))
    w("\n[END]\n\n")

    w("[FRAME]\n")
    w("# rotation FROM the input (ab initio) frame TO the frame of the stored\n"
      "# components; the identity when nothing was rotated. Rows x, y, z.\n")
    for row in np.asarray(model.frame_rotation, dtype=float):
        w("  {:+.14f} {:+.14f} {:+.14f}\n".format(*row))
    w("[END]\n\n")

    if model.has_hyperfine:
        w("# The nuclei the T matrices below belong to, in the order they are written and in\n"
          "# the order the nuclear sites of the product space are to be built: each nucleus a\n"
          "# pseudospin site of dimension 2I+1, with M_I = -I .. +I ascending, following the\n"
          "# electronic sites of [SITES]. Q is 'none' where no signed quadrupole moment is\n"
          "# tabulated -- refuse rather than substitute a magnitude, since the sign of Q is\n"
          "# the sign of every quadrupole splitting.\n")
        w("[NUCLEI]\n")
        w("# {:>3s} {:>5s}  {:<10s} {:<4s} {:>4s}  {:<10s} {:>4s}  {:>22s}  {:>16s}"
          "  {:>22s} {:>22s} {:>22s}  | source\n"
          .format("k", "atom", "label", "elem", "Z", "isotope", "2I", "g_N", "Q [barn]",
                  "x [bohr]", "y [bohr]", "z [bohr]"))
        for k, record in enumerate(model.hyperfine_nuclei):
            w(nucleus_line(k, record))
        w("[END]\n\n")

    w("[SITES]\n")
    w("# site   2S  dim   axis_x        axis_y        axis_z        axis_choice | N | orbitals\n")
    for s in model.sites:
        w("{:5d} {:4d} {:4d}  {:+.10f} {:+.10f} {:+.10f}  {} | {} | {}\n".format(
            s.index, s.twice_s, s.dim, s.axis[0], s.axis[1], s.axis[2],
            s.axis_choice.replace(" ", "_"),
            "?" if s.n_electrons is None else s.n_electrons,
            " ".join(str(x) for x in s.orbitals)))
    w("[END]\n\n")

    w("[BASIS]\n")
    w("# product basis state -> 2M at each site (site order as in [SITES])\n")
    for i, lab in enumerate(model.basis_labels()):
        w("{:6d}  {}\n".format(i, " ".join("{:+d}".format(x) for x in lab)))
    w("[END]\n\n")

    rel = (model.energies - model.energies.min()) * HARTREE_TO_CM
    w("[ENERGIES]\n")
    w("# eigenvalues of H. energy_shift is NOT included -- add it for absolute totals.\n"
      "# OuluSpin applies its own shift and does not read that field.\n")
    w("# index    energy [Eh]                relative [cm^-1]\n")
    for i, e in enumerate(model.energies):
        w("{:6d}  {:+.16e}  {:+.8e}\n".format(i, float(e), float(rel[i])))
    w("[END]\n\n")

    blocks: List[Tuple[str, np.ndarray, str, str]] = [
        ("H", model.h, "Eh", "effective Hamiltonian, pseudospin product basis")]
    for k, axis in enumerate("xyz"):
        blocks.append(("mu_" + axis, model.mu[k], "mu_B",
                       "magnetic moment, {}".format(axis)))
    # ⚠ Written whenever the reference ingested them and governed by no second switch: asking
    # for the nuclei at ingestion IS the request, and a file that computed the operators and
    # then did not write them would be the one thing a reader cannot recover from.
    for label in model.hyperfine_labels:
        for k, axis in enumerate("xyz"):
            blocks.append(("T_{}_{}".format(label, axis), model.hyperfine[label][k],
                           HYPERFINE_UNIT,
                           "hyperfine field of {}, {}; H_hf = g_N sum_u T_u (x) I_u"
                           .format(label, axis)))
    blocks.append(("U", model.unitary, "1",
                   "columns: ab initio states; rows: pseudospin product basis"))
    for name, mat, unit, note in blocks:
        a = np.asarray(mat, dtype=np.complex128)
        w("[MATRIX {}]\n".format(name))
        w("shape      {} {}\n".format(a.shape[0], a.shape[1]))
        w("unit       {}\n".format(unit))
        w("# {}\n".format(note))
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                w(_ELEMENT_FMT.format(i, j, float(a[i, j].real), float(a[i, j].imag)))
        w("[END]\n\n")

    if include_site_moments:
        for s in model.sites:
            for k, axis in enumerate("xyz"):
                a = s.moment[k]
                w("[SITE_MATRIX {} mu_{}]\n".format(s.index, axis))
                w("shape      {} {}\n".format(a.shape[0], a.shape[1]))
                w("unit       mu_B\n")
                w("# site-projected moment in the M-ordered site basis\n")
                for i in range(a.shape[0]):
                    for j in range(a.shape[1]):
                        w(_ELEMENT_FMT.format(i, j, float(a[i, j].real),
                                              float(a[i, j].imag)))
                w("[END]\n\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    # written whole, then moved into place: a file truncated by an interrupt parses, and
    # that is worse than no file (same discipline as the property dump and the checkpoints)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text("".join(lines))
    tmp.replace(path)
    out.blank(log)
    out.entry(log, "pseudospin model written to", str(path), "",
              "{} sites, model dimension {}".format(len(model.sites), d))
    return path


def read_pseudospin(path) -> Dict[str, object]:
    """Parse a file written by :func:`write_pseudospin` — the round-trip test.

    Returns ``{"header": {...}, "provenance": {...}, "frame_rotation": ndarray,
    "sites": [...], "nuclei": [...], "basis": ndarray, "energies": ndarray,
    "matrices": {name: ndarray}, "site_matrices": {(site, name): ndarray}}``. Refuses an
    unknown ``format_version`` rather than guessing.

    ``"nuclei"`` is the ``[NUCLEI]`` table when the file carries hyperfine operators and an
    empty list otherwise; its entries are keyed exactly as the property dump's are, so a
    consumer reads one vocabulary whichever of the two files it opened.
    """
    text = Path(path).read_text().splitlines()
    header: Dict[str, str] = {}
    provenance: Dict[str, object] = {}
    frame_rows: List[List[float]] = []
    sites: List[Dict[str, object]] = []
    nuclei: List[Dict[str, object]] = []
    basis: List[List[int]] = []
    energies: List[float] = []
    matrices: Dict[str, np.ndarray] = {}
    site_matrices: Dict[Tuple[int, str], np.ndarray] = {}

    section: Optional[str] = None
    buffer: List[str] = []
    current: Optional[np.ndarray] = None
    key: object = None
    for raw in text:
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("["):
            tag = line.strip()[1:-1]
            if tag == "END":
                if section == "PROVENANCE":
                    provenance = json.loads("\n".join(buffer))
                section, buffer, current, key = None, [], None, None
                continue
            parts = tag.split()
            section = parts[0]
            if section == "MATRIX":
                key = parts[1]
            elif section == "SITE_MATRIX":
                key = (int(parts[1]), parts[2])
            continue
        if section == "HEADER":
            k, _, v = line.strip().partition(" ")
            header[k] = v.strip()
        elif section == "PROVENANCE":
            buffer.append(line)
        elif section == "FRAME":
            frame_rows.append([float(x) for x in line.split()])
        elif section == "SITES":
            head, _, tail = line.partition("|")
            fields = head.split()
            n_str = tail.partition("|")[0].strip()
            orb_str = tail.partition("|")[2].strip()
            sites.append({
                "index": int(fields[0]), "twice_s": int(fields[1]),
                "dim": int(fields[2]),
                "axis": np.array([float(x) for x in fields[3:6]]),
                "axis_choice": fields[6].replace("_", " "),
                "n_electrons": None if n_str == "?" else int(n_str),
                "orbitals": tuple(int(x) for x in orb_str.split()) if orb_str else ()})
        elif section == "NUCLEI":
            nuclei.append(parse_nucleus_line(line))
        elif section == "BASIS":
            parts = line.split()
            basis.append([int(x) for x in parts[1:]])
        elif section == "ENERGIES":
            energies.append(float(line.split()[1]))
        elif section in ("MATRIX", "SITE_MATRIX"):
            parts = line.split()
            if parts[0] == "shape":
                current = np.zeros((int(parts[1]), int(parts[2])), dtype=np.complex128)
                (matrices if section == "MATRIX" else site_matrices)[key] = current
            elif parts[0] == "unit":
                continue
            elif current is None:
                raise ValueError("{}: an element line precedes the `shape` line of "
                                 "matrix {!r}".format(path, key))
            else:
                i, j = int(parts[0]), int(parts[1])
                current[i, j] = complex(float(parts[2]), float(parts[3]))

    version = int(header.get("format_version", -1))
    if version != FORMAT_VERSION:
        raise ValueError(
            "{} declares format_version {} and this parser knows version {}; refusing "
            "to guess (the version exists so a consumer can refuse rather than "
            "misinterpret)".format(path, version, FORMAT_VERSION))
    declared = int(header.get("n_hyperfine_nuclei", len(nuclei)))
    if declared != len(nuclei):
        raise ValueError(
            "{} declares {} hyperfine nuclei in its header and its [NUCLEI] table has {} rows. "
            "A T matrix matched to the wrong nucleus is Hermitian, plausible and wrong, so "
            "this is refused rather than reconciled.".format(path, declared, len(nuclei)))
    return {"header": header, "provenance": provenance,
            "frame_rotation": (np.array(frame_rows, dtype=float) if frame_rows
                               else np.eye(3)),
            "sites": sites, "nuclei": nuclei, "basis": np.array(basis, dtype=np.int64),
            "energies": np.array(energies, dtype=float), "matrices": matrices,
            "site_matrices": site_matrices}


__all__ = ["FORMAT_VERSION", "AXIS_DEGENERACY_RTOL", "PRODUCT_DIM_WARN", "PseudospinSite",
           "PseudospinModel",
           "assign_pseudospin", "pseudospin_from_model", "write_pseudospin",
           "read_pseudospin"]
