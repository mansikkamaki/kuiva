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

**What OuluSpin consumes, and therefore what this file is a contract for** (user
decision): the Hamiltonian, the magnetic-moment operators, the pseudospin transformation and
the basis — ``[MATRIX H]``, ``[MATRIX mu_*]``, ``[SITE_MATRIX k mu_*]``, ``[MATRIX U]``,
``[SITES]`` and ``[BASIS]``. Two things follow, and both are decisions rather than
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
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..util import output as out
from ..util.logging import get_logger
from .multiplet import (HARTREE_TO_CM, Multiplet, analyse_spectrum,
                        block_moment_tensor, multiplet_g_values)

log = get_logger(__name__)

#: Bumped when the *meaning* of anything already in the file changes (meaning changes bump it; additions do not).
FORMAT_VERSION = 1

#: Relative separation of the top two principal values of the site moment tensor below
#: which the principal axis is ambiguous (easy-plane site) and the labelling axis is an
#: arbitrary in-plane choice — warned about, never silently resolved.
AXIS_DEGENERACY_RTOL = 1.0e-6


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
    within a site ``M = +S`` first); ``energies``/``unitary`` its eigen-decomposition,
    ``unitary[:, i]`` the i-th ab initio state over product-basis rows. Phases arbitrary
    throughout.
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

    @property
    def model_dim(self) -> int:
        return int(self.h.shape[0])

    @property
    def dims(self) -> Tuple[int, ...]:
        return tuple(s.dim for s in self.sites)

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

    def analyse(self, tol_cm: float = 1.0) -> List[Multiplet]:
        """The phase-invariant reduction of the effective spectrum + moments."""
        return analyse_spectrum(self.energies, self.mu_in_eigenbasis(), tol_cm=tol_cm)

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
        for i, m in enumerate(self.analyse()):
            g = m.g_values if m.g_values else (float("nan"),) * 3
            table.row(i, m.size, m.energy_cm, g[0], g[1], g[2])
        table.end("compare only through these invariants; phases are arbitrary "
                  "")

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


def assign_pseudospin(h_eff: np.ndarray, mu: np.ndarray, site_dims: Sequence[int],
                      site_moments: Sequence[np.ndarray], *,
                      axes: Optional[Sequence[Optional[Sequence[float]]]] = None,
                      common_axis=None, rotate_frame: bool = False,
                      site_electrons: Optional[Sequence[Optional[int]]] = None,
                      orbitals: Optional[Sequence[Sequence[int]]] = None,
                      energy_shift: float = 0.0,
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
    energies, unitary = np.linalg.eigh(0.5 * (h_ps + h_ps.conj().T))

    return PseudospinModel(sites=tuple(sites), h=h_ps, mu=mu_ps, energies=energies,
                           unitary=unitary, energy_shift=float(energy_shift),
                           frame=frame, frame_rotation=frame_rotation,
                           provenance=dict(provenance or {}),
                           comments=tuple(comments))


def pseudospin_from_model(model, *, moments: Sequence[str] = ("mu_x", "mu_y", "mu_z"),
                          axes=None, common_axis=None, rotate_frame: bool = False,
                          energy_shift: float = 0.0,
                          provenance: Optional[Dict[str, object]] = None,
                          comments: Sequence[str] = ()) -> PseudospinModel:
    """:func:`assign_pseudospin` from an ``EffectiveModel``-shaped object.

    Duck-typed on ``sites`` (each with ``dim``, ``charges``, ``orbitals``), ``operators``
    and ``site_operators`` — the same one-way-dependency idiom as
    ``ttno_from_cas_integrals``: :mod:`kuiva.props` never imports :mod:`kuiva.dmrg`.

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
    missing = [n for n in moments if n not in model.operators]
    if missing:
        raise ValueError("the effective model carries no operator(s) {}; build it with "
                         "operators={{name: terms}} for the three moment components"
                         .format(missing))
    mu = np.stack([model.operators[n] for n in moments])
    site_moments = [np.stack([model.site_operators[n][k] for n in moments])
                    for k in range(len(model.sites))]
    return assign_pseudospin(
        model.h_eff, mu, [sp.dim for sp in model.sites], site_moments, axes=axes,
        common_axis=common_axis, rotate_frame=rotate_frame,
        site_electrons=[sp.n_electrons for sp in model.sites],
        orbitals=[sp.orbitals for sp in model.sites], energy_shift=energy_shift,
        provenance=provenance, comments=comments)


# --- the file -------------------------------------------------------------------------------

def write_pseudospin(path, model: PseudospinModel, *, title: str = "",
                     include_site_moments: bool = True) -> Path:
    """Write the OuluSpin file (module docstring: the format) and return its path."""
    path = Path(path)
    if not model.provenance:
        log.warning("the pseudospin file %s carries no Hamiltonian provenance; once the "
                    "ab initio route feeds this file, the 6.2/6.5 screening and "
                    "decoupling records belong in it ", path.name)

    d = model.model_dim
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
    "sites": [...], "basis": ndarray, "energies": ndarray, "matrices": {name: ndarray},
    "site_matrices": {(site, name): ndarray}}``. Refuses an unknown ``format_version``
    rather than guessing.
    """
    text = Path(path).read_text().splitlines()
    header: Dict[str, str] = {}
    provenance: Dict[str, object] = {}
    frame_rows: List[List[float]] = []
    sites: List[Dict[str, object]] = []
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
    return {"header": header, "provenance": provenance,
            "frame_rotation": (np.array(frame_rows, dtype=float) if frame_rows
                               else np.eye(3)),
            "sites": sites, "basis": np.array(basis, dtype=np.int64),
            "energies": np.array(energies, dtype=float), "matrices": matrices,
            "site_matrices": site_matrices}


__all__ = ["FORMAT_VERSION", "AXIS_DEGENERACY_RTOL", "PseudospinSite", "PseudospinModel",
           "assign_pseudospin", "pseudospin_from_model", "write_pseudospin",
           "read_pseudospin"]
