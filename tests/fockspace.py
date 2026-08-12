"""A brute-force second-quantization reference over a tiny Fock space (test-only).

⚠ **This exists because a check whose two sides share an implementation cannot see an error in
that implementation**. Every SC-NEVPT2 quantity — the perturber norms, the
Koopmans denominators, the class energies — is a matrix element of explicit ladder-operator
strings, and the only way to test the derivation in ``kuiva/pt/`` rather than test it against
itself is to build those strings directly. So this module shares **no code** with
``kuiva.pt``, ``kuiva.rdm`` or ``kuiva.ci``: determinants are Python integers, operators are
dense matrices over the *whole* Fock space, and the class projectors are built from occupation
patterns. It is the Tier-0 scale version of the uncontracted reference
schedules for stage 3.

It is unavoidably exponential — the Fock space of ``n`` spinors has ``2**n`` states and the
elementary excitation operators are ``n**2`` dense matrices of that size — so
:data:`MAX_MODES` refuses beyond a laptop-instant problem rather than letting a test quietly
take minutes.

Conventions, chosen to match what is being tested and stated here so the match is checkable:

* A determinant is a bitmask over spinors; bit ``p`` set means spinor ``p`` occupied. The
  reference ordering of a string is ``a+_0 a+_1 ... |vac>`` with **ascending** mode index,
  which is ``kuiva.ci.strings``' convention and the sign rule below implements it.
* ``H = sum_pq h_pq a+_p a_q + 1/2 sum_pqrs (pq|rs) a+_p a+_r a_s a_q``, chemists' notation.
* ``gamma_pq = <a+_p a_q>``, ``Gamma_pqrs = <a+_p a+_r a_s a_q>`` — ``kuiva.rdm.rdm``'s.
* Energies are **electronic only**: no nuclear repulsion anywhere, since it is an additive
  constant that cancels out of every energy *difference* this module produces.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np

#: Largest Fock space this module will build. ``n = 8`` is a 256-dimensional space and 67 MB
#: of elementary excitation matrices; ``n = 10`` is 1.7 GB, which is not a Tier-0 tool.
MAX_MODES = 8


class FockSpace:
    """Every occupation pattern of ``n`` spinors, with dense ladder operators."""

    def __init__(self, n: int) -> None:
        if int(n) > MAX_MODES:
            raise ValueError("the brute-force Fock space is a Tier-0 tool and refuses beyond "
                             "{} spinors; got {}".format(MAX_MODES, n))
        self.n = int(n)
        self.dim = 1 << self.n
        self.occupations = np.array(
            [[(d >> p) & 1 for p in range(self.n)] for d in range(self.dim)], dtype=np.int8)
        self.n_elec = self.occupations.sum(axis=1)
        self._excitations = None

    # -- ladder operators -------------------------------------------------------------------
    def _sign(self, det: int, p: int) -> int:
        """``(-1)**(number of occupied modes below p)`` — the Jordan-Wigner parity."""
        return -1 if bin(det & ((1 << p) - 1)).count("1") % 2 else 1

    def create(self, p: int) -> np.ndarray:
        """``a+_p`` as a dense ``(dim, dim)`` matrix."""
        op = np.zeros((self.dim, self.dim))
        for det in range(self.dim):
            if (det >> p) & 1:
                continue
            op[det | (1 << p), det] = self._sign(det, p)
        return op

    def annihilate(self, p: int) -> np.ndarray:
        return self.create(p).T

    def excitations(self) -> np.ndarray:
        """``E[p, q] = a+_p a_q`` for every pair, as ``(n, n, dim, dim)``. Cached."""
        if self._excitations is None:
            cre = [self.create(p) for p in range(self.n)]
            e = np.empty((self.n, self.n, self.dim, self.dim))
            for p in range(self.n):
                for q in range(self.n):
                    e[p, q] = cre[p] @ cre[q].T
            self._excitations = e
        return self._excitations

    # -- the Hamiltonian --------------------------------------------------------------------
    def hamiltonian(self, h: np.ndarray, eri: np.ndarray,
                    modes: Sequence[int] = None) -> np.ndarray:
        """``H`` over the whole Fock space, from the elementary excitation matrices.

        ``modes`` restricts *both* sums to a subset of spinors, which is how the Dyall active
        Hamiltonian is built. The ordering identity used is
        ``a+_p a+_r a_s a_q = E_pq E_rs - delta_qr E_ps`` — an operator identity, applied here
        rather than the ``E_pq E_rs`` shortcut being assumed anywhere in the code under test.
        """
        modes = list(range(self.n)) if modes is None else list(modes)
        e = self.excitations()
        m = len(modes)
        idx = np.ix_(modes, modes)
        eflat = e[idx].reshape(m * m, self.dim * self.dim)
        gflat = np.asarray(eri)[np.ix_(modes, modes, modes, modes)].reshape(m * m, m * m)

        # 1/2 sum (pq|rs) E_pq E_rs, as one GEMM over the fused (rs, y) index
        left = (gflat.T @ eflat).reshape(m * m, self.dim, self.dim)   # sum_pq (pq|rs) E_pq
        ham = 0.5 * (left.transpose(1, 0, 2).reshape(self.dim, m * m * self.dim)
                     @ eflat.reshape(m * m * self.dim, self.dim))
        # - 1/2 sum_pqs (pq|qs) E_ps
        w = np.einsum("pqqs->ps", np.asarray(eri)[np.ix_(modes, modes, modes, modes)])
        ham -= 0.5 * np.tensordot(w.reshape(m * m), eflat, axes=([0], [0])).reshape(
            self.dim, self.dim)
        # sum h_pq E_pq
        ham += np.tensordot(np.asarray(h)[idx].reshape(m * m), eflat,
                            axes=([0], [0])).reshape(self.dim, self.dim)
        return np.ascontiguousarray(ham, dtype=np.complex128)

    # -- selection --------------------------------------------------------------------------
    def sector(self, n_elec: int) -> np.ndarray:
        """Indices of the determinants with ``n_elec`` electrons."""
        return np.nonzero(self.n_elec == int(n_elec))[0]

    def pattern(self, inactive: Sequence[int], virtual: Sequence[int],
                holes: Sequence[int], particles: Sequence[int],
                n_active: int) -> np.ndarray:
        """Determinants with exactly these inactive holes and virtual particles.

        The active occupation is free apart from its electron count, which is what makes this
        the projector ``P_l`` onto one strongly contracted class label set.
        """
        occ = self.occupations
        keep = np.ones(self.dim, dtype=bool)
        for i in inactive:
            keep &= (occ[:, i] == (0 if i in holes else 1))
        for a in virtual:
            keep &= (occ[:, a] == (1 if a in particles else 0))
        active = [p for p in range(self.n) if p not in inactive and p not in virtual]
        if active:
            keep &= (occ[:, active].sum(axis=1) == int(n_active))
        elif int(n_active) != 0:
            return np.zeros(0, dtype=int)
        return np.nonzero(keep)[0]


class ReferenceNEVPT2:
    """SC-NEVPT2 by explicit projection, for a system small enough to enumerate.

    Builds ``H`` and ``H_D`` over the whole Fock space, obtains ``|Psi_0>`` by diagonalizing
    the CAS block of ``H``, and then, for every class and label set, forms
    ``|Psi_l> = P_l H |Psi_0>`` as a vector and reads off

    ::

        N_l  = <Psi_l|Psi_l>
        dE_l = <Psi_l|H_D|Psi_l> / N_l - E_0
        E^k  = - sum_l N_l / dE_l

    No NEVPT2 formula is used anywhere — only projectors, matrix-vector products and inner
    products — so an error in ``kuiva/pt/``'s algebra cannot hide here.

    ⚠ **The Dyall active Hamiltonian is built with the FULL inactive core**, once, and the same
    operator acts on every determinant regardless of how many core holes it carries. Restricting
    ``H`` to determinant blocks with equal core occupation instead — which looks equivalent and
    is not — would give each class a different ``H_act`` and produce plausible wrong
    denominators for every class with a core hole.

    Parameters
    ----------
    h, eri : the **full** one- and two-electron integrals over all ``n`` spinors, in the
        pseudo-canonical basis (i.e. after the caller has diagonalized the Fock blocks).
    inactive, active, virtual : index lists partitioning the spinors.
    eps_inactive, eps_virtual : the Dyall orbital energies for those spaces.
    frozen, deleted : index lists, optional
        Inactive spinors that keep their mean field but may not carry a hole, and virtual
        spinors that may not carry a particle — the frozen-core / deleted-virtual option of
        the frozen-core semantics. ⚠ Implemented here as a restriction of the **label enumeration**
        only, because that is what the approximation *is*; a reference that instead
        projected those orbitals out of ``H`` would be testing a different approximation and
        would agree with nothing.
    """

    def __init__(self, h: np.ndarray, eri: np.ndarray, inactive: Sequence[int],
                 active: Sequence[int], virtual: Sequence[int],
                 eps_inactive: Sequence[float], eps_virtual: Sequence[float],
                 n_active_elec: int, frozen: Sequence[int] = (),
                 deleted: Sequence[int] = ()) -> None:
        self.n = int(np.shape(h)[0])
        self.space = FockSpace(self.n)
        self.inactive = list(inactive)
        self.active = list(active)
        self.virtual = list(virtual)
        self.label_inactive = [i for i in self.inactive if i not in set(frozen)]
        self.label_virtual = [a for a in self.virtual if a not in set(deleted)]
        self.n_active_elec = int(n_active_elec)
        self.n_elec = len(self.inactive) + self.n_active_elec
        self.eps_inactive = np.asarray(eps_inactive, dtype=float)
        self.eps_virtual = np.asarray(eps_virtual, dtype=float)
        self.h = np.asarray(h, dtype=np.complex128)
        self.eri = np.asarray(eri, dtype=np.complex128)

        self.ham = self.space.hamiltonian(self.h, self.eri)
        self.f_inactive = self._inactive_fock()
        self.e_core = self._core_energy()
        self._build_reference()
        self._build_dyall()

    # -- the frozen core --------------------------------------------------------------------
    def _inactive_fock(self) -> np.ndarray:
        """``f^I_pq = h_pq + sum_i [(pq|ii) - (pi|iq)]``, over **all** spinors."""
        f = self.h.copy()
        for i in self.inactive:
            f += self.eri[:, :, i, i] - self.eri[:, i, i, :]
        return f

    def _core_energy(self) -> float:
        """``1/2 sum_i (h_ii + f^I_ii)`` — electronic only."""
        idx = self.inactive
        return 0.5 * float(np.real(sum(self.h[i, i] + self.f_inactive[i, i] for i in idx)))

    # -- the reference ------------------------------------------------------------------------
    def _build_reference(self) -> None:
        cas = self.space.pattern(self.inactive, self.virtual, holes=(), particles=(),
                                 n_active=self.n_active_elec)
        vals, vecs = np.linalg.eigh(self.ham[np.ix_(cas, cas)])
        self.cas_indices = cas
        self.cas_energies = vals                     # total electronic energies
        self.cas_vectors = vecs

    def state(self, root: int = 0) -> np.ndarray:
        """The ``root``-th CAS eigenvector, embedded in the full Fock space."""
        psi = np.zeros(self.space.dim, dtype=np.complex128)
        psi[self.cas_indices] = self.cas_vectors[:, root]
        return psi

    def energy(self, root: int = 0) -> float:
        return float(self.cas_energies[root])

    # -- the zeroth-order Hamiltonian --------------------------------------------------------
    def _build_dyall(self) -> None:
        occ = self.space.occupations
        diag = np.zeros(self.space.dim)
        for value, i in zip(self.eps_inactive, self.inactive):
            diag += value * occ[:, i]
        for value, a in zip(self.eps_virtual, self.virtual):
            diag += value * occ[:, a]
        self.h_act = self.space.hamiltonian(self.f_inactive, self.eri, modes=self.active)
        # C, fixed by H_D|Psi_0> = E_0|Psi_0>: the core carries e_core, and the eps sum counts
        # sum_i eps_i on the reference.
        self.dyall_diagonal = diag
        self.dyall_const = self.e_core - float(np.sum(self.eps_inactive))
        self.h_dyall = (self.h_act + np.diag(diag).astype(np.complex128)
                        + self.dyall_const * np.eye(self.space.dim))
        #: The active-space eigenvalue alone, i.e. the total minus the frozen-core energy.
        self.e_active = [float(v) - self.e_core for v in self.cas_energies]

    def dyall_residual(self, root: int = 0) -> float:
        """``|| (H_D - E_0) |Psi_0> ||`` — the reference's own consistency check.

        Zero to machine precision by construction, and the single most informative assertion
        about this module: it fails on a wrong ``f^I``, a wrong core energy, a wrong constant,
        and on any sign error in the ladder operators.
        """
        psi = self.state(root)
        return float(np.linalg.norm(self.h_dyall @ psi - self.energy(root) * psi))

    # -- the classes ----------------------------------------------------------------------------
    def _labels(self, n_holes: int, n_particles: int):
        return [(h, p)
                for h in itertools.combinations(self.label_inactive, n_holes)
                for p in itertools.combinations(self.label_virtual, n_particles)]

    def _eps_group(self, orbital: int) -> int:
        """Which exactly-degenerate ``eps`` group an external orbital belongs to.

        ⚠ Written out here rather than imported: the contraction group is *part of the method*
        (see :mod:`kuiva.pt.classes`), so a reference that borrowed Kuiva's grouping could not
        fail on a wrong one. Plain Python, no shared helper.
        """
        table = ([(i, e) for i, e in zip(self.inactive, self.eps_inactive)]
                 + [(a, e) for a, e in zip(self.virtual, self.eps_virtual)])
        value = dict(table)[orbital]
        distinct = []
        for _, e in sorted(table, key=lambda kv: kv[1]):
            if not distinct or abs(e - distinct[-1]) > 1e-9 * max(1.0, abs(e)):
                distinct.append(e)
        return min(range(len(distinct)), key=lambda k: abs(distinct[k] - value))

    def class_terms(self, n_holes: int, n_particles: int, root: int = 0
                    ) -> List[Tuple[float, float]]:
        """``(N_l, dE_l)`` for every label set of the class with this hole/particle count."""
        return [(n, d) for n, d, _ in self._class_terms_keyed(n_holes, n_particles, root)]

    def _class_terms_keyed(self, n_holes: int, n_particles: int, root: int = 0):
        """``(N_l, dE_l, contraction group key)`` per label set."""
        psi0 = self.state(root)
        e0 = self.energy(root)
        hpsi = self.ham @ psi0
        n_active = self.n_elec - (len(self.inactive) - n_holes) - n_particles
        terms = []
        for holes, particles in self._labels(n_holes, n_particles):
            idx = self.space.pattern(self.inactive, self.virtual, holes, particles, n_active)
            if idx.size == 0:
                continue
            vec = np.zeros(self.space.dim, dtype=np.complex128)
            vec[idx] = hpsi[idx]
            norm = float(np.real(vec.conj() @ vec))
            if norm <= 0.0:
                continue
            expectation = float(np.real(vec.conj() @ self.h_dyall @ vec)) / norm
            key = (tuple(sorted(self._eps_group(h) for h in holes)),
                   tuple(sorted(self._eps_group(p) for p in particles)))
            terms.append((norm, expectation - e0, key))
        return terms

    def class_energy(self, n_holes: int, n_particles: int,
                     root: int = 0) -> Tuple[float, float]:
        """``(sum_l N_l, -sum_G N_G / dE_G)`` for one class.

        ⚠ **Contracted over whole degenerate-``eps`` groups**, because that is what the method
        is: inside such a group the canonical orbitals are arbitrary, so a per-orbital label is
        not a label. A per-label sum here would disagree with Kuiva by ~4e-6 relative on
        ``Sir (0')`` and agree on everything else — which is exactly how the requirement was
        found, so the reference implements it rather than papering over it.
        """
        groups = {}
        for norm, denom, key in self._class_terms_keyed(n_holes, n_particles, root):
            acc = groups.setdefault(key, [0.0, 0.0])
            acc[0] += norm
            acc[1] += norm * denom
        total = sum(n for n, _ in groups.values())
        energy = -sum(n * n / d for n, d in groups.values())
        return total, energy

    def class_energy_uncontracted(self, n_holes: int, n_particles: int,
                                  root: int = 0) -> float:
        """The **uncontracted** second-order energy of one class: the Hylleraas minimum.

        Minimizing ``f(c) = 2 Re(c^dag b) + c^dag A c`` with ``b = P_S H|Psi_0>`` and
        ``A = P_S (H_D - E_0) P_S`` over the *whole* class subspace ``S`` — every determinant
        with the class's hole/particle counts, not one contracted function per label set —
        gives ``f_min = -b^dag A^-1 b``.

        This is the bottom rung of the Hylleraas hierarchy
        ``E2_unc <= E2_FIC <= E2_SC <= 0``: strongly contracted amplitudes are the same
        minimization restricted to a one-dimensional subspace per label set, so the SC energy
        can only be higher. ⚠ The ordering holds only while ``A`` is positive definite — with a
        genuine intruder the functional is unbounded below and the inequality means nothing.
        """
        psi0 = self.state(root)
        e0 = self.energy(root)
        hpsi = self.ham @ psi0
        n_active = self.n_elec - (len(self.inactive) - n_holes) - n_particles
        idx = np.concatenate([
            self.space.pattern(self.inactive, self.virtual, holes, particles, n_active)
            for holes, particles in self._labels(n_holes, n_particles)] or [np.zeros(0, int)])
        idx = np.unique(idx.astype(int))
        if idx.size == 0:
            return 0.0
        a = self.h_dyall[np.ix_(idx, idx)] - e0 * np.eye(idx.size)
        b = hpsi[idx]
        return -float(np.real(b.conj() @ np.linalg.solve(a, b)))

    #: ``kuiva.pt`` class name -> ``(inactive holes, virtual particles)``. ⚠ Derived from the
    #: class *definitions* — the hole/particle counts that define the subspace — not from the
    #: implementation being tested.
    CLASS_PATTERN: Dict[str, Tuple[int, int]] = {
        "Sijrs": (2, 2), "Srsi": (1, 2), "Sijr": (2, 1), "Srs": (0, 2),
        "Sij": (2, 0), "Sir": (1, 1), "Sr": (0, 1), "Si": (1, 0),
    }

    def by_name(self, name: str, root: int = 0) -> Tuple[float, float]:
        return self.class_energy(*self.CLASS_PATTERN[name], root=root)

    # -- density matrices, for the primitive-level tests ----------------------------------------
    def _annihilated(self, root: int) -> np.ndarray:
        """``A[p] = a_p |Psi>`` over the active spinors, as ``(n_act, dim)``."""
        psi = self.state(root)
        return np.array([self.space.create(p).T @ psi for p in self.active])

    def rdm1(self, root: int = 0) -> np.ndarray:
        """``gamma_pq = <a+_p a_q> = (a_p|Psi>)^dag (a_q|Psi>)``, active spinors only."""
        a = self._annihilated(root)
        return np.ascontiguousarray(a.conj() @ a.T)

    def rdm2(self, root: int = 0) -> np.ndarray:
        """``Gamma_pqrs = <a+_p a+_r a_s a_q> = (a_r a_p|Psi>)^dag (a_s a_q|Psi>)``."""
        psi = self.state(root)
        ann = [self.space.create(p).T for p in self.active]
        m = len(self.active)
        pairs = np.array([[ann[r] @ (ann[p] @ psi) for r in range(m)] for p in range(m)])
        flat = pairs.reshape(m * m, -1)
        gram = flat.conj() @ flat.T                        # [(p,r), (q,s)]
        return np.ascontiguousarray(
            gram.reshape(m, m, m, m).transpose(0, 2, 1, 3))   # -> [p, q, r, s]

    def hole_pair(self, root: int = 0) -> np.ndarray:
        """``<a_u a_t a+_v a+_w>`` as the ``(n^2, n^2)`` matrix indexed ``[(t,u), (v,w)]``."""
        psi = self.state(root)
        cre = [self.space.create(p) for p in self.active]
        m = len(self.active)
        pairs = np.array([[cre[t] @ (cre[u] @ psi) for u in range(m)] for t in range(m)])
        flat = pairs.reshape(m * m, -1)
        return np.ascontiguousarray(flat.conj() @ flat.T)

    def koopmans_annihilation(self, root: int = 0) -> np.ndarray:
        """``K_tu = <a+_t (H_act - E_act) a_u>`` by explicit operators.

        ⚠ ``E_act`` is the **active-space** eigenvalue, not the total: ``H_act`` here carries
        neither the frozen-core energy nor the Dyall constant, matching what
        :func:`kuiva.pt.contractions.koopmans_annihilation` is handed.
        """
        a = self._annihilated(root)                        # (n_act, dim), rows a_u|Psi>
        rhs = (self.h_act @ a.T) - self.e_active[root] * a.T
        return np.ascontiguousarray(a.conj() @ rhs)

    def koopmans_creation(self, root: int = 0) -> np.ndarray:
        """``K'_tu = <a_t (H_act - E_act) a+_u>`` by explicit operators."""
        psi = self.state(root)
        created = np.array([self.space.create(p) @ psi for p in self.active])
        rhs = (self.h_act @ created.T) - self.e_active[root] * created.T
        return np.ascontiguousarray(created.conj() @ rhs)


def random_integrals(n: int, seed: int = 0, scale: float = 1.0
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """A random complex Hermitian ``h`` and a random ``eri`` with **exactly** 4-fold symmetry.

    ⚠ Built as ``sum_P B^P_pq B^P_rs`` with ``B`` Hermitian in ``(p,q)``, which is the same
    structure a Cholesky/DF factorization produces — so ``(pq|rs) = (rs|pq)`` and
    ``(pq|rs) = (qp|sr)*`` hold and the 8-fold relations do not. Generating ``eri`` any other
    way would test the code against integrals it will never see.
    """
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = 0.5 * (h + h.conj().T)
    naux = 2 * n
    b = rng.normal(size=(naux, n, n)) + 1j * rng.normal(size=(naux, n, n))
    b = 0.5 * (b + b.transpose(0, 2, 1).conj())
    eri = scale * np.tensordot(b, b, axes=([0], [0])) / naux
    return h, eri
