# References

The published methods, algorithms, formulas, basis sets and libraries that Kuiva implements,
follows or depends on. The policy is to **over-cite rather than under-cite**: if a published
idea is used anywhere in the code, it is listed here and cited from the page (and the
docstring) that uses it.

**How this list works.** One entry per work (a software entry may bundle the citations its
authors ask for together). Entries are numbered, and **a number is permanent**: it is
assigned once, never reused and never reordered, so an inline citation `[N]` anywhere in
this documentation stays valid forever. New entries take the next free number and are placed
in whichever topical section fits — numbers inside a section are therefore not guaranteed to
be contiguous. A dash after the DOI gives the role the work plays in Kuiva, including the
occasional negative result or recorded departure.

Citations for software used only to generate the validation reference data live with that
data, under `tests/`, not here.

**Sections:**
[Software and libraries](#software-and-libraries) ·
[Relativistic Hamiltonians and X2C](#relativistic-hamiltonians-and-x2c) ·
[Basis sets](#basis-sets) ·
[Orthogonalization and the working basis](#orthogonalization-and-the-working-basis) ·
[Spinor basis, time reversal, and Kramers pairs](#spinor-basis-time-reversal-and-kramers-pairs) ·
[Broken-symmetry references](#broken-symmetry-references) ·
[Point-group and double-group symmetry](#point-group-and-double-group-symmetry) ·
[Integral factorization and transformation](#integral-factorization-and-transformation) ·
[Determinant CI, selected CI, and orbital optimization](#determinant-ci-selected-ci-and-orbital-optimization) ·
[Orbital entanglement](#orbital-entanglement) ·
[Fragment localization](#fragment-localization) ·
[DMRG and tree tensor networks](#dmrg-and-tree-tensor-networks) ·
[SC-NEVPT2](#sc-nevpt2) ·
[Quantum-computing CI solvers](#quantum-computing-ci-solvers) ·
[Population analysis and orbital files](#population-analysis-and-orbital-files) ·
[Multiplets, magnetic moments, and pseudospin](#multiplets-magnetic-moments-and-pseudospin) ·
[The CI sigma vector and determinant addressing](#the-ci-sigma-vector-and-determinant-addressing) ·
[Atomic Slater–Condon parameters](#atomic-slater-condon-parameters) ·
[Numerical methods](#numerical-methods)

---

## Software and libraries

<a id="r1"></a>**[1]** **PySCF** (pinned 2.14.0). Q. Sun *et al.*, *J. Chem. Phys.* **153**, 024109 (2020). DOI: [10.1063/5.0006074](https://doi.org/10.1063/5.0006074); Q. Sun *et al.*, *WIREs Comput. Mol. Sci.* **8**, e1340 (2018). DOI: [10.1002/wcms.1340](https://doi.org/10.1002/wcms.1340) — the scalar-relativistic X2C front end (SCF guess and integrals); the only external dependency in the multireference path is its ingestion output.

<a id="r2"></a>**[2]** **libcint**. Q. Sun, *J. Comput. Chem.* **36**, 1664–1671 (2015). DOI: [10.1002/jcc.23981](https://doi.org/10.1002/jcc.23981) — Gaussian integral evaluation, used via PySCF and linked directly by the compiled integral kernels.

<a id="r3"></a>**[3]** **HDF5**. The HDF Group, *Hierarchical Data Format, version 5*, <https://www.hdfgroup.org/HDF5/> — checkpoint and cache format, via `h5py`.

<a id="r4"></a>**[4]** **pybind11**. W. Jakob, J. Rhinelander, D. Moldovan, *pybind11 — Seamless operability between C++11 and Python* (2017), <https://github.com/pybind/pybind11> — the C++/Python boundary of the optional compiled kernel backend.

<a id="r5"></a>**[5]** **Intel oneAPI Math Kernel Library (MKL)**. Intel Corporation, *oneAPI MKL Developer Reference*, <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/> — the threaded BLAS/LAPACK behind NumPy on the reference build, called directly by the compiled kernels and asked for its per-region thread width.

<a id="r6"></a>**[6]** **OpenBLAS**. Z. Xianyi, W. Qian, Z. Chothia, *OpenBLAS*, <https://www.openblas.net/> — one of the three BLAS libraries whose thread width Kuiva controls per region.

<a id="r7"></a>**[7]** **BLIS**. F. G. Van Zee, R. A. van de Geijn, *ACM Trans. Math. Softw.* **41**, 14 (2015). DOI: [10.1145/2764454](https://doi.org/10.1145/2764454) — the third controlled BLAS.

<a id="r8"></a>**[8]** **OpenMP**. OpenMP Architecture Review Board, *OpenMP Application Programming Interface* 5.0 (2018), <https://www.openmp.org/specifications/> — thread-level parallelism inside the compiled kernels (via Intel's `libiomp5`), including the composability rules that let MKL and a kernel share one runtime.

<a id="r9"></a>**[9]** **Basis Set Exchange**. B. P. Pritchard, D. Altarawy, B. Didier, T. D. Gibson, T. L. Windus, *J. Chem. Inf. Model.* **59**, 4814–4820 (2019). DOI: [10.1021/acs.jcim.9b00725](https://doi.org/10.1021/acs.jcim.9b00725); D. Feller, *J. Comput. Chem.* **17**, 1571 (1996); K. L. Schuchardt *et al.*, *J. Chem. Inf. Model.* **47**, 1045 (2007) — authoritative basis data for the families PySCF does not bundle.

## Relativistic Hamiltonians and X2C

<a id="r10"></a>**[10]** W. Liu, D. Peng, *J. Chem. Phys.* **131**, 031104 (2009). DOI: [10.1063/1.3159445](https://doi.org/10.1063/1.3159445) — the scalar (spin-free) one-electron X2C used for the front-end SCF.

<a id="r11"></a>**[11]** D. Peng, M. Reiher, *Theor. Chem. Acc.* **131**, 1081 (2012). DOI: [10.1007/s00214-011-1081-y](https://doi.org/10.1007/s00214-011-1081-y) — review of exact two-component decoupling.

<a id="r12"></a>**[12]** T. Nakajima, K. Hirao, *Chem. Rev.* **112**, 385 (2012). DOI: [10.1021/cr200040s](https://doi.org/10.1021/cr200040s) — review of two-component relativistic methods.

<a id="r13"></a>**[13]** W. Kutzelnigg, W. Liu, *J. Chem. Phys.* **123**, 241102 (2005). DOI: [10.1063/1.2137315](https://doi.org/10.1063/1.2137315) — exact two-component decoupling.

<a id="r14"></a>**[14]** W. Liu, D. Peng, *J. Chem. Phys.* **125**, 044102 (2006). DOI: [10.1063/1.2222365](https://doi.org/10.1063/1.2222365) — exact two-component decoupling.

<a id="r15"></a>**[15]** M. Iliaš, T. Saue, *J. Chem. Phys.* **126**, 064102 (2007). DOI: [10.1063/1.2436882](https://doi.org/10.1063/1.2436882) — the one-step X2C formulation.

<a id="r16"></a>**[16]** J. Liu, L. Cheng, *J. Chem. Phys.* **148**, 144108 (2018). DOI: [10.1063/1.5023750](https://doi.org/10.1063/1.5023750) — **X2CAMF**: the atomic mean-field two-electron picture change, Kuiva's default two-electron spin–orbit treatment.

<a id="r17"></a>**[17]** B. A. Hess, C. M. Marian, U. Wahlgren, O. Gropen, *Chem. Phys. Lett.* **251**, 365 (1996). DOI: [10.1016/0009-2614(96)00119-4](https://doi.org/10.1016/0009-2614(96)00119-4) — AMFI, the atomic mean-field idea X2CAMF builds on.

<a id="r18"></a>**[18]** C. Zhang, L. Cheng, *J. Phys. Chem. A* **126**, 4537 (2022). DOI: [10.1021/acs.jpca.2c02181](https://doi.org/10.1021/acs.jpca.2c02181) — the X2CAMF reference implementation and its spherically constrained atomic solver.

<a id="r19"></a>**[19]** I. P. Grant, *Relativistic Quantum Theory of Atoms and Molecules*, Springer (2007) — four-component atomic structure theory, used for the atomic Dirac–Hartree–Fock reference inside the mean field.

<a id="r20"></a>**[20]** K. G. Dyall, K. Fægri, *Introduction to Relativistic Quantum Chemistry*, Oxford University Press (2007) — ch. 4, 7 and 11 for the Dirac–Coulomb, Gaunt and Breit operators; ch. 6 and 10 for Kramers-paired spinor bases and the barred/unbarred notation.

<a id="r21"></a>**[21]** T. Saue, H. J. Aa. Jensen, *J. Chem. Phys.* **111**, 6211 (1999). DOI: [10.1063/1.479958](https://doi.org/10.1063/1.479958) — average-of-configuration open-shell Dirac–Hartree–Fock; also the design precedent for abelian double groups and the two-component integral transformation.

<a id="r22"></a>**[22]** J. Thyssen, PhD thesis, University of Southern Denmark (2001) — average-of-configuration open-shell relativistic SCF practice.

<a id="r23"></a>**[23]** C. C. J. Roothaan, *Rev. Mod. Phys.* **32**, 179 (1960). DOI: [10.1103/RevModPhys.32.179](https://doi.org/10.1103/RevModPhys.32.179) — the configuration-average open-shell SCF energy functional.

<a id="r24"></a>**[24]** L. Visscher, K. G. Dyall, *At. Data Nucl. Data Tables* **67**, 207 (1997). DOI: [10.1006/adnd.1997.0751](https://doi.org/10.1006/adnd.1997.0751) — the Gaussian finite-nucleus model and its rms-radius-from-mass parametrization (`nuclear_model="gaussian"`).

<a id="r25"></a>**[25]** R. E. Stanton, S. Havriliak, *J. Chem. Phys.* **81**, 1910 (1984). DOI: [10.1063/1.447865](https://doi.org/10.1063/1.447865) — restricted kinetic balance.

<a id="r26"></a>**[26]** K. G. Dyall, K. Fægri, *Chem. Phys. Lett.* **174**, 25 (1990). DOI: [10.1016/0009-2614(90)85321-3](https://doi.org/10.1016/0009-2614(90)85321-3) — restricted kinetic balance.

<a id="r27"></a>**[27]** D. Peng, M. Reiher, *J. Chem. Phys.* **136**, 244108 (2012). DOI: [10.1063/1.4729788](https://doi.org/10.1063/1.4729788) — local (atom-blocked, DLU) exact decoupling; also the picture change of property operators and the renormalization matrix R.

<a id="r28"></a>**[28]** J. C. Boettger, *Phys. Rev. B* **57**, 8743 (1998). DOI: [10.1103/PhysRevB.57.8743](https://doi.org/10.1103/PhysRevB.57.8743) — empirical (SNSO) spin–orbit screening factors. Considered and **rejected** in favour of X2CAMF; listed because the analysis followed it.

<a id="r29"></a>**[29]** M. Filatov, W. Zou, D. Cremer, *J. Chem. Phys.* **139**, 014106 (2013). DOI: [10.1063/1.4811776](https://doi.org/10.1063/1.4811776) — screened spin–orbit operators; same rejected-alternative status as [28].

<a id="r30"></a>**[30]** B. de Souza, G. Farias, F. Neese, R. Izsák, *J. Chem. Theory Comput.* **15**, 1896 (2019). DOI: [10.1021/acs.jctc.8b00841](https://doi.org/10.1021/acs.jctc.8b00841) — mean-field spin–orbit practice; same rejected-alternative status as [28].

<a id="r31"></a>**[31]** N. N. Greenwood, A. Earnshaw, *Chemistry of the Elements*, 2nd ed., Butterworth-Heinemann (1997) — the curated common-oxidation-states table behind `configuration=`.

<a id="r32"></a>**[32]** NIST Atomic Spectra Database, National Institute of Standards and Technology, <https://www.nist.gov/pml/atomic-spectra-database> — atomic ground-state configurations, via PySCF's aufbau tables.

## Basis sets

<a id="r33"></a>**[33]** P. Pollak, F. Weigend, *J. Chem. Theory Comput.* **13**, 3696–3705 (2017). DOI: [10.1021/acs.jctc.7b00593](https://doi.org/10.1021/acs.jctc.7b00593) — Karlsruhe x2c-nZVPall DZ/TZ sets (segmented, H–Rn); the `-2c` variants are the default for two-component work.

<a id="r34"></a>**[34]** Y. J. Franzke, L. Spiske, P. Pollak, F. Weigend, *J. Chem. Theory Comput.* **16**, 5658–5674 (2020). DOI: [10.1021/acs.jctc.0c00546](https://doi.org/10.1021/acs.jctc.0c00546) — x2c-QZVPall and the x2c-JFIT Coulomb-fitting auxiliaries.

<a id="r35"></a>**[35]** J. G. Hill, K. A. Peterson, *J. Chem. Phys.* **147**, 244106 (2017). DOI: [10.1063/1.5010587](https://doi.org/10.1063/1.5010587) — cc-pVnZ-X2C for K–Ra.

<a id="r36"></a>**[36]** Q. Lu, K. A. Peterson, *J. Chem. Phys.* **145**, 054111 (2016). DOI: [10.1063/1.4959280](https://doi.org/10.1063/1.4959280) — cc-pVnZ-X2C for the lanthanides (La–Lu).

<a id="r37"></a>**[37]** R. Feng, K. A. Peterson, *J. Chem. Phys.* **147**, 084108 (2017). DOI: [10.1063/1.4994725](https://doi.org/10.1063/1.4994725) — cc-pVnZ-X2C for the actinides.

<a id="r38"></a>**[38]** J. G. Hill, *ccRepo: a correlation consistent basis sets repository*, <http://www.grant-hill.group.shef.ac.uk/ccrepo/> — distribution point for the Peterson X2C families.

<a id="r39"></a>**[39]** K. G. Dyall, *Theor. Chem. Acc.* **135**, 128 (2016). DOI: [10.1007/s00214-016-1884-y](https://doi.org/10.1007/s00214-016-1884-y) — the Dyall uncontracted heavy-element sets (and the series of Dyall papers referenced therein); benchmarking only.

<a id="r40"></a>**[40]** P.-O. Widmark, P.-Å. Malmqvist, B. O. Roos, *Theor. Chim. Acta* **77**, 291 (1990). DOI: [10.1007/BF01120130](https://doi.org/10.1007/BF01120130) — the ANO construction behind ANO-RCC.

<a id="r41"></a>**[41]** B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark, *J. Phys. Chem. A* **108**, 2851 (2004). DOI: [10.1021/jp031064+](https://doi.org/10.1021/jp031064+) — ANO-RCC, main group.

<a id="r42"></a>**[42]** B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark, *J. Phys. Chem. A* **109**, 6575 (2005). DOI: [10.1021/jp0581126](https://doi.org/10.1021/jp0581126) — ANO-RCC, transition metals.

<a id="r43"></a>**[43]** B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark, *Chem. Phys. Lett.* **409**, 295 (2005). DOI: [10.1016/j.cplett.2005.05.011](https://doi.org/10.1016/j.cplett.2005.05.011) — ANO-RCC, actinides.

## Orthogonalization and the working basis

<a id="r44"></a>**[44]** P.-O. Löwdin, *J. Chem. Phys.* **18**, 365 (1950). DOI: [10.1063/1.1747632](https://doi.org/10.1063/1.1747632) — symmetric orthogonalization; also the population analysis named after it.

<a id="r45"></a>**[45]** P.-O. Löwdin, *Adv. Quantum Chem.* **5**, 185–199 (1970). DOI: [10.1016/S0065-3276(08)60339-1](https://doi.org/10.1016/S0065-3276(08)60339-1) — canonical and symmetric orthogonalization.

<a id="r46"></a>**[46]** A. Szabo, N. S. Ostlund, *Modern Quantum Chemistry*, Dover (1996) — textbook treatment of orthogonalization (sec. 3.4.5) and of ⟨S²⟩ as a one- plus two-body operator (ch. 2.5, 3.8).

<a id="r47"></a>**[47]** J. Almlöf, K. Fægri, K. Korsell, *J. Comput. Chem.* **3**, 385 (1982). DOI: [10.1002/jcc.540030314](https://doi.org/10.1002/jcc.540030314) — linear-dependence removal by overlap-eigenvalue truncation; projected small-basis starting vectors as an SCF guess.

<a id="r48"></a>**[48]** F. Aquilante, T. B. Pedersen, V. Veryazov, R. Lindh, *WIREs Comput. Mol. Sci.* **3**, 143 (2013). DOI: [10.1002/wcms.1117](https://doi.org/10.1002/wcms.1117) — Cholesky techniques, including Cholesky orthogonalization.

<a id="r49"></a>**[49]** F. Aquilante, T. B. Pedersen, R. Lindh, *J. Chem. Phys.* **125**, 174101 (2006). DOI: [10.1063/1.2360264](https://doi.org/10.1063/1.2360264) — Cholesky orthogonalization (the optional scheme; it performs no linear-dependence detection).

<a id="r50"></a>**[50]** R. P. Steele, R. A. DiStasio Jr., Y. Shao, J. Kong, M. Head-Gordon, *J. Chem. Phys.* **125**, 074108 (2006). DOI: [10.1063/1.2234371](https://doi.org/10.1063/1.2234371) — projecting molecular orbitals between basis sets, and the Fock/density-matrix alternative to it.

<a id="r51"></a>**[51]** I. Fdez. Galván *et al.*, *J. Chem. Theory Comput.* **15**, 5925 (2019). DOI: [10.1021/acs.jctc.9b00532](https://doi.org/10.1021/acs.jctc.9b00532) — OpenMolcas; cited for the `EXPBAS` basis-projection workflow precedent and as a Tier-2 comparison program.

<a id="r52"></a>**[52]** F. Aquilante *et al.*, *J. Chem. Phys.* **152**, 214117 (2020). DOI: [10.1063/5.0004835](https://doi.org/10.1063/5.0004835) — OpenMolcas (modernization paper); same role as [51].

<a id="r53"></a>**[53]** B. C. Carlson, J. M. Keller, *Phys. Rev.* **105**, 102 (1957). DOI: [10.1103/PhysRev.105.102](https://doi.org/10.1103/PhysRev.105.102) — least-squares optimality of symmetric orthonormalization, which is why a projected orbital set is repaired that way.

<a id="r54"></a>**[54]** A. T. Amos, G. G. Hall, *Proc. R. Soc. London A* **263**, 483 (1961). DOI: [10.1098/rspa.1961.0175](https://doi.org/10.1098/rspa.1961.0175) — corresponding orbitals / principal angles between orbital subspaces, the invariant a basis projection is judged by.

## Spinor basis, time reversal, and Kramers pairs

<a id="r55"></a>**[55]** H. A. Kramers, *Proc. Amsterdam Acad.* **33**, 959 (1930) — time reversal and Kramers degeneracy.

<a id="r56"></a>**[56]** E. P. Wigner, *Gruppentheorie und ihre Anwendung auf die Quantenmechanik der Atomspektren*, Vieweg (1931), ch. 26 — time reversal as an antiunitary operation.

<a id="r57"></a>**[57]** T. Saue, *ChemPhysChem* **12**, 3077–3094 (2011). DOI: [10.1002/cphc.201100682](https://doi.org/10.1002/cphc.201100682) — relativistic Hamiltonians for chemistry; the −iσ_y K convention and barred/unbarred notation.

<a id="r58"></a>**[58]** J. Thyssen, T. Fleig, H. J. Aa. Jensen, *J. Chem. Phys.* **129**, 034109 (2008). DOI: [10.1063/1.2943670](https://doi.org/10.1063/1.2943670) — Kramers-restricted (time-reversal-adapted) two-component MCSCF.

<a id="r59"></a>**[59]** T. Fleig, J. Olsen, L. Visscher, *J. Chem. Phys.* **119**, 2963 (2003). DOI: [10.1063/1.1590636](https://doi.org/10.1063/1.1590636) — Kramers-restricted CI.

<a id="r60"></a>**[60]** A. Bunse-Gerstner, R. Byers, V. Mehrmann, *SIAM J. Matrix Anal. Appl.* **10**, 419 (1989). DOI: [10.1137/0610030](https://doi.org/10.1137/0610030) — self-dual (quaternion) Hermitian eigenvalue structure, the Kramers-restricted eigensolver's subspace problem.

<a id="r61"></a>**[61]** P. Pulay, T. P. Hamilton, *J. Chem. Phys.* **88**, 4926 (1988). DOI: [10.1063/1.454704](https://doi.org/10.1063/1.454704) — UHF natural orbitals as an active-space guess.

## Broken-symmetry references

<a id="r196"></a>**[196]** L. Noodleman, *J. Chem. Phys.* **74**, 5737 (1981). DOI: [10.1063/1.440939](https://doi.org/10.1063/1.440939) — the broken-symmetry (valence-bond) description of antiferromagnetic coupling; the construction behind the `broken_symmetry=` starting guess.

<a id="r197"></a>**[197]** L. Noodleman, E. R. Davidson, *Chem. Phys.* **109**, 131 (1986). DOI: [10.1016/0301-0104(86)80192-6](https://doi.org/10.1016/0301-0104(86)80192-6) — broken-symmetry practice: converge the high-spin state, flip localized magnetic orbitals.

## Point-group and double-group symmetry

<a id="r62"></a>**[62]** G. F. Koster, J. O. Dimmock, R. G. Wheeler, H. Statz, *Properties of the Thirty-Two Point Groups*, MIT Press (1963) — double-group character tables.

<a id="r63"></a>**[63]** S. L. Altmann, P. Herzig, *Point-Group Theory Tables*, Clarendon Press (1994) — double-group character tables.

<a id="r64"></a>**[64]** E. P. Wigner, *Group Theory and its Application to the Quantum Mechanics of Atomic Spectra*, Academic Press (1959), ch. 15 — the spin-1/2 representation and the double group.

<a id="r65"></a>**[65]** H. A. Bethe, *Ann. Phys.* **3**, 133 (1929). DOI: [10.1002/andp.19293950202](https://doi.org/10.1002/andp.19293950202) — the original double-group construction.

<a id="r66"></a>**[66]** L. Visscher, *Chem. Phys. Lett.* **253**, 20 (1996). DOI: [10.1016/0009-2614(96)00234-5](https://doi.org/10.1016/0009-2614(96)00234-5) — abelian double groups as the working symmetry of a relativistic molecular code (with [21]).

<a id="r67"></a>**[67]** J. Ivanic, K. Ruedenberg, *J. Phys. Chem.* **100**, 6342 (1996). DOI: [10.1021/jp953350u](https://doi.org/10.1021/jp953350u); erratum *J. Phys. Chem. A* **102**, 9099 (1998). DOI: [10.1021/jp9833350](https://doi.org/10.1021/jp9833350) — rotation matrices for real spherical harmonics by recursion, used to build the group operators on the AO basis.

<a id="r68"></a>**[68]** W. Burnside, *Theory of Groups of Finite Order*, 2nd ed., Cambridge University Press (1911), ch. XV — character tables from class-sum matrices.

<a id="r69"></a>**[69]** J. D. Dixon, *Numer. Math.* **10**, 446 (1967). DOI: [10.1007/BF02162876](https://doi.org/10.1007/BF02162876) — the numerical class-sum character-table algorithm; the printed full double-group tables are computed, not transcribed.

<a id="r70"></a>**[70]** R. S. Mulliken, *J. Chem. Phys.* **23**, 1997 (1955). DOI: [10.1063/1.1740655](https://doi.org/10.1063/1.1740655) — irrep naming conventions for the single-valued rows.

<a id="r71"></a>**[71]** M. Tinkham, *Group Theory and Quantum Mechanics*, McGraw-Hill (1964), ch. 3 — projection of a reducible representation onto characters, used to classify converged degenerate blocks.

<a id="r72"></a>**[72]** D. J. Thouless, *Nucl. Phys.* **21**, 225 (1960). DOI: [10.1016/0029-5582(60)90048-1](https://doi.org/10.1016/0029-5582(60)90048-1) — the Fock-space image of a one-body unitary, applied as a product of two-mode rotations so a symmetry operation acts on a CI vector exactly.

<a id="r73"></a>**[73]** G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins University Press (2013) — the Givens QR (sec. 5.2) behind the adjacent-pair elimination, and re-orthogonalized Gram–Schmidt.

## Integral factorization and transformation

<a id="r74"></a>**[74]** N. H. F. Beebe, J. Linderberg, *Int. J. Quantum Chem.* **12**, 683–705 (1977). DOI: [10.1002/qua.560120408](https://doi.org/10.1002/qua.560120408) — Cholesky decomposition of the two-electron integral matrix.

<a id="r75"></a>**[75]** H. Koch, A. Sánchez de Merás, T. B. Pedersen, *J. Chem. Phys.* **118**, 9481 (2003). DOI: [10.1063/1.1578621](https://doi.org/10.1063/1.1578621) — Cholesky pivoting, error bounds, and modern practice.

<a id="r76"></a>**[76]** F. Aquilante, T. B. Pedersen, R. Lindh, *J. Chem. Phys.* **126**, 194106 (2007). DOI: [10.1063/1.2736701](https://doi.org/10.1063/1.2736701) — integral-direct Cholesky, in which only the selected columns are ever evaluated (`fitting="cholesky-direct"`).

<a id="r77"></a>**[77]** F. Aquilante *et al.*, in *Linear-Scaling Techniques in Computational Chemistry and Physics*, Springer (2011), pp. 301–343. DOI: [10.1007/978-90-481-2853-2_13](https://doi.org/10.1007/978-90-481-2853-2_13) — Cholesky methods review.

<a id="r78"></a>**[78]** F. Aquilante, R. Lindh, T. B. Pedersen, *J. Chem. Phys.* **127**, 114107 (2007). DOI: [10.1063/1.2777146](https://doi.org/10.1063/1.2777146) — the atomic / one-centre decomposition: pivoting on complete shell-pair orbits, which keeps the factorization exactly invariant under rotations of an atom. The default pivot selection.

<a id="r79"></a>**[79]** J. L. Whitten, *J. Chem. Phys.* **58**, 4496 (1973). DOI: [10.1063/1.1679012](https://doi.org/10.1063/1.1679012) — density fitting / resolution of the identity.

<a id="r80"></a>**[80]** B. I. Dunlap, J. W. D. Connolly, J. R. Sabin, *J. Chem. Phys.* **71**, 3396 (1979). DOI: [10.1063/1.438728](https://doi.org/10.1063/1.438728) — the Coulomb-metric density fitting.

<a id="r81"></a>**[81]** O. Vahtras, J. Almlöf, M. W. Feyereisen, *Chem. Phys. Lett.* **213**, 514 (1993). DOI: [10.1016/0009-2614(93)89151-7](https://doi.org/10.1016/0009-2614(93)89151-7) — the RI-V formulation.

<a id="r82"></a>**[82]** M. Yoshimine, IBM Technical Report RJ-555 (1969) — integral transformation by successive quarter transformations.

<a id="r83"></a>**[83]** T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure Theory*, Wiley (2000) — the working reference for second quantization and Slater–Condon algebra (ch. 1–2), integral transformation (ch. 9), MCSCF theory and redundant-rotation classification (ch. 10, 12), and string-driven CI (ch. 11).

<a id="r84"></a>**[84]** L. Visscher, *Theor. Chem. Acc.* **98**, 68 (1997). DOI: [10.1007/s002140050280](https://doi.org/10.1007/s002140050280) — two-component integral transformation with spin-free AO integrals and complex spinor coefficients (with [21]).

## Determinant CI, selected CI, and orbital optimization

<a id="r85"></a>**[85]** J. C. Slater, *Phys. Rev.* **34**, 1293 (1929). DOI: [10.1103/PhysRev.34.1293](https://doi.org/10.1103/PhysRev.34.1293) — the Slater rules; also the theory of complex spectra behind the Slater–Condon parameters.

<a id="r86"></a>**[86]** E. U. Condon, *Phys. Rev.* **36**, 1121 (1930). DOI: [10.1103/PhysRev.36.1121](https://doi.org/10.1103/PhysRev.36.1121) — the Condon rules.

<a id="r87"></a>**[87]** A. Scemama, E. Giner, arXiv:[1311.6244](https://arxiv.org/abs/1311.6244) (2013) — bitmask determinant representation, excitation analysis by XOR/popcount.

<a id="r88"></a>**[88]** Y. Garniron *et al.*, *J. Chem. Theory Comput.* **15**, 3591 (2019). DOI: [10.1021/acs.jctc.9b00176](https://doi.org/10.1021/acs.jctc.9b00176) — Quantum Package 2.0; modern selected-CI determinant machinery.

<a id="r89"></a>**[89]** E. R. Sayfutyarova, Q. Sun, G. K.-L. Chan, G. Knizia, *J. Chem. Theory Comput.* **13**, 4063 (2017). DOI: [10.1021/acs.jctc.7b00128](https://doi.org/10.1021/acs.jctc.7b00128) — AVAS. ⚠ Kuiva follows the projection, the occupied/virtual separation and the eigenvalue threshold, but projects onto its **own free-atom reference orbitals** rather than a minimal MINAO basis — the orbitals selected agree, while the eigenvalues are not numerically comparable with another program's AVAS.

<a id="r90"></a>**[90]** B. Huron, J. P. Malrieu, P. Rancurel, *J. Chem. Phys.* **58**, 5745 (1973). DOI: [10.1063/1.1679199](https://doi.org/10.1063/1.1679199) — CIPSI, the perturbative selection criterion of the cheap CI.

<a id="r91"></a>**[91]** N. M. Tubman, J. Lee, T. Y. Takeshita, M. Head-Gordon, K. B. Whaley, *J. Chem. Phys.* **145**, 044112 (2016). DOI: [10.1063/1.4955109](https://doi.org/10.1063/1.4955109) — ASCI: selection against a bounded set of generators, as implemented here.

<a id="r92"></a>**[92]** A. A. Holmes, N. M. Tubman, C. J. Umrigar, *J. Chem. Theory Comput.* **12**, 3674 (2016). DOI: [10.1021/acs.jctc.6b00407](https://doi.org/10.1021/acs.jctc.6b00407) — heat-bath selected CI.

<a id="r93"></a>**[93]** B. O. Roos, P. R. Taylor, P. E. M. Siegbahn, *Chem. Phys.* **48**, 157 (1980). DOI: [10.1016/0301-0104(80)80045-0](https://doi.org/10.1016/0301-0104(80)80045-0) — the complete-active-space SCF method.

<a id="r94"></a>**[94]** P. E. M. Siegbahn, J. Almlöf, A. Heiberg, B. O. Roos, *J. Chem. Phys.* **74**, 2384 (1981). DOI: [10.1063/1.441359](https://doi.org/10.1063/1.441359) — the Newton–Raphson CASSCF formulation.

<a id="r95"></a>**[95]** H.-J. Werner, P. J. Knowles, *J. Chem. Phys.* **82**, 5053 (1985). DOI: [10.1063/1.448627](https://doi.org/10.1063/1.448627) — second-order MCSCF.

<a id="r96"></a>**[96]** H. J. Aa. Jensen, H. Ågren, *Chem. Phys. Lett.* **110**, 140 (1984). DOI: [10.1016/0009-2614(84)80166-1](https://doi.org/10.1016/0009-2614(84)80166-1) — augmented-Hessian MCSCF.

<a id="r97"></a>**[97]** H. J. Aa. Jensen, P. Jørgensen, *J. Chem. Phys.* **80**, 1204 (1984). DOI: [10.1063/1.446797](https://doi.org/10.1063/1.446797) — augmented-Hessian / norm-extended optimization.

<a id="r98"></a>**[98]** D. A. Kreplin, P. J. Knowles, H.-J. Werner, *J. Chem. Phys.* **150**, 194106 (2019). DOI: [10.1063/1.5094644](https://doi.org/10.1063/1.5094644) — modern second-order MCSCF practice.

<a id="r99"></a>**[99]** P. Jørgensen, P. Swanstrøm, D. L. Yeager, *J. Chem. Phys.* **78**, 347 (1983). DOI: [10.1063/1.444508](https://doi.org/10.1063/1.444508) — one-index-transformed Fock matrices, making a Hessian-vector product cost the same order as a gradient.

<a id="r100"></a>**[100]** S. C. Eisenstat, H. F. Walker, *SIAM J. Sci. Comput.* **17**, 16 (1996). DOI: [10.1137/0917003](https://doi.org/10.1137/0917003) — inexact-Newton forcing sequences.

<a id="r101"></a>**[101]** R. S. Dembo, S. C. Eisenstat, T. Steihaug, *SIAM J. Numer. Anal.* **19**, 400 (1982). DOI: [10.1137/0719025](https://doi.org/10.1137/0719025) — inexact Newton methods.

<a id="r102"></a>**[102]** J. Nocedal, S. J. Wright, *Numerical Optimization*, 2nd ed., Springer (2006), ch. 4 — trust-region methods.

<a id="r103"></a>**[103]** T. Helgaker, *Chem. Phys. Lett.* **182**, 503 (1991). DOI: [10.1016/0009-2614(91)90115-P](https://doi.org/10.1016/0009-2614(91)90115-P) — level shifting in orbital optimization.

<a id="r104"></a>**[104]** J. Nocedal, *Math. Comput.* **35**, 773 (1980). DOI: [10.1090/S0025-5718-1980-0572855-7](https://doi.org/10.1090/S0025-5718-1980-0572855-7) — L-BFGS.

<a id="r105"></a>**[105]** D. C. Liu, J. Nocedal, *Math. Program.* **45**, 503 (1989). DOI: [10.1007/BF01589116](https://doi.org/10.1007/BF01589116) — L-BFGS.

<a id="r106"></a>**[106]** H. J. Aa. Jensen, K. G. Dyall, T. Saue, K. Fægri, *J. Chem. Phys.* **104**, 4083 (1996). DOI: [10.1063/1.471644](https://doi.org/10.1063/1.471644) — relativistic (Kramers-paired, complex-parameter) MCSCF; the CI roots are already the spin–orbit eigenstates.

<a id="r107"></a>**[107]** T. Fleig, J. Olsen, C. M. Marian, *J. Chem. Phys.* **114**, 4775 (2001). DOI: [10.1063/1.1349076](https://doi.org/10.1063/1.1349076) — two-component (spinor) CI/MCSCF.

<a id="r108"></a>**[108]** P.-O. Löwdin, *Phys. Rev.* **97**, 1474 (1955). DOI: [10.1103/PhysRev.97.1474](https://doi.org/10.1103/PhysRev.97.1474) — natural orbitals and occupation numbers; also the natural-orbital decomposition the spinor-density analysis performs.

## Orbital entanglement

<a id="r109"></a>**[109]** J. Rissler, R. M. Noack, S. R. White, *Chem. Phys.* **323**, 519 (2006). DOI: [10.1016/j.chemphys.2005.10.018](https://doi.org/10.1016/j.chemphys.2005.10.018) — orbital entanglement entropies and mutual information.

<a id="r110"></a>**[110]** Ö. Legeza, J. Sólyom, *Phys. Rev. B* **68**, 195116 (2003). DOI: [10.1103/PhysRevB.68.195116](https://doi.org/10.1103/PhysRevB.68.195116) — single-orbital entropies in DMRG.

<a id="r111"></a>**[111]** G. Barcza, Ö. Legeza, K. H. Marti, M. Reiher, *Phys. Rev. A* **83**, 012508 (2011). DOI: [10.1103/PhysRevA.83.012508](https://doi.org/10.1103/PhysRevA.83.012508) — entanglement-based orbital ordering.

<a id="r112"></a>**[112]** M. Fiedler, *Czechoslovak Math. J.* **23**, 298 (1973) — the Fiedler vector, used for path-network orbital ordering.

<a id="r113"></a>**[113]** C. J. Stein, M. Reiher, *J. Chem. Theory Comput.* **12**, 1760 (2016). DOI: [10.1021/acs.jctc.6b00156](https://doi.org/10.1021/acs.jctc.6b00156) — entanglement-driven automated active-space selection.

<a id="r114"></a>**[114]** K. Boguslawski, P. Tecmer, *Int. J. Quantum Chem.* **115**, 1289 (2015). DOI: [10.1002/qua.24832](https://doi.org/10.1002/qua.24832) — review of entanglement measures and conventions.

## Fragment localization

<a id="r115"></a>**[115]** D. Claudino, N. J. Mayhall, *J. Chem. Theory Comput.* **15**, 1053–1064 (2019). DOI: [10.1021/acs.jctc.8b01112](https://doi.org/10.1021/acs.jctc.8b01112) — SPADE: orbital partition by singular value decomposition. Published for one fragment and its environment; Kuiva applies it **sequentially** so that several sites partition one active space.

<a id="r116"></a>**[116]** J. Pipek, P. G. Mezey, *J. Chem. Phys.* **90**, 4916 (1989). DOI: [10.1063/1.456588](https://doi.org/10.1063/1.456588) — Pipek–Mezey localization; the classical alternative, **not** the method used (a recorded departure: the fragment projection is exact, non-iterative and complex-safe on spinors unchanged).

## DMRG and tree tensor networks

<a id="r117"></a>**[117]** S. R. White, *Phys. Rev. Lett.* **69**, 2863 (1992). DOI: [10.1103/PhysRevLett.69.2863](https://doi.org/10.1103/PhysRevLett.69.2863) — the density-matrix renormalization group; two-site DMRG.

<a id="r118"></a>**[118]** U. Schollwöck, *Ann. Phys.* **326**, 96 (2011). DOI: [10.1016/j.aop.2010.09.012](https://doi.org/10.1016/j.aop.2010.09.012) — canonical forms, sweeps, and truncation in the MPS language.

<a id="r119"></a>**[119]** T. Xiang, *Phys. Rev. B* **53**, R10445 (1996). DOI: [10.1103/PhysRevB.53.R10445](https://doi.org/10.1103/PhysRevB.53.R10445) — complementary-operator technique for long-range Hamiltonians.

<a id="r120"></a>**[120]** S. R. White, R. L. Martin, *J. Chem. Phys.* **110**, 4127 (1999). DOI: [10.1063/1.478522](https://doi.org/10.1063/1.478522) — ab initio DMRG with complementary operators.

<a id="r121"></a>**[121]** C. Hubig, I. P. McCulloch, U. Schollwöck, *Phys. Rev. B* **95**, 035129 (2017). DOI: [10.1103/PhysRevB.95.035129](https://doi.org/10.1103/PhysRevB.95.035129) — generic MPO/TTNO compilation; also the sparse (list-of-transitions) operator storage the compiled TTNO is kept in.

<a id="r122"></a>**[122]** G. K.-L. Chan, A. Keselman, N. Nakatani, Z. Li, S. R. White, *J. Chem. Phys.* **145**, 014102 (2016). DOI: [10.1063/1.4955108](https://doi.org/10.1063/1.4955108) — matrix product operators and states in quantum chemistry.

<a id="r123"></a>**[123]** Y.-Y. Shi, L.-M. Duan, G. Vidal, *Phys. Rev. A* **74**, 022320 (2006). DOI: [10.1103/PhysRevA.74.022320](https://doi.org/10.1103/PhysRevA.74.022320) — tree tensor network states.

<a id="r124"></a>**[124]** V. Murg, F. Verstraete, Ö. Legeza, R. M. Noack, *Phys. Rev. B* **82**, 205105 (2010). DOI: [10.1103/PhysRevB.82.205105](https://doi.org/10.1103/PhysRevB.82.205105) — TTNS for quantum chemistry.

<a id="r125"></a>**[125]** N. Nakatani, G. K.-L. Chan, *J. Chem. Phys.* **138**, 134113 (2013). DOI: [10.1063/1.4798639](https://doi.org/10.1063/1.4798639) — ab initio TTNS algorithms.

<a id="r126"></a>**[126]** K. Gunst, F. Verstraete, S. Wouters, Ö. Legeza, D. Van Neck, *J. Chem. Theory Comput.* **14**, 2026 (2018). DOI: [10.1021/acs.jctc.8b00098](https://doi.org/10.1021/acs.jctc.8b00098) — T3NS; tree-network quantum chemistry.

<a id="r127"></a>**[127]** J. J. Dorando, J. Hachmann, G. K.-L. Chan, *J. Chem. Phys.* **127**, 084109 (2007). DOI: [10.1063/1.2768360](https://doi.org/10.1063/1.2768360) — state-averaged DMRG in a shared renormalized basis.

<a id="r128"></a>**[128]** Ö. Legeza, J. Röder, B. A. Hess, *Phys. Rev. B* **67**, 125114 (2003). DOI: [10.1103/PhysRevB.67.125114](https://doi.org/10.1103/PhysRevB.67.125114) — dynamic block-state selection.

<a id="r129"></a>**[129]** S. R. White, *J. Chem. Phys.* **122**, 084108 (2005). DOI: [10.1063/1.1854132](https://doi.org/10.1063/1.1854132) — density-matrix perturbation (noise); implemented as the deterministic subspace expansion.

<a id="r130"></a>**[130]** C. Hubig, I. P. McCulloch, U. Schollwöck, F. A. Wolf, *Phys. Rev. B* **91**, 155115 (2015). DOI: [10.1103/PhysRevB.91.155115](https://doi.org/10.1103/PhysRevB.91.155115) — strictly single-site DMRG with subspace expansion.

<a id="r131"></a>**[131]** G. K.-L. Chan, M. Head-Gordon, *J. Chem. Phys.* **116**, 4462 (2002). DOI: [10.1063/1.1449459](https://doi.org/10.1063/1.1449459) — energy extrapolation in the discarded weight.

<a id="r132"></a>**[132]** R. Olivares-Amaya, W. Hu, N. Nakatani, S. Sharma, J. Yang, G. K.-L. Chan, *J. Chem. Phys.* **142**, 034102 (2015). DOI: [10.1063/1.4905329](https://doi.org/10.1063/1.4905329) — DMRG in practice; extrapolation protocol.

<a id="r133"></a>**[133]** T. Hikihara, H. Ueda, K. Okunishi, K. Harada, T. Nishino, *Phys. Rev. Research* **5**, 013031 (2023). DOI: [10.1103/PhysRevResearch.5.013031](https://doi.org/10.1103/PhysRevResearch.5.013031) — automatic structural optimization of tree tensor networks (the adaptive-topology moves).

<a id="r134"></a>**[134]** S. Singh, R. N. C. Pfeifer, G. Vidal, *Phys. Rev. B* **83**, 115125 (2011). DOI: [10.1103/PhysRevB.83.115125](https://doi.org/10.1103/PhysRevB.83.115125) — symmetry-blocked tensors.

<a id="r135"></a>**[135]** S. Knecht, Ö. Legeza, M. Reiher, *J. Chem. Phys.* **140**, 041101 (2014). DOI: [10.1063/1.4862495](https://doi.org/10.1063/1.4862495) — relativistic (general-spinor) DMRG.

<a id="r136"></a>**[136]** H. Zhai *et al.*, *J. Chem. Phys.* **159**, 234801 (2023). DOI: [10.1063/5.0180424](https://doi.org/10.1063/5.0180424) — block2; modern DMRG implementation practice, including the relativistic path.

<a id="r137"></a>**[137]** P. Jordan, E. Wigner, *Z. Phys.* **47**, 631 (1928). DOI: [10.1007/BF01331938](https://doi.org/10.1007/BF01331938) — the Jordan–Wigner transformation: fermionic modes to spins/qubits; fixes the network's mode-ordering convention and the qubit mapping alike.

<a id="r138"></a>**[138]** Y. Kurashige, T. Yanai, *J. Chem. Phys.* **135**, 094104 (2011). DOI: [10.1063/1.3629454](https://doi.org/10.1063/1.3629454) — higher-order RDMs from matrix-product states for multireference perturbation theory.

<a id="r139"></a>**[139]** S. Guo, M. A. Watson, W. Hu, Q. Sun, G. K.-L. Chan, *J. Chem. Theory Comput.* **12**, 1583 (2016). DOI: [10.1021/acs.jctc.6b00118](https://doi.org/10.1021/acs.jctc.6b00118) — NEVPT2 on a DMRG reference; the precedent for the network contraction provider (Kuiva serves the same primitives through applied-string Gram contractions instead of stored higher densities).

<a id="r140"></a>**[140]** C. Bloch, *Nucl. Phys.* **6**, 329 (1958). DOI: [10.1016/0029-5582(58)90116-0](https://doi.org/10.1016/0029-5582(58)90116-0) — effective Hamiltonians on a model space.

<a id="r141"></a>**[141]** J. des Cloizeaux, *Nucl. Phys.* **20**, 321 (1960). DOI: [10.1016/0029-5582(60)90177-2](https://doi.org/10.1016/0029-5582(60)90177-2) — the Hermitian (canonical) effective Hamiltonian.

<a id="r142"></a>**[142]** C. J. Morningstar, M. Weinstein, *Phys. Rev. D* **54**, 4131 (1996). DOI: [10.1103/PhysRevD.54.4131](https://doi.org/10.1103/PhysRevD.54.4131) — CORE; the Rayleigh–Ritz block compression used for the local-multiplet model space is its zeroth step.

## SC-NEVPT2

<a id="r143"></a>**[143]** C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger, J.-P. Malrieu, *J. Chem. Phys.* **114**, 10252 (2001). DOI: [10.1063/1.1361246](https://doi.org/10.1063/1.1361246) — n-electron valence state perturbation theory (NEVPT2).

<a id="r144"></a>**[144]** C. Angeli, R. Cimiraglia, J.-P. Malrieu, *J. Chem. Phys.* **117**, 9138 (2002). DOI: [10.1063/1.1515317](https://doi.org/10.1063/1.1515317) — the strongly contracted variant.

<a id="r145"></a>**[145]** K. G. Dyall, *J. Chem. Phys.* **102**, 4909 (1995). DOI: [10.1063/1.469539](https://doi.org/10.1063/1.469539) — the Dyall zeroth-order Hamiltonian.

<a id="r146"></a>**[146]** C. Angeli, M. Pastore, R. Cimiraglia, *Theor. Chem. Acc.* **117**, 743 (2007). DOI: [10.1007/s00214-006-0207-0](https://doi.org/10.1007/s00214-006-0207-0) — review with the class-by-class working equations. ⚠ Every published equation is spin-free and real; the ones Kuiva implements were **re-derived in spinor second quantization** for a complex two-component Hamiltonian with 4-fold integral symmetry only, and the method page states them in that form.

<a id="r147"></a>**[147]** K. Kollmar, K. Sivalingam, Y. Guo, F. Neese, *J. Chem. Phys.* **155**, 234104 (2021). DOI: [10.1063/5.0072129](https://doi.org/10.1063/5.0072129) — avoiding the stored four-particle density matrix in NEVPT2. Kuiva goes one step further and contracts the integrals into one perturber vector per external label, so no rank-3 or rank-4 object is formed at all.

<a id="r148"></a>**[148]** B. O. Roos, K. Andersson, *Chem. Phys. Lett.* **245**, 215 (1995). DOI: [10.1016/0009-2614(95)01010-7](https://doi.org/10.1016/0009-2614(95)01010-7) — the real level shift for intruder states.

<a id="r149"></a>**[149]** N. Forsberg, P.-Å. Malmqvist, *Chem. Phys. Lett.* **274**, 196 (1997). DOI: [10.1016/S0009-2614(97)00669-6](https://doi.org/10.1016/S0009-2614(97)00669-6) — the imaginary level shift.

<a id="r150"></a>**[150]** C. Angeli, S. Borini, M. Cestari, R. Cimiraglia, *J. Chem. Phys.* **121**, 4043 (2004). DOI: [10.1063/1.1778711](https://doi.org/10.1063/1.1778711) — quasi-degenerate NEVPT2. Not implemented and not planned; cited because the decision was taken on a measurement, and so that the boundary is documented.

<a id="r151"></a>**[151]** A. A. Granovsky, *J. Chem. Phys.* **134**, 214113 (2011). DOI: [10.1063/1.3596699](https://doi.org/10.1063/1.3596699) — extended multi-configuration quasi-degenerate PT; same status as [150].

<a id="r152"></a>**[152]** S. Sharma, G. Jeanmairet, A. Alavi, *J. Chem. Phys.* **144**, 034103 (2016). DOI: [10.1063/1.4939752](https://doi.org/10.1063/1.4939752) — model-space invariance in multireference PT; same status as [150].

<a id="r153"></a>**[153]** R. Majumder, A. Yu. Sokolov, *J. Phys. Chem. A* **127**, 546 (2023). DOI: [10.1021/acs.jpca.2c07953](https://doi.org/10.1021/acs.jpca.2c07953) — spin–orbit QD-NEVPT2, the closest prior art to a two-component NEVPT2.

## Quantum-computing CI solvers

<a id="r154"></a>**[154]** J. T. Seeley, M. J. Richard, P. J. Love, *J. Chem. Phys.* **137**, 224109 (2012). DOI: [10.1063/1.4768229](https://doi.org/10.1063/1.4768229) — the Jordan–Wigner electronic-structure mapping and its symplectic (X-mask, Z-mask) bookkeeping.

<a id="r155"></a>**[155]** S. B. Bravyi, A. Y. Kitaev, *Ann. Phys.* **298**, 210 (2002). DOI: [10.1006/aphy.2002.6254](https://doi.org/10.1006/aphy.2002.6254) — the Bravyi–Kitaev encoding; the alternative the mapping registry stays open to.

<a id="r156"></a>**[156]** "Chemistry beyond the scale of exact diagonalization on a quantum-centric supercomputer", *Sci. Adv.* (2025). DOI: [10.1126/sciadv.adu9991](https://doi.org/10.1126/sciadv.adu9991) — sample-based quantum diagonalization (SQD), the primary algorithm implemented.

<a id="r157"></a>**[157]** "Localized sample-based quantum diagonalization for strongly correlated chemistry", *PNAS* (2025). DOI: [10.1073/pnas.2603914123](https://doi.org/10.1073/pnas.2603914123) — localized SQD.

<a id="r158"></a>**[158]** "Sample-based Krylov quantum diagonalization", arXiv:[2501.09702](https://arxiv.org/abs/2501.09702) (2025) — the Krylov variant, sampling Trotterized time-evolved states.

<a id="r159"></a>**[159]** M. Motta *et al.*, *Electron. Struct.* **6**, 013001 (2024). DOI: [10.1088/2516-1075/ad3592](https://doi.org/10.1088/2516-1075/ad3592) — the quantum-subspace method family.

<a id="r160"></a>**[160]** J. Romero, R. Babbush, J. R. McClean, C. Hempel, P. J. Love, A. Aspuru-Guzik, *Quantum Sci. Technol.* **4**, 014008 (2018). DOI: [10.1088/2058-9565/aad3e4](https://doi.org/10.1088/2058-9565/aad3e4) — unitary coupled cluster for VQE; generalized here to complex spinor excitation generators.

<a id="r161"></a>**[161]** F. A. Evangelista, G. K.-L. Chan, G. E. Scuseria, *J. Chem. Phys.* **151**, 244112 (2019). DOI: [10.1063/1.5133059](https://doi.org/10.1063/1.5133059) — exact parametrizations of fermionic wavefunctions via unitary CC.

<a id="r162"></a>**[162]** A. Kandala *et al.*, *Nature* **549**, 242 (2017). DOI: [10.1038/nature23879](https://doi.org/10.1038/nature23879) — the hardware-efficient ansatz family, used as the structure-free control.

<a id="r163"></a>**[163]** I. D. Kivlichan, J. McClean, N. Wiebe, C. Gidney, A. Aspuru-Guzik, G. K.-L. Chan, R. Babbush, *Phys. Rev. Lett.* **120**, 110501 (2018). DOI: [10.1103/PhysRevLett.120.110501](https://doi.org/10.1103/PhysRevLett.120.110501) — fermionic circuit compilation: Givens networks realizing an orbital rotation exactly.

<a id="r164"></a>**[164]** M. Reck, A. Zeilinger, H. J. Bernstein, P. Bertani, *Phys. Rev. Lett.* **73**, 58 (1994). DOI: [10.1103/PhysRevLett.73.58](https://doi.org/10.1103/PhysRevLett.73.58) — the triangular decomposition of a unitary into two-mode rotations.

<a id="r165"></a>**[165]** H. F. Trotter, *Proc. Am. Math. Soc.* **10**, 545 (1959) — product formulas.

<a id="r166"></a>**[166]** M. Suzuki, *Commun. Math. Phys.* **51**, 183 (1976). DOI: [10.1007/BF01609348](https://doi.org/10.1007/BF01609348) — higher-order product formulas.

<a id="r167"></a>**[167]** A. Peruzzo *et al.*, *Nat. Commun.* **5**, 4213 (2014). DOI: [10.1038/ncomms5213](https://doi.org/10.1038/ncomms5213) — the variational quantum eigensolver.

<a id="r168"></a>**[168]** J. R. McClean, J. Romero, R. Babbush, A. Aspuru-Guzik, *New J. Phys.* **18**, 023023 (2016). DOI: [10.1088/1367-2630/18/2/023023](https://doi.org/10.1088/1367-2630/18/2/023023) — the theory of variational hybrid quantum-classical algorithms.

<a id="r169"></a>**[169]** K. Mitarai, M. Negoro, M. Kitagawa, K. Fujii, *Phys. Rev. A* **98**, 032309 (2018). DOI: [10.1103/PhysRevA.98.032309](https://doi.org/10.1103/PhysRevA.98.032309) — the parameter-shift rule.

<a id="r170"></a>**[170]** M. Schuld, V. Bergholm, C. Gogolin, J. Izaac, N. Killoran, *Phys. Rev. A* **99**, 032331 (2019). DOI: [10.1103/PhysRevA.99.032331](https://doi.org/10.1103/PhysRevA.99.032331) — analytic gradients on quantum hardware.

<a id="r171"></a>**[171]** M. J. D. Powell, in *Advances in Optimization and Numerical Analysis*, Springer (1994). DOI: [10.1007/978-94-015-8330-5_4](https://doi.org/10.1007/978-94-015-8330-5_4) — COBYLA, the derivative-free optimizer alternative.

<a id="r172"></a>**[172]** S. Bravyi, J. M. Gambetta, A. Mezzacapo, K. Temme, arXiv:[1701.08213](https://arxiv.org/abs/1701.08213) (2017) — qubit tapering from Z₂ Pauli symmetries. Cited for the **negative** result that it does not apply to time reversal, which is antiunitary and has no Pauli generator.

<a id="r173"></a>**[173]** A. Javadi-Abhari *et al.*, "Quantum computing with Qiskit", arXiv:[2405.08810](https://arxiv.org/abs/2405.08810) (2024) — Qiskit and Qiskit Aer: the first backend adapter and the local simulator; testing-only, never a runtime dependency.

## Population analysis and orbital files

<a id="r174"></a>**[174]** I. Mayer, *Chem. Phys. Lett.* **393**, 209 (2004). DOI: [10.1016/j.cplett.2004.06.031](https://doi.org/10.1016/j.cplett.2004.06.031) — basis-set dependence of Mulliken and Löwdin partitions; why a charge from either compares like with like and is not a physical oxidation state.

<a id="r175"></a>**[175]** R. S. Mulliken, *J. Chem. Phys.* **23**, 1833 (1955). DOI: [10.1063/1.1740588](https://doi.org/10.1063/1.1740588) — the original population partition.

<a id="r176"></a>**[176]** G. Schaftenaar, J. H. Noordik, *J. Comput.-Aided Mol. Design* **14**, 123 (2000). DOI: [10.1023/A:1008193805436](https://doi.org/10.1023/A:1008193805436) — the molden file format.

<a id="r177"></a>**[177]** G. Schaftenaar, E. Vlieg, G. Vriend, *J. Comput.-Aided Mol. Design* **31**, 789 (2017). DOI: [10.1007/s10822-017-0042-5](https://doi.org/10.1007/s10822-017-0042-5) — molden 2.0: AO ordering and normalization conventions.

<a id="r178"></a>**[178]** H. B. Schlegel, M. J. Frisch, *Int. J. Quantum Chem.* **54**, 83 (1995). DOI: [10.1002/qua.560540202](https://doi.org/10.1002/qua.560540202) — real solid-harmonic Gaussian conventions.

## Multiplets, magnetic moments, and pseudospin

<a id="r179"></a>**[179]** L. F. Chibotaru, L. Ungur, *J. Chem. Phys.* **137**, 064112 (2012). DOI: [10.1063/1.4739763](https://doi.org/10.1063/1.4739763) — ab initio pseudospin Hamiltonians; the g·gᵀ construction behind the phase-invariant moment reductions.

<a id="r180"></a>**[180]** A. Abragam, B. Bleaney, *Electron Paramagnetic Resonance of Transition Ions*, Clarendon Press, Oxford (1970) — pseudospin and g-tensor conventions for Kramers doublets; the non-Kramers doublet (ch. 3.11, 18.3).

<a id="r181"></a>**[181]** R. D. Cowan, *The Theory of Atomic Structure and Spectra*, University of California Press (1981) — Landé g factors, free-ion multiplets and Russell–Saunders counting (ch. 4, 11); configuration-average energies and Slater–Condon parameter conventions (ch. 6, 10, 14).

<a id="r182"></a>**[182]** J. S. Griffith, *Phys. Rev.* **132**, 316 (1963). DOI: [10.1103/PhysRev.132.316](https://doi.org/10.1103/PhysRev.132.316) — the non-Kramers doublet: tunnelling splitting and the effective-spin form in which the transverse g components vanish identically.

<a id="r183"></a>**[183]** J. Olsen, B. O. Roos, P. Jørgensen, H. J. Aa. Jensen, *J. Chem. Phys.* **89**, 2185 (1988). DOI: [10.1063/1.455063](https://doi.org/10.1063/1.455063) — determinant-based CI with RAS spaces; the one-particle transition-density route to moment matrices, and the string-driven sigma-vector formulation with the folded one-electron operator. ⚠ Kuiva applies **no picture-change transformation** to L and S by default (the same choice OpenMolcas RASSI makes); what removing that approximation requires is [27].

<a id="r184"></a>**[184]** E. Tiesinga, P. J. Mohr, D. B. Newell, B. N. Taylor, *Rev. Mod. Phys.* **93**, 025010 (2021). DOI: [10.1103/RevModPhys.93.025010](https://doi.org/10.1103/RevModPhys.93.025010) — CODATA 2018 recommended values: the free-electron g factor and the reporting value of the speed of light.

## The CI sigma vector and determinant addressing

<a id="r185"></a>**[185]** B. O. Roos, *Chem. Phys. Lett.* **15**, 153 (1972). DOI: [10.1016/0009-2614(72)80140-4](https://doi.org/10.1016/0009-2614(72)80140-4) — direct CI.

<a id="r186"></a>**[186]** P. E. M. Siegbahn, *J. Chem. Phys.* **72**, 1647 (1980). DOI: [10.1063/1.439365](https://doi.org/10.1063/1.439365) — the two-step E_pq resolution of the sigma vector: gather / dense GEMM / gather.

<a id="r187"></a>**[187]** P. J. Knowles, N. C. Handy, *Chem. Phys. Lett.* **111**, 315 (1984). DOI: [10.1016/0009-2614(84)85513-X](https://doi.org/10.1016/0009-2614(84)85513-X) — string-driven full CI.

<a id="r188"></a>**[188]** D. E. Knuth, *The Art of Computer Programming*, Vol. 4A, sec. 7.2.1.3, Addison-Wesley (2011) — lexicographic combinatorial ranking of occupation strings, the complete-CAS address map used in place of a hash table.

## Atomic Slater-Condon parameters

<a id="r189"></a>**[189]** E. U. Condon, G. H. Shortley, *The Theory of Atomic Spectra*, Cambridge University Press (1935), ch. VI and XI — the radial parameters F^k, G^k, R^k and the c^k coefficients, in the ordering and phase conventions the feature uses.

<a id="r190"></a>**[190]** G. Racah, "Theory of Complex Spectra. II", *Phys. Rev.* **62**, 438 (1942). DOI: [10.1103/PhysRev.62.438](https://doi.org/10.1103/PhysRev.62.438) — the 3j symbols, evaluated in exact rational arithmetic from the single-sum formula.

<a id="r191"></a>**[191]** H. A. Bethe, E. E. Salpeter, *Quantum Mechanics of One- and Two-Electron Atoms*, Springer (1957), sec. 12 — hydrogenic spin–orbit constants, the closed form the fits are validated against.

## Numerical methods

<a id="r192"></a>**[192]** E. R. Davidson, *J. Comput. Phys.* **17**, 87 (1975). DOI: [10.1016/0021-9991(75)90065-0](https://doi.org/10.1016/0021-9991(75)90065-0) — the Davidson eigensolver and its diagonal preconditioner.

<a id="r193"></a>**[193]** B. Liu, "The simultaneous expansion method", in *Numerical Algorithms in Chemistry: Algebraic Methods*, LBL-8158, Lawrence Berkeley Laboratory (1978), p. 49 — the block generalization implemented here.

<a id="r194"></a>**[194]** M. Crouzeix, B. Philippe, M. Sadkane, *SIAM J. Sci. Comput.* **15**, 62 (1994). DOI: [10.1137/0915004](https://doi.org/10.1137/0915004) — Davidson convergence analysis and the role of the restart subspace.

<a id="r195"></a>**[195]** Å. Björck, *BIT* **7**, 1 (1967). DOI: [10.1007/BF01934122](https://doi.org/10.1007/BF01934122) — re-orthogonalized modified Gram–Schmidt.
