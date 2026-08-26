#!/usr/bin/env bash
# setup.sh — SOURCE THIS BEFORE RUNNING KUIVA.
#
#   source setup.sh
#
# It prepares the shell and then gets out of the way. There are two situations and it
# detects which one it is in:
#
#   developer sandbox   external/env.sh exists -> source it, and nothing else happens here.
#                       That script owns the Intel oneAPI toolchain, the pinned interpreter
# and the source-built dependencies.
#
#   user install        no sandbox -> check that the interpreter on PATH can actually run
#                       Kuiva (Python, NumPy 1.x, SciPy, h5py, PySCF, a working BLAS) and
#                       say precisely what is missing if it cannot. No Intel toolchain is
#                       needed: the compiled kernel backend is optional and the pure-NumPy
# path is a first-class way to run.
#
# Two things it does in both situations:
#
#   * puts the repository root on PYTHONPATH, so a run uses THIS tree rather than whatever
#     copy happens to be installed;
#   * makes sure a memory limit has been configured, asking for one exactly once if it has
#     not (Kuiva has no built-in limit and never guesses one).
#
# ⚠ It deliberately does NOT set a thread count. That is `KUIVA_NUM_THREADS` and the
# precedence (explicit argument > KUIVA_NUM_THREADS > OMP_NUM_THREADS >
# every core the process may use); a setup script silently choosing a width would make every
# run's cost figures a property of this file. The sandbox's env.sh does set it, because the
# development box's cap is part of the reference contract.
#
# The probe is cached, keyed on a fingerprint of the interpreter, so repeat sourcing costs
# nothing and a changed or upgraded interpreter is re-probed.

# --- Must be sourced ----------------------------------------------------------------------
if [ -n "${BASH_SOURCE:-}" ] && [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "ERROR: setup.sh must be sourced, not executed:  source setup.sh" >&2
    exit 1
fi

_kuiva_root() {
    local src="${BASH_SOURCE[0]:-$0}"
    (cd "$(dirname "${src}")" && pwd)
}

# The dependency floor this file checks against. Keep in step with pyproject.toml; it is part
# of the fingerprint, so changing it re-probes every cached environment.
_KUIVA_REQUIRES="python>=3.9 numpy>=1.22,<2 scipy>=1.6 h5py>=3 pyscf==2.14.0"

_kuiva_say()  { printf '[kuiva] %s\n' "$*"; }
_kuiva_warn() { printf '[kuiva WARNING] %s\n' "$*" >&2; }
_kuiva_err()  { printf '[kuiva ERROR] %s\n' "$*" >&2; }

# --- The interpreter probe ------------------------------------------------------------------
# Everything expensive is here, and it runs once per interpreter. It reports EVERY problem it
# finds rather than the first, because "install this, now install that" is three round trips
# to learn one thing.
_kuiva_probe() {
    local py="$1"
    "${py}" - <<'PYPROBE'
import sys

problems = []
notes = []

if sys.version_info < (3, 9):
    problems.append("Python {}.{} is too old; Kuiva needs 3.9 or newer."
                    .format(sys.version_info[0], sys.version_info[1]))

try:
    import numpy
except ImportError as exc:
    problems.append("NumPy is missing ({}). Install numpy>=1.22,<2.".format(exc))
    numpy = None
else:
    major = int(numpy.__version__.split(".")[0])
    if major >= 2:
        problems.append(
            "NumPy is {} and Kuiva runs on the 1.x series (numpy>=1.22,<2). NumPy 2 changes "
            "promotion rules and the pinned PySCF is built against 1.x."
            .format(numpy.__version__))
    else:
        notes.append("numpy  {}".format(numpy.__version__))

try:
    import scipy
except ImportError as exc:
    problems.append("SciPy is missing ({}). Install scipy>=1.6.".format(exc))
else:
    notes.append("scipy  {}".format(scipy.__version__))

try:
    import h5py
except ImportError as exc:
    problems.append("h5py is missing ({}). Install h5py>=3; checkpoints are HDF5."
                    .format(exc))
else:
    notes.append("h5py   {}".format(h5py.__version__))

try:
    import pyscf
except ImportError as exc:
    problems.append("PySCF is missing ({}). Install pyscf==2.14.0; it is the front-end that "
                    "produces the scalar-relativistic starting orbitals and the integrals."
                    .format(exc))
else:
    notes.append("pyscf  {}".format(pyscf.__version__))
    if pyscf.__version__ != "2.14.0":
        notes.append("       ^ the pinned and tested version is 2.14.0; the X2C API has "
                     "changed across PySCF releases, so treat results with care")

# A working BLAS. NumPy without one is not a theoretical worry: a partially built or
# mis-linked install imports and then produces wrong or catastrophically slow linear algebra,
# and every kernel in Kuiva is a GEMM underneath.
if numpy is not None and not any(p.startswith("NumPy is ") for p in problems):
    try:
        a = numpy.arange(16.0).reshape(4, 4) + 10.0 * numpy.eye(4)
        err = float(numpy.abs(a.dot(numpy.linalg.inv(a)) - numpy.eye(4)).max())
        z = a + 1j * numpy.arange(16.0).reshape(4, 4)
        w = numpy.linalg.eigvalsh(z.dot(z.conj().T))          # complex Hermitian: the one
        if not (err < 1e-8) or not numpy.isfinite(w).all():    # every solver here reduces to
            raise ValueError("a 4x4 inverse is wrong by {:.1e}".format(err))
    except Exception as exc:                                    # noqa: BLE001 - report anything
        problems.append("NumPy's linear algebra does not work ({}): the BLAS/LAPACK it links "
                        "is missing or broken.".format(exc))

if problems:
    sys.stderr.write("\n")
    for p in problems:
        sys.stderr.write("[kuiva ERROR] {}\n".format(p))
    sys.stderr.write("\n[kuiva] The whole set at once:\n"
                     "            pip install 'numpy>=1.22,<2' 'scipy>=1.6' 'h5py>=3' "
                     "'pyscf==2.14.0'\n"
                     "        or, from a checkout of Kuiva:  pip install .\n\n")
    sys.exit(1)

print("KUIVA_PROBE_PREFIX={}".format(sys.prefix))
print("KUIVA_PROBE_PYVER={}.{}.{}".format(*sys.version_info[:3]))
for n in notes:
    print("KUIVA_PROBE_NOTE={}".format(n))
PYPROBE
}

# --- Memory limit ----------------------------------------------------------
# Mirrors kuiva/util/resources.py's search path. Kuiva has no built-in memory limit and
# refuses to start without one, on purpose: whoever knows the machine chooses the number.
_kuiva_memory_is_configured() {
    local root="$1" prefix="$2" candidate
    [ -n "${KUIVA_MEMORY_GB:-}" ] && return 0
    for candidate in \
        "${KUIVA_CONFIG:-}" \
        "${XDG_CONFIG_HOME:-${HOME}/.config}/kuiva/defaults.conf" \
        "/etc/kuiva/defaults.conf" \
        "${prefix:+${prefix}/etc/kuiva/defaults.conf}" \
        "${root}/defaults.conf"
    do
        [ -n "${candidate}" ] || continue
        [ -f "${candidate}" ] || continue
        # An uncommented memory_gb is what counts: a file may exist and configure only the
        # cache directory, and the limit is the one setting with no default.
        if grep -qE '^[[:space:]]*memory_gb[[:space:]]*=' "${candidate}"; then
            return 0
        fi
    done
    return 1
}

_kuiva_write_memory_config() {
    local target="$1" value="$2"
    mkdir -p "$(dirname "${target}")" || return 1
    cat > "${target}" <<EOF
# Kuiva site defaults. Written by setup.sh; edit freely.
#
# Search order, first match wins:
#   \$KUIVA_CONFIG
#   ~/.config/kuiva/defaults.conf          per user   <- this file
#   /etc/kuiva/defaults.conf               per machine
#   <sys.prefix>/etc/kuiva/defaults.conf   per install
#   <source tree>/defaults.conf
#
# Anything passed explicitly (memory_gb=..., \$KUIVA_MEMORY_GB) overrides this file.

[memory]
# Total working memory Kuiva may commit to its own arrays, in GB (1 GB = 2^30 B). This is
# NOT the machine's total RAM: leave room for the operating system, for the parts of a run
# Kuiva does not govern (the PySCF SCF, BLAS work buffers), and for everything else the
# machine has to do. Kuiva refuses a calculation whose plan exceeds this, before it starts.
memory_gb = ${value}

# Fraction of the limit above which an allocation warns while still proceeding.
warn_fraction = 0.7

# Downgrade every memory refusal to a warning. With this on, an over-large run fails as an
# out-of-memory kill instead of as a diagnosed refusal before it starts.
allow_overcommit = false

[amf]
# Persistent cache of the atomic mean-field spin-orbit corrections. A correction depends only
# on the element and its basis and configuration, never on geometry, so one entry serves every
# geometry and every job -- and a lanthanide's four-component atomic solve takes tens of
# minutes. Unset: \$XDG_CACHE_HOME/kuiva/amf, else ~/.cache/kuiva/amf. Set to "off" to
# disable. \$KUIVA_AMF_CACHE overrides it.
# amf_cache_dir = /scratch/\$USER/kuiva-amf

[scratch]
# Directory for scratch files (out-of-core arrays). Like memory_gb this has NO built-in
# default: any scratch use refuses until it is set here or with \$KUIVA_SCRATCH. Pick a real
# disk with room, never a tmpfs. Calculations that never touch scratch do not need it.
# scratch_dir = /scratch/\$USER
# scratch_gb = 100.0
EOF
}

# --- Scratch directory -----------------------------------------------------
# Same rule as the memory limit (user decision, 2026-08-26): no built-in default and no
# $TMPDIR/cwd fallback, because a guessed location lands on whatever filesystem happens to
# be there — a RAM-backed /tmp spends exactly the memory a spill exists to save. Any scratch
# *use* (a factor spill, environment paging) refuses until this is set; calculations that
# never touch scratch are unaffected, but it is asked for here, on install, exactly like the
# memory limit, so the refusal never lands mid-project.
_kuiva_scratch_is_configured() {
    local root="$1" prefix="$2" candidate
    [ -n "${KUIVA_SCRATCH:-}" ] && return 0
    for candidate in \
        "${KUIVA_CONFIG:-}" \
        "${XDG_CONFIG_HOME:-${HOME}/.config}/kuiva/defaults.conf" \
        "/etc/kuiva/defaults.conf" \
        "${prefix:+${prefix}/etc/kuiva/defaults.conf}" \
        "${root}/defaults.conf"
    do
        [ -n "${candidate}" ] || continue
        [ -f "${candidate}" ] || continue
        if grep -qE '^[[:space:]]*scratch_dir[[:space:]]*=' "${candidate}"; then
            return 0
        fi
    done
    return 1
}

_kuiva_write_scratch_config() {
    # Append scratch_dir to the per-user config, creating what is missing. Both templates
    # this script writes keep [scratch] as the last section, so a plain append lands inside
    # it; a hand-reordered file would need a manual edit and the parse error at the next
    # run says which file.
    local target="$1" value="$2"
    mkdir -p "$(dirname "${target}")" || return 1
    if [ ! -f "${target}" ]; then
        cat > "${target}" <<EOF
# Kuiva site defaults. Written by setup.sh; edit freely.

[scratch]
# Directory for scratch files (out-of-core arrays). No built-in default: any scratch use
# refuses until this is set. Pick a real disk with room, never a tmpfs.
scratch_dir = ${value}
EOF
        return $?
    fi
    if grep -qE '^\[scratch\]' "${target}"; then
        printf 'scratch_dir = %s\n' "${value}" >> "${target}"
    else
        printf '\n[scratch]\nscratch_dir = %s\n' "${value}" >> "${target}"
    fi
}

_kuiva_ask_for_scratch() {
    local root="$1" prefix="$2"
    local target="${XDG_CONFIG_HOME:-${HOME}/.config}/kuiva/defaults.conf"
    local answer="" fstype="" attempt=0

    if [ ! -t 0 ]; then
        _kuiva_err "no scratch directory is configured, and this shell cannot ask for one."
        _kuiva_err "Kuiva has no built-in scratch location (same rule as the memory limit):"
        _kuiva_err "a guessed directory lands on an unvetted filesystem. Set one of:"
        _kuiva_err "    export KUIVA_SCRATCH=<directory>       for this job"
        _kuiva_err "    ${target}"
        _kuiva_err "        [scratch]"
        _kuiva_err "        scratch_dir = <directory>"
        return 1
    fi

    printf '\n'
    _kuiva_say "No scratch directory is configured yet, and Kuiva never guesses one."
    _kuiva_say "It is where out-of-core arrays go (the three-index factor spill, and any"
    _kuiva_say "future paging). Pick a real disk with room for tens of GB — never a"
    _kuiva_say "RAM-backed tmpfs, which would spend the memory a spill exists to save."

    while [ "${attempt}" -lt 3 ]; do
        attempt=$((attempt + 1))
        printf '[kuiva] scratch directory: '
        read -r answer || answer=""
        if [ -z "${answer}" ]; then
            _kuiva_warn "expected a directory path, e.g. /scratch/${USER:-me}/kuiva"
            continue
        fi
        if ! mkdir -p "${answer}" 2>/dev/null || [ ! -w "${answer}" ]; then
            _kuiva_warn "cannot create or write to ${answer}"
            continue
        fi
        fstype="$(stat -f -c %T "${answer}" 2>/dev/null || true)"
        case "${fstype}" in
            tmpfs|ramfs)
                _kuiva_warn "${answer} is on a ${fstype} (RAM-backed) filesystem: a spill"
                _kuiva_warn "there spends the memory it exists to save. Accepted, but a"
                _kuiva_warn "real disk is what this setting is for."
                ;;
        esac
        if _kuiva_write_scratch_config "${target}" "${answer}"; then
            _kuiva_say "wrote ${target}  (scratch_dir = ${answer})"
            return 0
        fi
        _kuiva_err "could not write ${target}"
        return 1
    done
    _kuiva_err "no directory given; any scratch use will refuse until one is set."
    return 1
}

_kuiva_ask_for_memory() {
    local root="$1" prefix="$2"
    local target="${XDG_CONFIG_HOME:-${HOME}/.config}/kuiva/defaults.conf"
    local total="" answer="" attempt=0

    if [ ! -t 0 ]; then
        _kuiva_err "no memory limit is configured, and this shell cannot ask for one."
        _kuiva_err "Kuiva has no built-in default: it refuses to start rather than guess how"
        _kuiva_err "much of this machine it may use. Set one of:"
        _kuiva_err "    export KUIVA_MEMORY_GB=<gb>            for this job"
        _kuiva_err "    ${target}"
        _kuiva_err "        [memory]"
        _kuiva_err "        memory_gb = <gb>"
        return 1
    fi

    if [ -r /proc/meminfo ]; then
        total="$(awk '/^MemTotal:/ {printf "%.0f", $2 / 1048576}' /proc/meminfo 2>/dev/null)"
    fi

    printf '\n'
    _kuiva_say "No memory limit is configured yet, and Kuiva never guesses one."
    _kuiva_say "It is the total working memory Kuiva may commit to its own arrays, in GB."
    _kuiva_say "It is not the machine's RAM: leave room for the OS, for PySCF's SCF and for"
    _kuiva_say "whatever else runs here. A calculation whose plan exceeds it is refused"
    _kuiva_say "before it allocates anything."
    if [ -n "${total}" ]; then
        _kuiva_say "This machine reports about ${total} GB of RAM in total."
    fi

    while [ "${attempt}" -lt 3 ]; do
        attempt=$((attempt + 1))
        printf '[kuiva] memory limit in GB: '
        read -r answer || answer=""
        case "${answer}" in
            *[!0-9.]* | "" | .) ;;
            *)
                if [ "$(echo "${answer}" | awk '{print ($1 > 0) ? 1 : 0}')" = "1" ]; then
                    if _kuiva_write_memory_config "${target}" "${answer}"; then
                        _kuiva_say "wrote ${target}  (memory_gb = ${answer})"
                        return 0
                    fi
                    _kuiva_err "could not write ${target}"
                    return 1
                fi
                ;;
        esac
        _kuiva_warn "expected a positive number of GB, e.g. 8 or 12.5"
    done
    _kuiva_err "no limit given; Kuiva will refuse to start until one is set."
    return 1
}

# --- Main -----------------------------------------------------------------------------------
_kuiva_setup() {
    local root py fingerprint cache line prefix="" pyver="" cached=0
    root="$(_kuiva_root)"

    # 1. Developer sandbox, if it is there.
    if [ -f "${root}/external/env.sh" ]; then
        local had_u=0
        case "$-" in *u*) had_u=1 ;; esac
        set +u
        # shellcheck source=/dev/null
        . "${root}/external/env.sh"
        [ "${had_u}" -eq 1 ] && set -u
        py="$(command -v python 2>/dev/null || command -v python3 2>/dev/null)"
        prefix="$("${py}" -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
    else
        # 2. User install: find an interpreter and check it can run Kuiva.
        py="${KUIVA_PYTHON:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null)}"
        if [ -z "${py}" ]; then
            _kuiva_err "no python3 on PATH. Kuiva needs Python 3.9 or newer."
            return 1
        fi

        # The fingerprint is deliberately cheap: it must not import anything. The interpreter's
        # own path, size and timestamp change on every upgrade or venv switch, which is exactly
        # when the probe has to run again.
        local real stamp
        real="$(cd "$(dirname "${py}")" && pwd)/$(basename "${py}")"
        stamp="$(stat -Lc '%Y %s' "${py}" 2>/dev/null || echo unknown)"
        fingerprint="$(printf '%s|%s|%s|%s' "${real}" "${stamp}" "${VIRTUAL_ENV:-}" \
                       "${_KUIVA_REQUIRES}" | sha256sum | cut -c1-32)"
        cache="${XDG_CACHE_HOME:-${HOME}/.cache}/kuiva/setup/${fingerprint}"

        if [ -f "${cache}" ]; then
            cached=1
        else
            local probe
            if ! probe="$(_kuiva_probe "${py}")"; then
                _kuiva_err "this interpreter cannot run Kuiva: ${py}"
                return 1
            fi
            mkdir -p "$(dirname "${cache}")" 2>/dev/null
            printf '%s\n' "${probe}" > "${cache}" 2>/dev/null || cache=""
            _kuiva_say "checked ${py}"
            printf '%s\n' "${probe}" | sed -n 's/^KUIVA_PROBE_NOTE=/          /p'
        fi

        if [ -n "${cache}" ] && [ -f "${cache}" ]; then
            while IFS= read -r line; do
                case "${line}" in
                    KUIVA_PROBE_PREFIX=*) prefix="${line#KUIVA_PROBE_PREFIX=}" ;;
                    KUIVA_PROBE_PYVER=*)  pyver="${line#KUIVA_PROBE_PYVER=}" ;;
                esac
            done < "${cache}"
        fi
    fi

    # 3. This tree is the one that runs. An installed copy shadowing the working tree is a
    #    silent way to test something other than what is being edited.
    case ":${PYTHONPATH:-}:" in
        *":${root}:"*) ;;
        *) export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac

    # 4. A memory limit and a scratch directory, once, ever — the two settings with no
    #    built-in default, asked for together so neither refusal lands mid-project.
    if ! _kuiva_memory_is_configured "${root}" "${prefix}"; then
        _kuiva_ask_for_memory "${root}" "${prefix}" || return 1
    fi
    if ! _kuiva_scratch_is_configured "${root}" "${prefix}"; then
        _kuiva_ask_for_scratch "${root}" "${prefix}" || return 1
    fi

    if [ "${cached}" -eq 1 ]; then
        _kuiva_say "ready (python ${pyver:-?} at ${py})"
    else
        _kuiva_say "ready"
    fi
    return 0
}

_kuiva_setup
_kuiva_status=$?
unset -f _kuiva_setup _kuiva_probe _kuiva_root _kuiva_say _kuiva_warn _kuiva_err \
         _kuiva_memory_is_configured _kuiva_ask_for_memory _kuiva_write_memory_config \
         _kuiva_scratch_is_configured _kuiva_ask_for_scratch _kuiva_write_scratch_config
unset _KUIVA_REQUIRES
return ${_kuiva_status} 2>/dev/null || true
