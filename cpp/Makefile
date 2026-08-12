# Makefile — the compiled kernel backend, kuiva/_native<soabi>.so.
#
#     ./configure && make && make check
#
# This is the only build path for the extension; scripts/bootstrap/95_native.sh is a wrapper
# around it. The module is written in place next to the package and is git-ignored, so a
# rebuilt backend is picked up with no install step and a deleted one silently reverts every
# kernel to its NumPy implementation.
#
# ⚠ TWO LINKING RULES, asserted after every link rather than trusted:
#
#   1. ONE MKL. The extension links mkl_rt — the single runtime library NumPy has already
#      bound — and must never link its own threading layer (libmkl_sequential,
#      libmkl_intel_thread): two threading layers on one libmkl_core in one process is a
#      silent misconfiguration. Thread counts are set per region at run time instead.
#
#   2. ONE OpenMP RUNTIME: libiomp5, the one MKL uses. libgomp appearing here means a
#      non-Intel compiler crept in, and two OpenMP runtimes oversubscribe every core with no
#      error message. This is why there is no CXX override to be had past configure.
#
# The BUILD_ID compiled into the module is sha256(sources) + sha256(compiler version). A
# source-hash fingerprint of the Python tree cannot see a .so, so the .so carries its own, and
# that is what checkpoints and stage fingerprints record.

CONFIG := config.mk
ifeq ($(wildcard $(CONFIG)),)
$(error no config.mk — run ./configure first)
endif
include $(CONFIG)

SRCS     := $(sort $(wildcard *.cpp))
HDRS     := $(sort $(wildcard *.h *.hpp))
TARGET   := $(PKG_DIR)/_native$(EXT_SUFFIX)

SRC_HASH := $(shell cat $(SRCS) | sha256sum | cut -c1-16)
CXX_HASH := $(shell printf '%s' "$(CXX_VERSION)" | sha256sum | cut -c1-8)
BUILD_ID := $(SRC_HASH)-$(CXX_HASH)

CXXFLAGS := -std=c++20 $(OPTFLAGS) -fPIC -shared -qopenmp -fvisibility=hidden
INCLUDES := -I$(PY_INC) $(PB_INC) -I$(MKLROOT)/include
LIBS     := -L$(MKL_LIB) -lmkl_rt -Wl,-rpath,$(MKL_LIB)

.PHONY: all check clean print-config

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS) $(CONFIG)
	@echo "  CXX      $(notdir $@)  (build id $(BUILD_ID))"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DKUIVA_NATIVE_BUILD_ID='"$(BUILD_ID)"' \
	    $(SRCS) $(LIBS) -o $@
	@if ldd $@ | grep -q 'libgomp'; then \
	    echo "ERROR: linked libgomp — two OpenMP runtimes in one process (see the header)"; \
	    rm -f $@; exit 1; fi
	@if ldd $@ | grep -qE 'libmkl_(sequential|intel_thread|gnu_thread)'; then \
	    echo "ERROR: linked an MKL threading layer directly; it must link mkl_rt only"; \
	    rm -f $@; exit 1; fi
	@ldd $@ | grep -q 'libmkl_rt' || { echo "ERROR: did not link mkl_rt"; rm -f $@; exit 1; }
	@echo "  LINK     ok (mkl_rt, libiomp5, no libgomp)"

# Imports the extension, checks its interface version, and compares one production-shaped
# kernel against the NumPy implementation BITWISE. A backend that is merely close is a
# backend that moves committed numbers.
check: $(TARGET)
	@KUIVA_KERNELS=native PYTHONPATH=$(ROOT_DIR) $(PYTHON) check_native.py

clean:
	rm -f $(TARGET)

print-config:
	@echo "CXX        $(CXX)"
	@echo "CXX_VERSION $(CXX_VERSION)"
	@echo "OPTFLAGS   $(OPTFLAGS)"
	@echo "PYTHON     $(PYTHON)"
	@echo "TARGET     $(TARGET)"
	@echo "BUILD_ID   $(BUILD_ID)"
