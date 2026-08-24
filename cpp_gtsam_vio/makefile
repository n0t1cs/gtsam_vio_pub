# =============================================================================
# makefile
# VIO pipeline C++17 build
#
# Key fix: uses GTSAM's own bundled Eigen (not system Eigen) to match the
#          version GTSAM was compiled against.
#
# Usage:
#   make info               <- print detected paths WITHOUT compiling
#   make                    <- build into build/vio_pipeline
#   make GTSAM_PREFIX=/your/path
#   make run DATASET=/path/to/MH_01_easy
#   make clean
# =============================================================================

CXX     := g++
CXXSTD  := -std=c++17
OPT     := -O2 -g -Wall -Wno-deprecated-declarations -Wno-unused-variable -DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT
BUILDDIR := build
BIN     := $(BUILDDIR)/vio_pipeline

# --------------------------------------------------------------------------
# pkg-config: use full path to avoid make's restricted PATH
# --------------------------------------------------------------------------
PKGCFG := /usr/bin/pkg-config

pkg_cflags = $(shell $(PKGCFG) --cflags $(1) 2>/dev/null)
pkg_libs   = $(shell $(PKGCFG) --libs   $(1) 2>/dev/null)

# --------------------------------------------------------------------------
# GTSAM  (find this FIRST so its bundled Eigen takes priority)
# --------------------------------------------------------------------------
GTSAM_PREFIX ?= $(firstword $(wildcard /usr/local /usr))

GTSAM_INC_DIR := $(firstword $(wildcard \
  $(GTSAM_PREFIX)/include/gtsam \
  /usr/include/gtsam \
  /usr/local/include/gtsam))

ifneq ($(GTSAM_INC_DIR),)
  GTSAM_PARENT := $(patsubst %/gtsam,%,$(GTSAM_INC_DIR))
  GTSAM_INC    := -I$(GTSAM_PARENT)
else
  GTSAM_PARENT :=
  GTSAM_INC    :=
endif

# GTSAM bundles Eigen — use it to match the version GTSAM was built against
GTSAM_EIGEN := $(firstword $(wildcard \
  $(GTSAM_PARENT)/gtsam/3rdparty/Eigen \
  $(GTSAM_PARENT)/gtsam/3rdparty/eigen \
  $(GTSAM_PREFIX)/include/gtsam/3rdparty/Eigen \
  $(GTSAM_PREFIX)/include/gtsam/3rdparty/eigen))

ifneq ($(GTSAM_EIGEN),)
  EIGEN_INC := -I$(GTSAM_EIGEN)
  EIGEN_SRC := GTSAM bundled ($(GTSAM_EIGEN))
else
  EIGEN_FALLBACK := $(firstword $(wildcard \
    /usr/include/eigen3 \
    /usr/local/include/eigen3))
  EIGEN_INC := $(if $(EIGEN_FALLBACK),-I$(EIGEN_FALLBACK),)
  EIGEN_SRC := system ($(EIGEN_FALLBACK)) WARNING: may mismatch GTSAM build
endif

GTSAM_LIB_DIR := $(firstword $(wildcard \
  $(GTSAM_PREFIX)/lib \
  /usr/local/lib \
  /usr/lib/x86_64-linux-gnu))

GTSAM_LIBS := -L$(GTSAM_LIB_DIR) -Wl,-rpath,$(GTSAM_LIB_DIR) -lgtsam -lgtsam_unstable

TBB_LIBS := $(if $(wildcard \
  /usr/lib/x86_64-linux-gnu/libtbb.so \
  /usr/lib/x86_64-linux-gnu/libtbb.so.2 \
  /usr/local/lib/libtbb.so),-ltbb -ltbbmalloc,)

BOOST_LIBS := -lboost_system -lboost_filesystem -lboost_serialization -lboost_timer

# --------------------------------------------------------------------------
# OpenCV 4
# --------------------------------------------------------------------------
OPENCV_INC  := $(call pkg_cflags,opencv4)
OPENCV_LIBS := $(call pkg_libs,opencv4)
ifeq ($(strip $(OPENCV_INC)),)
  OPENCV_INC  := $(call pkg_cflags,opencv)
  OPENCV_LIBS := $(call pkg_libs,opencv)
endif
ifeq ($(strip $(OPENCV_INC)),)
  OPENCV_CANDIDATE := $(firstword $(wildcard \
    /usr/include/opencv4 \
    /usr/local/include/opencv4))
  ifneq ($(OPENCV_CANDIDATE),)
    OPENCV_INC  := -I$(OPENCV_CANDIDATE)
    OPENCV_LIBS := -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
                   -lopencv_calib3d -lopencv_features2d -lopencv_video
  endif
endif

# --------------------------------------------------------------------------
# yaml-cpp
# --------------------------------------------------------------------------
YAML_INC  := $(call pkg_cflags,yaml-cpp)
YAML_LIBS := $(call pkg_libs,yaml-cpp)
ifeq ($(strip $(YAML_LIBS)),)
  ifneq ($(wildcard \
    /usr/lib/x86_64-linux-gnu/libyaml-cpp*.so \
    /usr/local/lib/libyaml-cpp*.so),)
    YAML_LIBS := -lyaml-cpp
  endif
endif
ifeq ($(strip $(YAML_INC)),)
  YAML_INC_CANDIDATE := $(firstword $(wildcard \
    /usr/include/yaml-cpp \
    /usr/local/include/yaml-cpp))
  ifneq ($(YAML_INC_CANDIDATE),)
    YAML_INC := -I$(dir $(YAML_INC_CANDIDATE))
  endif
endif

# --------------------------------------------------------------------------
# Source files (exclude test_* files and build/ dir)
# --------------------------------------------------------------------------
SRC := $(shell find . -name "*.cpp" ! -name "test_*" ! -path "./$(BUILDDIR)/*" | sort)

# Mirror source tree under build/
OBJ := $(patsubst ./%,$(BUILDDIR)/%,$(SRC:.cpp=.o))

# Auto-include every directory that contains headers
HDR_DIRS := $(sort $(dir $(shell find . -name "*.h" ! -path "./$(BUILDDIR)/*" | sort)))
HDR_INC  := $(addprefix -I,$(HDR_DIRS))

# --------------------------------------------------------------------------
# Aggregated flags
# EIGEN_INC must appear BEFORE any system path that might shadow it
# --------------------------------------------------------------------------
ALL_INC := \
  $(HDR_INC) \
  $(EIGEN_INC) \
  $(OPENCV_INC) \
  $(YAML_INC) \
  $(GTSAM_INC)

ALL_LIBS := \
  $(GTSAM_LIBS) \
  $(TBB_LIBS) \
  $(BOOST_LIBS) \
  $(OPENCV_LIBS) \
  $(YAML_LIBS) \
  -lpthread \
  -lstdc++fs

# --------------------------------------------------------------------------
# Build rules
# --------------------------------------------------------------------------
.PHONY: all clean run info

all: $(BIN)

$(BIN): $(OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXSTD) $(OPT) $^ -o $@ $(ALL_LIBS)
	@echo ""
	@echo "Built: $@"

# Compile: mirror source tree under build/
$(BUILDDIR)/%.o: ./%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXSTD) $(OPT) $(ALL_INC) -c $< -o $@

clean:
	rm -rf $(BUILDDIR)

DATASET ?= ../MH_01_easy
run: $(BIN)
	./$(BIN) $(DATASET)

info:
	@echo ""
	@echo "=== pkg-config ==="
	@echo "  PKGCFG        : $(PKGCFG)"
	@echo "  exists        : $(shell test -f $(PKGCFG) && echo YES || echo NO)"
	@echo ""
	@echo "=== Eigen ==="
	@echo "  source        : $(EIGEN_SRC)"
	@echo "  EIGEN_INC     : $(EIGEN_INC)"
	@echo ""
	@echo "=== OpenCV ==="
	@echo "  OPENCV_INC    : $(OPENCV_INC)"
	@echo "  OPENCV_LIBS   : $(OPENCV_LIBS)"
	@echo ""
	@echo "=== yaml-cpp ==="
	@echo "  YAML_INC      : $(YAML_INC)"
	@echo "  YAML_LIBS     : $(YAML_LIBS)"
	@echo ""
	@echo "=== GTSAM ==="
	@echo "  GTSAM_PREFIX  : $(GTSAM_PREFIX)"
	@echo "  GTSAM_INC_DIR : $(GTSAM_INC_DIR)"
	@echo "  GTSAM_INC     : $(GTSAM_INC)"
	@echo "  GTSAM_EIGEN   : $(GTSAM_EIGEN)"
	@echo "  GTSAM_LIB_DIR : $(GTSAM_LIB_DIR)"
	@echo "  GTSAM_LIBS    : $(GTSAM_LIBS)"
	@echo "  TBB_LIBS      : $(TBB_LIBS)"
	@echo ""
	@echo "=== Sources ==="
	@for f in $(SRC); do echo "  $$f"; done
	@echo ""
	@echo "=== Objects (in build/) ==="
	@for f in $(OBJ); do echo "  $$f"; done
	@echo ""
	@echo "=== Full INC ==="
	@echo "  $(ALL_INC)"
	@echo ""
	@echo "=== Full LIBS ==="
	@echo "  $(ALL_LIBS)"
	@echo ""