# =============================================================================
# voxl-gtsam-vio: the online GTSAM VIO on ModalAI MPA pipes.
#
#   gvio_core              vendored estimator (estimator/) + transport-free online engine (src/engine)
#   gvio-engine-replay     the engine on a recording, in process (desktop test bench, no MPA)
#   voxl-gtsam-vio-server  the service: imu_apps + a tracking camera in, vio pipes out (needs MPA)
#   gvio-replay            a VOXL log or a dataset published as MPA pipes (needs MPA)
#
# Compile flags reproduce the offline estimator's build: -std=c++17 -O1 -g, GTSAM's bundled Eigen
# first, no GTSAM compile definitions. Same flags, same compiler and same GTSAM give the same
# floating point, which the parity test relies on. -O1 is also a known-good level: an -O2 build
# once produced a binary that segfaulted at init/vins_realign.cpp:53 (unresolved).
# =============================================================================
cmake_minimum_required(VERSION 3.10)
project(voxl-gtsam-vio C CXX)

set(GTSAM_PREFIX "/usr/local" CACHE PATH "GTSAM 4.2a9 install prefix")
set(MPA_ROOT "" CACHE PATH "root holding usr/include + usr/lib of libmodal_pipe and libmodal_json (empty: the system, as on VOXL)")
option(GVIO_WITH_MPA "build the MPA server and the pipe replay (needs libmodal_pipe)" ON)
option(GVIO_WITH_TBB "GTSAM was built with TBB: pin it to one thread in deterministic mode" ON)
set(GVIO_OPT "-O1 -g" CACHE STRING "optimisation flags (-O1 -g = the validated offline build)")

set(CMAKE_BUILD_TYPE "" CACHE STRING "" FORCE)        # flags come from GVIO_OPT only
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${GVIO_OPT} -Wall -Wno-deprecated-declarations -Wno-unused-variable -DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT")
find_package(Threads REQUIRED)

# ---- GTSAM 4.2a9 with its bundled Eigen (the makefile's EIGEN_INC precedes every system path)
set(GTSAM_EIGEN "${GTSAM_PREFIX}/include/gtsam/3rdparty/Eigen")
if(NOT EXISTS "${GTSAM_PREFIX}/include/gtsam/config.h")
  message(FATAL_ERROR "GTSAM not found under GTSAM_PREFIX=${GTSAM_PREFIX}")
endif()
file(STRINGS "${GTSAM_PREFIX}/include/gtsam/config.h" _gv REGEX "GTSAM_VERSION_STRING")
message(STATUS "GTSAM: ${_gv} at ${GTSAM_PREFIX}")
if(NOT _gv MATCHES "4\\.2a9")
  message(WARNING "GTSAM is not 4.2a9: the estimate will differ from the offline system")
endif()
find_library(GTSAM_LIB gtsam PATHS "${GTSAM_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(GTSAM_UNSTABLE_LIB gtsam_unstable PATHS "${GTSAM_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d video features2d)
message(STATUS "OpenCV ${OpenCV_VERSION}")
find_library(YAML_CPP_LIB yaml-cpp REQUIRED)
find_path(YAML_CPP_INC yaml-cpp/yaml.h REQUIRED)
set(BOOST_LIBS boost_system boost_filesystem boost_serialization boost_timer)
# VOXL2 build (scripts/build_voxl2.sh): GTSAM, Boost, yaml-cpp and OpenCV are static libraries in a
# staging prefix, so GTSAM's own static helpers and every Boost library it uses are linked by path.
option(GVIO_STATIC_DEPS "dependencies are static libraries (scripts/build_voxl2.sh)" OFF)
set(BOOST_LIB_DIR "" CACHE PATH "directory of the static Boost libraries (GVIO_STATIC_DEPS)")
if(GVIO_STATIC_DEPS)
  set(BOOST_LIBS "")
  foreach(b serialization filesystem timer thread chrono date_time regex program_options system)
    list(APPEND BOOST_LIBS "${BOOST_LIB_DIR}/libboost_${b}.a")
  endforeach()
  foreach(h metis-gtsam cephes-gtsam)
    find_library(_h_${h} ${h} PATHS "${GTSAM_PREFIX}/lib" NO_DEFAULT_PATH)
    if(_h_${h})
      list(APPEND GTSAM_UNSTABLE_LIB ${_h_${h}})
    endif()
  endforeach()
  list(APPEND BOOST_LIBS dl)
endif()
set(TBB_LIBS "")
if(GVIO_WITH_TBB)
  set(TBB_LIBS tbb tbbmalloc)
endif()
set(FS_LIB "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.1)
  set(FS_LIB stdc++fs)                                 # gcc 7/8 (VOXL2 Ubuntu 18.04): std::filesystem
endif()

# ---- gvio_core: the vendored estimator and the online engine
file(GLOB EST_SRC estimator/*.cpp estimator/*/*.cpp)
file(GLOB ENGINE_SRC src/engine/*.cpp)
add_library(gvio_core STATIC ${EST_SRC} ${ENGINE_SRC})
target_include_directories(gvio_core PUBLIC
  estimator estimator/factors estimator/frontend estimator/init estimator/utils src/engine
  ${GTSAM_EIGEN} ${OpenCV_INCLUDE_DIRS} ${YAML_CPP_INC} ${GTSAM_PREFIX}/include)
if(GVIO_WITH_TBB)
  target_compile_definitions(gvio_core PUBLIC GVIO_WITH_TBB)
endif()
target_link_libraries(gvio_core PUBLIC
  ${GTSAM_LIB} ${GTSAM_UNSTABLE_LIB} ${TBB_LIBS} ${BOOST_LIBS} ${OpenCV_LIBS} ${YAML_CPP_LIB}
  Threads::Threads ${FS_LIB})
set(CMAKE_BUILD_RPATH "${GTSAM_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH "${GTSAM_PREFIX}/lib")

add_executable(gvio-engine-replay tools/gvio_engine_replay.cpp)
target_link_libraries(gvio-engine-replay gvio_core)
add_executable(gvio-publish-check tools/gvio_publish_check.cpp)
target_link_libraries(gvio-publish-check gvio_core)
install(TARGETS gvio-engine-replay RUNTIME DESTINATION bin)

# ---- MPA: the server and the pipe replay
if(GVIO_WITH_MPA)
  set(_mpa_hint "")
  if(MPA_ROOT)
    set(_mpa_hint "${MPA_ROOT}/usr")
  endif()
  find_path(MPA_INC modal_pipe.h HINTS ${_mpa_hint}/include)
  find_library(MPA_PIPE_LIB modal_pipe HINTS ${_mpa_hint}/lib ${_mpa_hint}/lib64)
  find_library(MPA_JSON_LIB modal_json HINTS ${_mpa_hint}/lib ${_mpa_hint}/lib64)
  if(MPA_INC AND MPA_PIPE_LIB AND MPA_JSON_LIB)
    message(STATUS "libmodal_pipe: ${MPA_PIPE_LIB}")
    file(GLOB SERVER_SRC src/server/*.cpp)
    add_executable(voxl-gtsam-vio-server ${SERVER_SRC})
    target_include_directories(voxl-gtsam-vio-server PRIVATE src/server ${MPA_INC})
    target_link_libraries(voxl-gtsam-vio-server gvio_core ${MPA_PIPE_LIB} ${MPA_JSON_LIB})
    add_executable(gvio-replay tools/gvio_replay.cpp)
    target_include_directories(gvio-replay PRIVATE ${MPA_INC})
    target_link_libraries(gvio-replay gvio_core ${MPA_PIPE_LIB} ${MPA_JSON_LIB})
    if(_mpa_hint)
      set_target_properties(voxl-gtsam-vio-server gvio-replay PROPERTIES BUILD_RPATH "${GTSAM_PREFIX}/lib;${_mpa_hint}/lib")
    endif()
    install(TARGETS voxl-gtsam-vio-server gvio-replay RUNTIME DESTINATION bin)
    install(FILES config/voxl-gtsam-vio-server.yaml DESTINATION /etc/modalai)
    install(FILES config/estimator_mantis2.yaml DESTINATION share/voxl-gtsam-vio)
    install(FILES service/voxl-gtsam-vio-server.service DESTINATION /etc/systemd/system)
  else()
    message(WARNING "libmodal_pipe/libmodal_json not found (MPA_ROOT='${MPA_ROOT}'): building the engine and "
                    "gvio-engine-replay only. On a desktop run scripts/build_mpa_x86.sh first.")
  endif()
endif()
