# Copied unchanged from ../cpp/CMakeLists.txt (FA-QEM engine), except that the Python module is off
# by default. Keep the compiler flags exactly as they are: they make the results bit-identical.

cmake_minimum_required(VERSION 3.20)
project(faqem_cpp C CXX)

# Two build modes:
#   FAQEM_PYTHON_PARITY=OFF (default)  portable: bit-identical results on macOS, Linux and Windows
#                                      (correctly rounded math, fixed-order linear algebra, pinned
#                                      codecs from cpp/deps). Very close to, not bit-equal to, Python.
#   FAQEM_PYTHON_PARITY=ON             macOS arm64 only: bit-identical to the Python reference here
#                                      (replays numpy's Apple Accelerate / libm calls).
option(FAQEM_PYTHON_PARITY "Bit-exact parity with the Python reference (macOS arm64 only)" OFF)
set(FAQEM_DEPS "" CACHE PATH "Install prefix of cpp/deps (portable build)")
option(FAQEM_PYTHON_MODULE "Build the faqem_cpp Python module" OFF)

if(FAQEM_PYTHON_PARITY AND NOT (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64"))
  message(FATAL_ERROR "FAQEM_PYTHON_PARITY needs macOS arm64 (it replays numpy's Apple Accelerate calls)")
endif()
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  message(FATAL_ERROR "Build with clang/clang-cl or gcc (the correctly rounded math library needs them); "
                      "on Windows: cmake -T ClangCL ... or -DCMAKE_CXX_COMPILER=clang-cl")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Exactness: no FMA contraction, no fast-math, pow() never folded. clang-cl takes clang options
# through /clang:.
if(CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
  set(CLANG_OPT "/clang:")
  add_compile_options(/fp:precise /EHsc)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  set(CLANG_OPT "")
  add_compile_options(-Wall -Wno-unused-function)
endif()
add_compile_options(${CLANG_OPT}-ffp-contract=off ${CLANG_OPT}-fno-fast-math ${CLANG_OPT}-fno-builtin-pow)
if(CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")
  set(CMAKE_C_FLAGS_RELEASE "/O2")
elseif(FAQEM_PYTHON_PARITY)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=apple-m2")
  set(CMAKE_C_FLAGS_RELEASE "-O3")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")  # baseline ISA: no -march=native
  set(CMAKE_C_FLAGS_RELEASE "-O3")
endif()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

set(FAQEM_SOURCES
  src/compat/npcompat.cpp
  src/compat/nprandom.cpp
  src/compat/csgraph.cpp
  src/compat/fmath.cpp
  src/bvh.cpp
  src/prep.cpp
  src/core.cpp
  src/metrics.cpp
  src/search.cpp
  src/virtual_edges.cpp
  src/io/image.cpp
  src/io/gltf_load.cpp
  src/io/asset.cpp
  src/io/simple_load.cpp
  src/io/objply_load.cpp
  src/io/export.cpp
  src/bake.cpp
  src/pipeline.cpp
  src/digest.cpp
  src/parallel.cpp
)
if(FAQEM_PYTHON_PARITY)
  list(APPEND FAQEM_SOURCES src/compat/accelerate.cpp)
else()
  list(APPEND FAQEM_SOURCES src/compat/linalg_portable.cpp)
endif()

# scipy's cKDTree. Parity: compiled like the scipy wheel (clang defaults: -O3, FP contraction on).
# Portable: the engine's own exact flags plus stable partitioning (see third_party/ckdtree/build.cxx).
add_library(ckdtree STATIC third_party/ckdtree/build.cxx third_party/ckdtree/query.cxx)
target_include_directories(ckdtree PUBLIC third_party/ckdtree)
target_compile_definitions(ckdtree PRIVATE NDEBUG)
if(FAQEM_PYTHON_PARITY)
  target_compile_options(ckdtree PRIVATE -O3 -ffp-contract=on -w)
else()
  target_compile_definitions(ckdtree PRIVATE FAQEM_PORTABLE=1)
  if(NOT CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    target_compile_options(ckdtree PRIVATE -w)
  endif()
endif()

add_library(faqem STATIC ${FAQEM_SOURCES})
target_include_directories(faqem PUBLIC src)
target_compile_definitions(faqem PUBLIC FAQEM_PYTHON_PARITY=$<BOOL:${FAQEM_PYTHON_PARITY}>)
target_link_libraries(faqem PUBLIC ckdtree)

if(FAQEM_PYTHON_PARITY)
  target_link_libraries(faqem PUBLIC "-framework Accelerate")
  # the codecs Pillow 12.3 bundles (libjpeg-turbo 3.x, libwebp 1.6), Draco 1.5.7 as DracoPy: Homebrew
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(CODECS REQUIRED IMPORTED_TARGET libpng libturbojpeg libwebp)
  find_package(nlohmann_json CONFIG REQUIRED)
  find_package(draco CONFIG REQUIRED)
  target_link_libraries(faqem PUBLIC PkgConfig::CODECS nlohmann_json::nlohmann_json draco::draco)
else()
  # correctly rounded binary64 functions (CORE-MATH)
  set(CM third_party/core-math)
  add_library(coremath STATIC ${CM}/atan2/atan2.c ${CM}/acos/acos.c ${CM}/cos/cos.c ${CM}/pow/pow.c)
  target_include_directories(coremath PRIVATE ${CM}/atan2 ${CM}/pow)
  if(NOT CMAKE_C_SIMULATE_ID STREQUAL "MSVC")
    target_compile_options(coremath PRIVATE -w)
  endif()
  target_link_libraries(faqem PUBLIC coremath)
  if(CMAKE_C_SIMULATE_ID STREQUAL "MSVC")
    # CORE-MATH uses 128-bit integers; with clang-cl their helpers live in compiler-rt
    execute_process(COMMAND ${CMAKE_C_COMPILER} /clang:-print-libgcc-file-name /clang:--rtlib=compiler-rt
                    OUTPUT_VARIABLE CLANG_RT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(CLANG_RT AND EXISTS "${CLANG_RT}")
      target_link_libraries(coremath PUBLIC "${CLANG_RT}")
    endif()
  endif()
  # pinned codecs from cpp/deps
  if(NOT FAQEM_DEPS)
    message(FATAL_ERROR "Portable build: build cpp/deps first and pass -DFAQEM_DEPS=<its install dir> "
                        "(see cpp/README.md)")
  endif()
  list(PREPEND CMAKE_PREFIX_PATH ${FAQEM_DEPS})
  if(NOT WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)  # link the pinned static libraries
  endif()
  find_path(TJ_INC turbojpeg.h PATHS ${FAQEM_DEPS}/include NO_DEFAULT_PATH REQUIRED)
  find_library(TJ_LIB NAMES turbojpeg-static turbojpeg PATHS ${FAQEM_DEPS}/lib NO_DEFAULT_PATH REQUIRED)
  find_library(PNG_LIB NAMES png16 libpng16_static libpng16 png16_static PATHS ${FAQEM_DEPS}/lib NO_DEFAULT_PATH REQUIRED)
  find_library(Z_LIB NAMES z zlibstatic zlib PATHS ${FAQEM_DEPS}/lib NO_DEFAULT_PATH REQUIRED)
  find_library(WEBP_LIB NAMES webp libwebp PATHS ${FAQEM_DEPS}/lib NO_DEFAULT_PATH REQUIRED)
  find_library(SHARPYUV_LIB NAMES sharpyuv libsharpyuv PATHS ${FAQEM_DEPS}/lib NO_DEFAULT_PATH)
  find_library(DRACO_LIB NAMES draco PATHS ${FAQEM_DEPS}/lib NO_DEFAULT_PATH REQUIRED)
  find_package(nlohmann_json CONFIG REQUIRED PATHS ${FAQEM_DEPS} NO_DEFAULT_PATH)
  target_include_directories(faqem PUBLIC ${FAQEM_DEPS}/include)
  target_link_libraries(faqem PUBLIC ${TJ_LIB} ${PNG_LIB} ${Z_LIB} ${WEBP_LIB} nlohmann_json::nlohmann_json ${DRACO_LIB})
  if(SHARPYUV_LIB)
    target_link_libraries(faqem PUBLIC ${SHARPYUV_LIB})
  endif()
endif()
find_package(Threads REQUIRED)
target_link_libraries(faqem PUBLIC Threads::Threads)

add_executable(faqem_cli cli/main.cpp)
set_target_properties(faqem_cli PROPERTIES OUTPUT_NAME faqem)
target_link_libraries(faqem_cli PRIVATE faqem)

if(FAQEM_PYTHON_MODULE)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  # portable build: module faqem_cpp; python-parity build: faqem_cpp_parity (both can be loaded at once)
  if(FAQEM_PYTHON_PARITY)
    set(FAQEM_MODULE faqem_cpp_parity)
  else()
    set(FAQEM_MODULE faqem_cpp)
  endif()
  pybind11_add_module(${FAQEM_MODULE} python/bindings.cpp)
  target_compile_definitions(${FAQEM_MODULE} PRIVATE FAQEM_MODULE_NAME=${FAQEM_MODULE})
  target_link_libraries(${FAQEM_MODULE} PRIVATE faqem)
endif()
