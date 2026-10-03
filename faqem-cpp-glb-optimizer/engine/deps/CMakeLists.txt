# Pinned third-party libraries for the portable FA-QEM build, built from source with the same
# floating-point flags on every OS (static, FP contraction off, SIMD paths off), so image decoding
# and Draco decompression give the same bits on macOS, Linux and Windows.
#
#   cmake -S cpp/deps -B cpp/build-deps -G Ninja && cmake --build cpp/build-deps
#   -> installs into cpp/build-deps/install (pass it to the engine as -DFAQEM_DEPS=...)
cmake_minimum_required(VERSION 3.20)
project(faqem_deps C CXX)
include(ExternalProject)

set(PREFIX ${CMAKE_BINARY_DIR}/install)
if(MSVC OR CMAKE_C_SIMULATE_ID STREQUAL "MSVC")
  set(FP_FLAGS "/clang:-ffp-contract=off")
  if(NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set(FP_FLAGS "/fp:precise")
  endif()
else()
  set(FP_FLAGS "-ffp-contract=off -fno-fast-math")
endif()
set(COMMON
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX=${PREFIX}
  -DCMAKE_PREFIX_PATH=${PREFIX}
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  "-DCMAKE_C_FLAGS=${FP_FLAGS}"
  "-DCMAKE_CXX_FLAGS=${FP_FLAGS}"
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  -DBUILD_SHARED_LIBS=OFF)
if(CMAKE_OSX_ARCHITECTURES)
  list(APPEND COMMON "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}")
endif()
if(CMAKE_MSVC_RUNTIME_LIBRARY)
  list(APPEND COMMON "-DCMAKE_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}")
endif()

ExternalProject_Add(zlib
  URL https://github.com/madler/zlib/archive/refs/tags/v1.3.1.tar.gz
  CMAKE_ARGS ${COMMON} -DZLIB_BUILD_EXAMPLES=OFF)

ExternalProject_Add(libpng
  DEPENDS zlib
  URL https://github.com/pnggroup/libpng/archive/refs/tags/v1.6.58.tar.gz
  CMAKE_ARGS ${COMMON} -DPNG_SHARED=OFF -DPNG_STATIC=ON -DPNG_FRAMEWORK=OFF -DPNG_TESTS=OFF -DPNG_TOOLS=OFF
             -DPNG_HARDWARE_OPTIMIZATIONS=OFF -DZLIB_ROOT=${PREFIX})

ExternalProject_Add(libjpeg-turbo
  URL https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.1.4.1.tar.gz
  CMAKE_ARGS ${COMMON} -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DWITH_SIMD=OFF -DWITH_TURBOJPEG=ON
             -DWITH_TOOLS=OFF -DWITH_TESTS=OFF -DCMAKE_INSTALL_LIBDIR=lib)

ExternalProject_Add(libwebp
  URL https://github.com/webmproject/libwebp/archive/refs/tags/v1.6.0.tar.gz
  CMAKE_ARGS ${COMMON} -DWEBP_ENABLE_SIMD=OFF -DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF
             -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_GIF2WEBP=OFF -DWEBP_BUILD_IMG2WEBP=OFF
             -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_WEBPMUX=OFF
             -DWEBP_BUILD_EXTRAS=OFF -DWEBP_USE_THREAD=OFF -DCMAKE_INSTALL_LIBDIR=lib)

ExternalProject_Add(draco
  URL https://github.com/google/draco/archive/refs/tags/1.5.7.tar.gz
  CMAKE_ARGS ${COMMON} -DDRACO_TRANSCODER_SUPPORTED=OFF -DDRACO_JS_GLUE=OFF -DDRACO_TESTS=OFF
             -DCMAKE_INSTALL_LIBDIR=lib)

ExternalProject_Add(nlohmann_json
  URL https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz
  CMAKE_ARGS ${COMMON} -DJSON_BuildTests=OFF -DJSON_Install=ON)
