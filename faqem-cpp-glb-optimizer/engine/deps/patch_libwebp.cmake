# libwebp only takes its MSVC branch for cl.exe. clang-cl falls through to the gcc branch, which adds
# -mno-sse2 to every file when SIMD is off, and x64 clang cannot return doubles without SSE2. Treat
# clang-cl like cl.exe. Non-MSVC compilers are unaffected.
#   cmake -DFILE=<libwebp>/cmake/cpu.cmake -P patch_libwebp.cmake
file(READ "${FILE}" src)
string(REPLACE [[if(MSVC AND CMAKE_C_COMPILER_ID STREQUAL "MSVC")]] [[if(MSVC)]] src "${src}")
file(WRITE "${FILE}" "${src}")
