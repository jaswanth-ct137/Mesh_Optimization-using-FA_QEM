# Patches libwebp's cmake/cpu.cmake so it builds with SIMD off under clang and clang-cl on x86_64:
# - clang-cl falls through to the gcc branch; treat it like cl.exe.
# - SIMD off adds -mno-sse2 to every file, and x86_64 clang cannot return doubles without SSE2.
#   SSE2 is baseline on x86_64, so the flag is dropped. Other targets reject it anyway.
#   cmake -DFILE=<libwebp>/cmake/cpu.cmake -P patch_libwebp.cmake
file(READ "${FILE}" src)
string(REPLACE [[if(MSVC AND CMAKE_C_COMPILER_ID STREQUAL "MSVC")]] [[if(MSVC)]] src "${src}")
string(REPLACE [[-mno-sse4.1;-mno-sse2;]] [[-mno-sse4.1;;]] src "${src}")
file(WRITE "${FILE}" "${src}")
