cmake_minimum_required(VERSION 3.21)
project(faqem_cpp_glb_optimizer VERSION 2.0.0 LANGUAGES C CXX)

if(APPLE)
  enable_language(OBJC OBJCXX)  # OBJC: GLFW's .m sources
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The FA-QEM engine: an unchanged copy of our cpp/ engine (portable build). It sets its own exact
# floating-point flags for its targets only. Needs -DFAQEM_DEPS=<build-deps/install> (see build.sh).
add_subdirectory(engine)

find_package(OpenGL REQUIRED)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
  GIT_SHALLOW TRUE)

FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.9b
  GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(glfw imgui)

add_library(engine_bridge STATIC src/engine_bridge.cpp)
target_include_directories(engine_bridge PUBLIC include)
target_link_libraries(engine_bridge PUBLIC faqem)

add_executable(faqem_optimizer
  src/main.cpp
  src/file_dialog.mm
  src/viewer.cpp)

target_sources(faqem_optimizer PRIVATE
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl2.cpp)
target_include_directories(faqem_optimizer PRIVATE ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(faqem_optimizer PRIVATE engine_bridge glfw OpenGL::GL)

if(APPLE)
  target_compile_definitions(faqem_optimizer PRIVATE GL_SILENCE_DEPRECATION)
  target_link_libraries(faqem_optimizer PRIVATE "-framework Cocoa")
endif()

foreach(target engine_bridge faqem_optimizer)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>)
endforeach()

include(CTest)
if(BUILD_TESTING)
  add_executable(engine_smoke_tests tests/test_engine_smoke.cpp)
  target_link_libraries(engine_smoke_tests PRIVATE engine_bridge)
  add_test(NAME engine_smoke_test COMMAND engine_smoke_tests)

  add_executable(glb_roundtrip_tests tests/test_glb_roundtrip.cpp)
  target_link_libraries(glb_roundtrip_tests PRIVATE engine_bridge)
  add_test(NAME glb_roundtrip_test COMMAND glb_roundtrip_tests)
  foreach(target engine_smoke_tests glb_roundtrip_tests)
    target_compile_options(${target} PRIVATE -UNDEBUG)  # keep assert() in Release
  endforeach()

  # identical results to our engine: digests recorded by cpp/tests/golden_portable.json
  add_test(NAME golden_digest_test
    COMMAND ${CMAKE_COMMAND}
      -DFAQEM=$<TARGET_FILE:faqem_cli>
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/data/tripo3-wooden+dresser+3d+model.glb.glb
      "-DARGS=--auto Medium"
      -DGOLDEN=${CMAKE_CURRENT_SOURCE_DIR}/tests/data/golden_dresser_auto_medium.txt
      -DOUT=${CMAKE_CURRENT_BINARY_DIR}/golden_out
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/check_golden.cmake)
endif()
