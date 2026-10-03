# Building the portable FA-QEM engine on Windows (x86_64)

The portable build is the default. It gives bit-identical results on Windows, Linux and macOS. It needs
**clang-cl**. MSVC's own `cl.exe` cannot compile the correctly rounded math library.

## 1. Install the tools

1. Visual Studio 2022, or the Build Tools, with these components:
   - "Desktop development with C++"
   - "C++ Clang tools for Windows" (installs clang-cl and LLVM)
2. CMake 3.20 or newer and Ninja: `winget install Kitware.CMake Ninja-build.Ninja`
3. Python 3.12: needed only to run the identity check.

## 2. Build

Open **"x64 Native Tools Command Prompt for VS 2022"** in the project folder and run:

```bat
cmake -S cpp\deps -B build-deps -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl
cmake --build build-deps
cmake -S cpp -B build -G Ninja -DFAQEM_DEPS=%CD%\build-deps\install -DFAQEM_PYTHON_MODULE=OFF ^
      -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl
cmake --build build
```

The first command line downloads and builds the pinned libraries: zlib, libpng, libjpeg-turbo, libwebp,
Draco and nlohmann-json.

## 3. Check that Windows gives the same bits as the Mac

```bat
python tools\portable_golden.py --faqem build\faqem.exe
```

A correct build ends with `all 36 runs identical (Windows AMD64)`.

The check needs the corpus files in the project folder:
- `checking copy\*.glb`
- `tripo3-wooden+dresser+3d+model.glb.glb`
- `samples\*\*.obj`

`--quick` runs a smaller subset.

## 4. Use

```bat
build\faqem.exe model.glb --auto High -o out
```

To use the Python module (for the app or TripoSG), configure with `-DFAQEM_PYTHON_MODULE=ON`. Python 3.12
and `pip install pybind11` are needed, and you pass
`-Dpybind11_DIR=<python -m pybind11 --cmakedir>`.

## Automatic check on GitHub

`.github/workflows/portable.yml` runs the same build and check on Windows, Linux (gcc and clang) and macOS
for every push. Push the project, including the corpus files above, to a GitHub repository and open the
"Actions" tab. Do not push `checking copy.zip` (255 MB): GitHub rejects files over 100 MB.
