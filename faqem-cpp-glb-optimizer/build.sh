#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v cmake >/dev/null 2>&1; then
  echo "ERROR: CMake was not found in PATH."
  echo "On macOS, install it with: brew install cmake"
  exit 1
fi

BUILD_DIR="build"
CONFIG="Release"

# Pinned image/Draco libraries of the FA-QEM engine, built once from source (about 2 minutes).
# Set FAQEM_DEPS to reuse an existing install, e.g. ../cpp/build-deps/install.
if [[ -z "${FAQEM_DEPS:-}" ]]; then
  FAQEM_DEPS="$SCRIPT_DIR/build-deps/install"
  if [[ ! -d "$FAQEM_DEPS/lib" ]]; then
    echo "[0/3] Building the engine's pinned libraries (first time only)..."
    cmake -S engine/deps -B build-deps -DCMAKE_BUILD_TYPE=Release
    cmake --build build-deps --parallel
    echo
  fi
fi

echo "[1/3] Configuring FA-QEM C++ GLB Optimizer..."
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$CONFIG" -DFAQEM_DEPS="$FAQEM_DEPS"

echo
echo "[2/3] Building $CONFIG..."
cmake --build "$BUILD_DIR" --config "$CONFIG" --parallel

echo
echo "[3/3] Running tests..."
ctest --test-dir "$BUILD_DIR" -C "$CONFIG" --output-on-failure

APP="$BUILD_DIR/faqem_optimizer"
if [[ ! -x "$APP" ]]; then
  APP="$BUILD_DIR/$CONFIG/faqem_optimizer"
fi

echo
echo "Build completed successfully."
echo "Executable: $SCRIPT_DIR/$APP"
echo "Engine CLI: $SCRIPT_DIR/$BUILD_DIR/engine/faqem (same flags as our faqem CLI)"

if [[ "${1:-}" == "--build-only" ]]; then
  exit 0
fi

if [[ $# -eq 0 ]]; then
  echo
  echo "Opening viewer. Use 'Open model...' to choose a model."
  exec "$APP"
fi

INPUT="$1"
if [[ ! -f "$INPUT" ]]; then
  echo "ERROR: Input file does not exist: $INPUT" >&2
  exit 1
fi

echo
echo "Launching optimizer..."
echo "Input:  $INPUT"
shift
exec "$APP" "$INPUT" "$@"
