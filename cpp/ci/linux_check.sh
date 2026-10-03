#!/bin/sh
# Build the portable engine on Linux and check it against the golden digests recorded on macOS.
#   docker run --rm --platform linux/amd64 -v "$PWD":/src:ro faqem-linux /src/cpp/ci/linux_check.sh [gcc|clang] [--quick]
set -e
CC_KIND=${1:-gcc}
shift || true
if [ "$CC_KIND" = clang ]; then export CC=clang CXX=clang++; else export CC=gcc CXX=g++; fi
B=/tmp/faqem-$CC_KIND
echo "== $(uname -m) $($CXX --version | head -1)"
# under x86 emulation on Apple Silicon (Rosetta) gcc occasionally crashes; a resumed build only
# redoes the file that failed, so retry a few times
build() { for i in 1 2 3 4 5; do cmake --build "$1" -- -j2 >/dev/null 2>&1 && return 0; echo "  (retrying build of $1)"; done; cmake --build "$1"; }
cmake -S /src/cpp/deps -B $B/deps -G Ninja -Wno-dev >/dev/null
build $B/deps
cmake -S /src/cpp -B $B/eng -G Ninja -DFAQEM_DEPS=$B/deps/install -DFAQEM_PYTHON_MODULE=OFF >/dev/null
build $B/eng
python3 /src/tools/portable_golden.py --faqem $B/eng/faqem "$@"
