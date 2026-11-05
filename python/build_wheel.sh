#!/usr/bin/env bash
# Build the engine shared library and a platform wheel around it.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    make -f Makefile.win pydll
    lib=build/tinytorch_engine.dll
    plat=win_amd64
    ;;
  Darwin)
    make pylib
    lib=build/libtinytorch_engine.dylib
    plat=macosx_$(uname -m)
    ;;
  *)
    make pylib
    lib=build/libtinytorch_engine.so
    plat=linux_$(uname -m)
    ;;
esac

[ -f "$lib" ] || { echo "engine library missing: $lib" >&2; exit 1; }

dest="$root/python/tinyinference/bin"
mkdir -p "$dest"
cp -f "$lib" "$dest/"

# The engine links cudart dynamically; ship the redistributable next to it.
for dll in build/cudart64_*.dll; do
  if [ -f "$dll" ]; then cp -f "$dll" "$dest/"; fi
done

cd python
rm -rf dist build *.egg-info
python -m pip wheel . -w dist --no-deps --no-build-isolation
# pip no longer exposes --plat-name; retag with wheel's own tool so the
# bundled .dll/.so is not published as a pure-Python wheel.
python -m wheel tags --platform-tag "$plat" dist/tinyinference-*.whl >/dev/null
# wheel tags writes a copy; the pure-python original must not ship.
rm -f dist/tinyinference-*-py3-none-any.whl
echo "wheel written to python/dist"
ls dist
