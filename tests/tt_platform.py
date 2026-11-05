"""Small platform shim for the Python gates.

Keeps Linux behavior identical while making the same scripts usable on
Windows (MSYS2/mingw-w64 builds): DLL suffix, .exe suffix, PATH-based DLL
lookup instead of LD_LIBRARY_PATH, and a writable temp directory.
"""
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
IS_WIN = os.name == "nt"


def lib(name):
    """Absolute path of a build shared library (name without suffix)."""
    return os.path.join(BUILD, name + (".dll" if IS_WIN else ".so"))


def exe(name):
    """Absolute path of a build executable (name with or without suffix)."""
    if IS_WIN and not name.lower().endswith(".exe"):
        name += ".exe"
    return os.path.join(BUILD, name)


def tmp(name):
    return os.path.join(tempfile.gettempdir(), name)


def env_with_lib(extra=None):
    """Environment whose loader path includes the build directory."""
    env = dict(os.environ)
    extras = extra if isinstance(extra, (list, tuple)) else [extra]
    dirs = [d for d in extras if d] + [BUILD]
    if IS_WIN:
        env["PATH"] = os.pathsep.join(dirs + [env.get("PATH", "")])
    else:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            dirs + [env.get("LD_LIBRARY_PATH", "")])
    return env


def _msys_dirs():
    roots = [os.environ.get("MSYS2_ROOT"), r"C:\msys64",
             os.path.expanduser(r"~\msys64")]
    dirs = []
    for root in roots:
        if root:
            dirs += [os.path.join(root, "mingw64", "bin"),
                     os.path.join(root, "usr", "bin")]
    return dirs


MSYS2_DIRS = _msys_dirs()
if IS_WIN:
    _present = [d for d in MSYS2_DIRS if os.path.isdir(d)]
    os.environ["PATH"] = os.pathsep.join(
        _present + [os.environ.get("PATH", "")])


def find_tool(name):
    """Locate a tool on PATH, falling back to the default MSYS2 install."""
    p = shutil.which(name)
    if p:
        return p
    if IS_WIN:
        for d in MSYS2_DIRS:
            q = os.path.join(d, name + ".exe")
            if os.path.isfile(q):
                return q
    return name


def make_cuda_cmd():
    if IS_WIN:
        # build_windows.sh sources the MSVC/SDK/CUDA environment first.
        return [find_tool("bash"), "scripts/build_windows.sh", "cuda"]
    return ["make", "-s", "cuda"]


def find_cc():
    env_cc = os.environ.get("TT_CC")
    if env_cc:
        return env_cc
    for name in ("cc", "gcc", "clang"):
        p = shutil.which(name)
        if p:
            return p
    if IS_WIN:
        for d in MSYS2_DIRS:
            for name in ("gcc", "clang"):
                q = os.path.join(d, name + ".exe")
                if os.path.isfile(q):
                    return q
    raise RuntimeError("no C compiler found (cc/gcc/clang)")


def utf8_stdout():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
