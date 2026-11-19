"""Locate, check and record the external checkouts used by the Python benchmarks.

The direct Numba comparator and the shared benchmark fixture live on the
``faster_mobility`` branch of RigidMultiblobsWall
(https://github.com/annabroms/RigidMultiblobsWall, GPL-3.0). The benchmarks
read that checkout in place; neither the ``blake`` package nor its tests import
it. Run this file directly to check a checkout without benchmarking::

    python benchmarks/rigid_multiblobs.py [--rigid-root PATH]
"""

import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import sys

URL = "https://github.com/annabroms/RigidMultiblobsWall"
BRANCH = "faster_mobility"
# Commit on BRANCH whose kernel and fixture the documented measurements used.
TESTED_COMMIT = "799fbeca3d4715c4cdba7ba8bfd4dd7cd97b42e7"
KERNEL = Path("mobility/blake_numba.py")
DRIVER = Path("faster_mobility/blakeComparison.py")
REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPOSITORY.parent / "RigidMultiblobsWall"


def clone_hint(root):
    """Return the commands that create a matching checkout at root."""
    return (f"  git clone --branch {BRANCH} {URL}.git {root}\n"
            f"or, in an existing clone:\n"
            f"  git -C {root} fetch origin && git -C {root} switch {BRANCH}")


def check_root(root, required=(KERNEL, DRIVER)):
    """Return the resolved checkout, or exit with clone instructions if files are missing."""
    root = Path(root).resolve()
    missing = [str(root / name) for name in required if not (root / name).is_file()]
    if missing:
        raise SystemExit(
            f"This benchmark needs a RigidMultiblobsWall checkout of branch '{BRANCH}' "
            f"({URL}).\nMissing:\n  " + "\n  ".join(missing)
            + f"\nCreate one with:\n{clone_hint(root)}\n"
            "or pass the location of an existing checkout with --rigid-root.")
    return root


def git_record(path):
    """Record a checkout's branch, commit and uncommitted-change state, or None outside git."""
    def run(*arguments):
        """Run one read-only git query in the checkout."""
        return subprocess.run(["git", "-C", str(path), *arguments], capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return dict(branch=run("rev-parse", "--abbrev-ref", "HEAD"),
                    commit=run("rev-parse", "HEAD"),
                    dirty=bool(run("status", "--porcelain", "--untracked-files=no")))
    except (OSError, subprocess.CalledProcessError):
        return None


def describe_checkout(root):
    """Record the checkout and warn, without failing, when it differs from the tested commit."""
    record = dict(root=str(root), url=URL, expected_branch=BRANCH, tested_commit=TESTED_COMMIT,
                  git=git_record(root))
    git = record["git"]
    if git is None:
        print(f"Warning: {root} is not a git checkout; cannot confirm branch '{BRANCH}'.",
              file=sys.stderr, flush=True)
    elif git["commit"] != TESTED_COMMIT or git["dirty"]:
        print(f"Warning: RigidMultiblobsWall is at {git['branch']} {git['commit'][:7]}"
              f"{' with uncommitted changes' if git['dirty'] else ''}; documented results used "
              f"{BRANCH} {TESTED_COMMIT[:7]}. The commit is recorded in the report.",
              file=sys.stderr, flush=True)
    return record


def cpu_model():
    """Return a CPU description on Linux, macOS or any other platform."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    sysctl = shutil.which("sysctl")
    if sysctl:
        result = subprocess.run([sysctl, "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return platform.processor() or platform.machine()


def require_fmm_threads(openmp, threads, expected_extensions=2):
    """Exit with build guidance unless every fmm3dpy extension reports the requested threads."""
    extensions = openmp["extensions"]
    reported = [item.get("openmp_max_threads") for item in extensions]
    serial = all("openmp_max_threads" not in item for item in extensions)
    if len(extensions) == expected_extensions and (
            all(value == threads for value in reported) or (threads == 1 and serial)):
        return
    names = [Path(item["file"]["path"]).name for item in extensions]
    raise SystemExit(
        f"fmm3dpy extensions {names} report OpenMP thread limits {reported}; expected "
        f"{expected_extensions} extensions reporting {threads}. A serial fmm3dpy build "
        "(limit None) is accepted only with --threads 1. Build OpenMP bindings as described "
        "in docs/dependencies.md#python-bindings-with-openmp.")


def _self_test():
    """Check path handling and the thread check without importing numerical libraries."""
    assert check_root(REPOSITORY, required=(Path("README.md"),)) == REPOSITORY
    try:
        check_root(REPOSITORY / "does-not-exist")
    except SystemExit as stop:
        assert BRANCH in str(stop) and "git clone" in str(stop)
    else:
        raise AssertionError("missing checkout was accepted")
    assert git_record(REPOSITORY.parent / "does-not-exist") is None
    assert isinstance(cpu_model(), str)
    record = lambda value: dict(file=dict(path=f"x{value}.so"), **(
        {} if value is None else dict(openmp_max_threads=value)))
    require_fmm_threads(dict(extensions=[record(8), record(8)]), 8)
    require_fmm_threads(dict(extensions=[record(None), record(None)]), 1)
    for extensions, threads in (([record(None), record(None)], 8), ([record(8), record(4)], 8),
                                ([record(8)], 8)):
        try:
            require_fmm_threads(dict(extensions=extensions), threads)
        except SystemExit:
            continue
        raise AssertionError(f"accepted {extensions} for {threads} threads")
    print("rigid_multiblobs self-test passed")


def main():
    """Self-test, then report whether a RigidMultiblobsWall checkout is ready for benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rigid-root", type=Path, default=DEFAULT_ROOT,
                        help="RigidMultiblobsWall checkout (default: %(default)s)")
    args = parser.parse_args()
    _self_test()
    root = check_root(args.rigid_root)
    record = describe_checkout(root)
    print(f"RigidMultiblobsWall checkout ready: {root} {record['git']}")


if __name__ == "__main__":
    main()
