"""Build the dashboard and ship it inside the package (TDD §10.1, ADR 0008).

Usage:
  python scripts/build_ui.py                 # npm ci + static export + copy to src/qlassifier/_ui/ + stamp
  python scripts/build_ui.py --skip-install  # same, reusing frontend/node_modules
  python scripts/build_ui.py --check         # no Node: the committed build matches the committed sources
  python scripts/build_ui.py --budget DIR    # gzip size of a fresh export (e.g. frontend/out) within budget

The stamp (`_ui/build-info.json`) holds a SHA-256 over the build inputs (tracked files of frontend/app,
components, lib, public and the config files) and a SHA-256 per shipped file. `--check` recomputes both and
also enforces the bundle budget (NFR-5: all `_ui` files <= 1.5 MB gzip). Standard library only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
OUT = FRONTEND / "out"
UI = ROOT / "src" / "qlassifier" / "_ui"
STAMP = "build-info.json"
BUDGET_BYTES = 1_500_000
INPUTS = [
    "frontend/app",
    "frontend/components",
    "frontend/lib",
    "frontend/public",
    "frontend/package.json",
    "frontend/package-lock.json",
    "frontend/next.config.ts",
    "frontend/tsconfig.json",
    "frontend/postcss.config.mjs",
    "frontend/components.json",
]
FIX = "python scripts/build_ui.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> list[str]:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if line]


def source_files() -> list[str]:
    """Tracked build inputs, sorted. Untracked inputs are an error: CI would hash a different set."""
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *INPUTS)
    if untracked:
        sys.exit("Untracked build inputs (git add them, then rebuild):\n  " + "\n  ".join(untracked))
    files = sorted(git("ls-files", "--", *INPUTS))
    missing = [f for f in files if not (ROOT / f).is_file()]
    if missing:
        sys.exit("Tracked build inputs missing from the working tree:\n  " + "\n  ".join(missing))
    return files


def source_digest() -> str:
    digest = hashlib.sha256()
    for name in source_files():
        digest.update(name.encode() + b"\0" + sha256(ROOT / name).encode() + b"\n")
    return digest.hexdigest()


def shipped_files() -> dict[str, str]:
    return {
        path.relative_to(UI).as_posix(): sha256(path)
        for path in sorted(UI.rglob("*"))
        if path.is_file() and path.name != STAMP
    }


def gzip_size(directory: Path) -> int:
    return sum(len(gzip.compress(p.read_bytes(), 9)) for p in directory.rglob("*") if p.is_file())


def run(cmd: list[str]) -> None:
    env = {**os.environ, "NEXT_TELEMETRY_DISABLED": "1"}
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=FRONTEND, check=True, env=env)


def build(skip_install: bool) -> None:
    npm = shutil.which("npm")
    node = shutil.which("node")
    if not npm or not node:
        sys.exit("Node.js and npm are needed to build the dashboard (users never need them).")
    digest_before = source_digest()
    if not skip_install:
        run([npm, "ci", "--no-audit", "--no-fund"])
    # Always a clean build: Turbopack's persistent cache has served a stale CSS chunk after a source edit.
    for stale in (FRONTEND / ".next", OUT):
        if stale.exists():
            shutil.rmtree(stale)
    run([npm, "run", "build"])
    if not (OUT / "index.html").is_file():
        sys.exit('frontend/out/index.html missing: is `output: "export"` set in next.config.ts?')
    if UI.exists():
        shutil.rmtree(UI)
    shutil.copytree(OUT, UI)
    next_version = json.loads((FRONTEND / "node_modules/next/package.json").read_text())["version"]
    node_version = subprocess.run([node, "--version"], check=True, capture_output=True, text=True).stdout.strip()
    stamp = {
        "source_sha256": digest_before,
        "files": shipped_files(),
        "next": next_version,
        "node": node_version,
        "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (UI / STAMP).write_text(json.dumps(stamp, indent=1, sort_keys=True) + "\n")
    if source_digest() != digest_before:
        sys.exit("Build inputs changed during the build; run it again.")
    size = gzip_size(UI)
    print(f"wrote {UI.relative_to(ROOT)} ({len(stamp['files'])} files, {size / 1e6:.2f} MB gzip)")
    if size > BUDGET_BYTES:
        sys.exit(f"Bundle {size / 1e6:.2f} MB gzip exceeds the {BUDGET_BYTES / 1e6:.1f} MB budget (NFR-5).")


def check() -> int:
    stamp_path = UI / STAMP
    if not stamp_path.is_file():
        print(f"{stamp_path.relative_to(ROOT)} missing. Fix: {FIX}", file=sys.stderr)
        return 1
    stamp = json.loads(stamp_path.read_text())
    problems: list[str] = []
    if stamp.get("source_sha256") != source_digest():
        problems.append("dashboard sources changed since the committed build")
    expected: dict[str, str] = stamp.get("files", {})
    actual = shipped_files()
    for name in sorted(set(expected) - set(actual)):
        problems.append(f"missing file: {name}")
    for name in sorted(set(actual) - set(expected)):
        problems.append(f"unexpected file: {name}")
    for name in sorted(set(expected) & set(actual)):
        if expected[name] != actual[name]:
            problems.append(f"changed file: {name}")
    size = gzip_size(UI)
    if size > BUDGET_BYTES:
        problems.append(f"bundle {size / 1e6:.2f} MB gzip exceeds the {BUDGET_BYTES / 1e6:.1f} MB budget (NFR-5)")
    if problems:
        print("Committed dashboard build is stale or edited:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(f"Fix: {FIX}, then commit src/qlassifier/_ui/.", file=sys.stderr)
        return 1
    print(
        f"dashboard build matches its sources ({len(actual)} files, {size / 1e6:.2f} MB gzip, next {stamp.get('next')})"
    )
    return 0


def budget(directory: Path) -> int:
    if not (directory / "index.html").is_file():
        print(f"{directory} has no index.html: build the dashboard first.", file=sys.stderr)
        return 1
    size = gzip_size(directory)
    files = sum(1 for p in directory.rglob("*") if p.is_file())
    if size > BUDGET_BYTES:
        print(f"{directory}: {size / 1e6:.2f} MB gzip exceeds the {BUDGET_BYTES / 1e6:.1f} MB budget.", file=sys.stderr)
        return 1
    print(f"{directory}: {files} files, {size / 1e6:.2f} MB gzip (budget {BUDGET_BYTES / 1e6:.1f} MB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify the committed build (no Node needed)")
    mode.add_argument("--skip-install", action="store_true", help="reuse frontend/node_modules")
    mode.add_argument("--budget", type=Path, metavar="DIR", help="only check the gzip size of a built export")
    args = parser.parse_args()
    if args.check:
        return check()
    if args.budget is not None:
        return budget(args.budget.resolve())
    build(args.skip_install)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
