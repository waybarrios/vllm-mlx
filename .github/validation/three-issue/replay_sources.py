#!/usr/bin/env python3
"""Materialize pinned archives and verify the complete frozen source manifests."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile


BUNDLE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_bundle():
    for line in (BUNDLE / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = BUNDLE / relative
        assert path.resolve().is_relative_to(BUNDLE.resolve()), relative
        assert digest(path) == expected, relative


def verify_tree(root, expected, *, allow_runtime_artifacts=False):
    observed = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() or path.is_symlink()}
    extra = observed - set(expected)
    if allow_runtime_artifacts:
        extra = {name for name in extra if not any(part in {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"} for part in Path(name).parts) and name not in {".coverage", "coverage.xml"}}
    assert not (set(expected) - observed) and not extra, {"missing": sorted(set(expected) - observed), "extra": sorted(extra)}
    for relative, row in expected.items():
        path = root / relative
        actual = hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest() if path.is_symlink() else digest(path)
        assert actual == row["sha256"], relative
        if row.get("untracked"):
            assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(row["mode"]), relative
        elif not path.is_symlink():
            assert bool(path.stat().st_mode & stat.S_IXUSR) == (row["git_mode"] == "100755"), relative


def materialize(repo, run_root):
    verify_bundle()
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    assert run_root.is_relative_to(repo) and not run_root.exists(), "Use a fresh project-local run root"
    run_root.mkdir(parents=True)
    temporary = run_root / "tmp"
    temporary.mkdir()
    os.environ["TMPDIR"] = str(temporary)
    archive = subprocess.check_output(["git", "-C", str(repo), "archive", manifest["base"]])
    records = []

    def apply(root, layer):
        patch = BUNDLE / layer["patch"]
        prefix = str(root.relative_to(repo))
        for operation in ("check", "apply"):
            argv = ["git", "-C", str(repo), "apply", "--directory", prefix]
            if operation == "check":
                argv.append("--check")
            argv.append(str(patch))
            result = subprocess.run(argv, capture_output=True, text=True)
            records.append({"operation": operation, "target": root.name, "argv": argv, "exit_code": result.returncode, "output": result.stdout + result.stderr})
            assert result.returncode == 0, records[-1]
        for row in layer.get("untracked", []):
            destination = root / row["path"]
            payload = BUNDLE / row["payload"]
            assert digest(payload) == row["sha256"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                assert digest(destination) == row["sha256"]
            else:
                shutil.copyfile(payload, destination)
            destination.chmod(stat.S_IMODE(row["mode"]))

    for variant in ("baseline", "common", "850", "859", "793"):
        root = run_root / variant
        root.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(root, filter="data")
        base_expected = json.loads((BUNDLE / manifest["base_files"]).read_text())
        verify_tree(root, base_expected)
        if variant != "baseline":
            apply(root, manifest["common"])
            if variant != "common":
                apply(root, manifest["issues"][variant])
            expected_path = manifest["common"]["expected_files"] if variant == "common" else manifest["issues"][variant]["expected_files"]
            expected = json.loads((BUNDLE / expected_path).read_text())
            verify_tree(root, expected)
        else:
            expected = base_expected
        records.append({"operation": "verify-all-source-files", "variant": variant, "file_count": len(expected), "expected_manifest": manifest["base_files"] if variant == "baseline" else expected_path, "passed": True})
    # The baseline imports its own pinned source, never the candidate's package.
    benchmark = run_root / "baseline/scripts/benchmark_cache_release.py"
    assert not benchmark.exists(), "Do not overwrite a pinned baseline source file"
    shutil.copyfile(BUNDLE / "scripts/benchmark_cache_release.py", benchmark)
    benchmark.chmod(stat.S_IMODE(manifest["benchmark_mode"]))
    assert digest(benchmark) == manifest["benchmark_sha256"]
    assert benchmark.resolve().parents[1] == run_root / "baseline"
    assert digest(benchmark) == digest(run_root / "793/scripts/benchmark_cache_release.py")
    records.append({"operation": "baseline-harness-only", "path": str(benchmark), "sha256": digest(benchmark), "import_root": str(benchmark.resolve().parents[1]), "baseline_production_unchanged": True})
    proof = {"base": manifest["base"], "archive_sha256": hashlib.sha256(archive).hexdigest(), "bundle_manifest_sha256": digest(BUNDLE / "manifest.json"), "source_replay_only": True, "native_execution": "NOT RUN", "records": records}
    (run_root / "replay-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    proof = materialize(args.repo.resolve(), args.run_root.resolve())
    print(json.dumps({"proof": str(args.run_root.resolve() / "replay-proof.json"), "verified_variants": 5, "source_file_counts": {row["variant"]: row["file_count"] for row in proof["records"] if row["operation"] == "verify-all-source-files"}, "native_execution": "NOT RUN"}, indent=2))


if __name__ == "__main__":
    main()
