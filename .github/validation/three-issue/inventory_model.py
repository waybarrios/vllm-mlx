#!/usr/bin/env python3
"""Hash an already-present local model without loading it or accessing a network."""

import argparse
import json
from pathlib import Path

from replay_sources import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo, model, output = args.repo.resolve(), args.model.expanduser().resolve(), args.output.resolve()
    assert output.is_relative_to(repo) and not output.exists()
    assert (model / "config.json").is_file()
    files = {str(path.relative_to(model)): digest(path) for path in sorted(model.rglob("*")) if path.is_file()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"model_path": str(model), "files": files}, indent=2) + "\n")
    print(json.dumps({"manifest": str(output), "sha256": digest(output), "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
