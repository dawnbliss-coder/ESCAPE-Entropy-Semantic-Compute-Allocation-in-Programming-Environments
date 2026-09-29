"""Run `python -m escape_scoring --help`."""
import argparse
import json
from pathlib import Path

from .calibration import SEED, calibrate, split_manifest
from .contract import Dataset, read_jsonl, validate_record, write_json, write_jsonl
from .engine import evaluate, write_csv
from .parallel import ordered_map


def _load(dataset, sample_id):
    dataset.load(sample_id)


def _region_rows(dataset, sample_id):
    from .o5 import region_table
    return region_table([dataset.load(sample_id)])


def main(argv=None):
    ap = argparse.ArgumentParser(description="ESCAPE Workstream C v1: byte-level scoring")
    commands = ap.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("synthetic", help="create labeled synthetic data, no model")
    demo.add_argument("--output", required=True)
    demo.add_argument("--random-targets", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--data", required=True)
    validate.add_argument("--workers", type=int, default=1)
    split = commands.add_parser("split")
    split.add_argument("--data", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", type=int, default=SEED)
    cal = commands.add_parser("calibrate")
    cal.add_argument("--data", required=True)
    cal.add_argument("--output", required=True)
    cal.add_argument("--seed", type=int, default=SEED)
    cal.add_argument("--permutations", type=int, default=10000)
    cal.add_argument("--workers", type=int, default=1, help="worker processes; results do not depend on it")
    score = commands.add_parser("score")
    score.add_argument("--data", required=True)
    score.add_argument("--calibration", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--permutations", type=int, default=10000)
    score.add_argument("--bootstrap", type=int, default=2000)
    score.add_argument("--workers", type=int, default=1, help="worker processes; results do not depend on it")
    base = commands.add_parser("baselines", help="newline/indent/word baselines under the frozen k")
    base.add_argument("--data", required=True)
    base.add_argument("--calibration", required=True)
    base.add_argument("--output", required=True)
    base.add_argument("--bootstrap", type=int, default=2000)
    base.add_argument("--workers", type=int, default=1)
    o5 = commands.add_parser("o5")
    o5.add_argument("--data", required=True)
    o5.add_argument("--output", required=True)
    o5.add_argument("--outcome", choices=["all_boundary_density", "inverse_bpp_compute_density"], default="all_boundary_density")
    o5.add_argument("--controls", nargs="*", default=["sample_id"])
    o5.add_argument("--numeric-controls", nargs="*", default=[])
    o5.add_argument("--workers", type=int, default=1)
    adapter = commands.add_parser("convert-legacy")
    adapter.add_argument("--root", default=".")
    adapter.add_argument("--output", required=True)
    adapter.add_argument("--run-id", required=True)
    adapter.add_argument("--source", choices=["corpus", "golden"], default="corpus")
    adapter.add_argument("--sample-ids", nargs="+")
    args = ap.parse_args(argv)
    if args.command == "synthetic":
        from .fixtures import create_fixture
        create_fixture(args.output, aligned=not args.random_targets)
        print(f"SYNTHETIC data written to {args.output}")
        return
    if args.command == "convert-legacy":
        from .adapter import convert
        print(f"Converted {convert(args.root, args.output, args.run_id, source=args.source, sample_ids=args.sample_ids)} files")
        return
    if args.command == "split":
        manifest = list(read_jsonl(Path(args.data) / "sample_manifest.jsonl"))
        for row in manifest:
            validate_record("sample_manifest", row)
        if Path(args.output).exists():
            raise FileExistsError("refusing to overwrite split manifest")
        write_jsonl(args.output, split_manifest(manifest, args.seed))
        return
    dataset = Dataset(args.data)
    if args.command == "validate":
        for _ in ordered_map(_load, [(row["sample_id"],) for row in dataset.manifest], dataset, args.workers):
            pass
        print(f"Validated {len(dataset.manifest)} samples")
    elif args.command == "calibrate":
        result = calibrate(dataset, args.output, seed=args.seed, permutations=args.permutations, workers=args.workers)
        print(json.dumps(result["selected_k"], sort_keys=True))
    elif args.command == "score":
        result = evaluate(dataset, json.loads(Path(args.calibration).read_text()), args.output,
                          permutations=args.permutations, bootstrap=args.bootstrap, workers=args.workers)
        print(json.dumps(next(row for row in result["scores"] if row["axis"] == "global" and row["kind"] == "start")))
    elif args.command == "baselines":
        from .baselines import score_baselines
        result = score_baselines(dataset, json.loads(Path(args.calibration).read_text()), args.output,
                                 bootstrap=args.bootstrap, workers=args.workers)
        print(f"{len(result['rows'])} baseline rows written to {args.output}")
    elif args.command == "o5":
        from .o5 import region_table, regress
        jobs = [(m["sample_id"],) for m in dataset.manifest]
        rows = [row for part in ordered_map(_region_rows, jobs, dataset, args.workers) for row in part]
        result, design = regress(rows, outcome=args.outcome, controls=args.controls, numeric_controls=args.numeric_controls)
        result["synthetic"] = all(m["synthetic"] for m in dataset.manifest)
        out = Path(args.output)
        write_json(out / "o5.json", result, exclusive=True)
        write_csv(out / "o5_regions.csv", rows)
        write_csv(out / "o5_design.csv", design)
        print(result["inference"])


if __name__ == "__main__":
    main()
