#!/usr/bin/env python3

"""Write averaged benchmark results for each numbered results run."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METRIC_COLUMNS = (
	"samples",
	"total_ms",
	"average_ms",
	"min_ms",
	"max_ms",
	"stddev_ms",
	"p50_ms",
	"p90_ms",
	"p95_ms",
	"p99_ms",
)


def read_results(run_directory: Path) -> tuple[dict[str, list[dict[str, float]]], list[Path]]:
	rows_by_stage: dict[str, list[dict[str, float]]] = defaultdict(list)
	csv_files = sorted(run_directory.rglob("*.csv"))
	if not csv_files:
		raise ValueError(f"No CSV files found in {run_directory}")

	for csv_file in csv_files:
		with csv_file.open(newline="") as input_file:
			reader = csv.DictReader(input_file)
			if reader.fieldnames is None or reader.fieldnames != ["stage", *METRIC_COLUMNS]:
				raise ValueError(f"Unexpected columns in {csv_file}")
			for row in reader:
				if not row.get("stage"):
					raise ValueError(f"Missing stage name in {csv_file}")
				rows_by_stage[row["stage"]].append(
					{column: float(row[column]) for column in METRIC_COLUMNS}
				)

	return rows_by_stage, csv_files


def write_report(run_directory: Path, rows_by_stage: dict[str, list[dict[str, float]]], csv_files: list[Path]) -> Path:
	report_path = run_directory / "average.md"
	mode = {
		"1": "File input (without camera)",
		"2": "Camera input",
	}.get(run_directory.name, f"Run {run_directory.name}")

	lines = [
		f"# Average Benchmark Results: {mode}",
		"",
		f"Averages across {len(csv_files)} CSV runs: "
		+ ", ".join(csv_file.stem for csv_file in csv_files)
		+ ".",
		"",
		"All latency values are in milliseconds. Stages are averaged only across runs that contain the stage.",
		"",
		"| Stage | Runs | Samples | Total (ms) | Average (ms) | Min (ms) | Max (ms) | Stddev (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) |",
		"| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
	]

	for stage, rows in rows_by_stage.items():
		averages = {
			column: sum(row[column] for row in rows) / len(rows)
			for column in METRIC_COLUMNS
		}
		lines.append(
			"| {stage} | {runs} | {samples:.0f} | {total_ms:.3f} | {average_ms:.3f} | "
			"{min_ms:.3f} | {max_ms:.3f} | {stddev_ms:.3f} | {p50_ms:.3f} | "
			"{p90_ms:.3f} | {p95_ms:.3f} | {p99_ms:.3f} |".format(
				stage=stage,
				runs=len(rows),
				**averages,
			)
		)

	report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	return report_path


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"results_directory",
		nargs="?",
		type=Path,
		default=Path(__file__).resolve().parents[1] / "results" / "runs",
		help="Directory containing numbered run folders (default: results/runs)",
	)
	args = parser.parse_args()

	run_directories = sorted(
		directory
		for directory in args.results_directory.iterdir()
		if directory.is_dir() and directory.name.isdigit()
	)
	if not run_directories:
		raise ValueError(f"No numbered run folders found in {args.results_directory}")

	for run_directory in run_directories:
		rows_by_stage, csv_files = read_results(run_directory)
		report_path = write_report(run_directory, rows_by_stage, csv_files)
		print(f"Wrote {report_path}")


if __name__ == "__main__":
	main()
import argparse