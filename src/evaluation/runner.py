import csv
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from src.classifiers.base import Classifier
from src.config.config import PROJECT_ROOT, DatasetConfig, EvalConfig, RunConfig
from src.evaluation.factory import USECASES, build_classifier
from src.evaluation.metrics import accuracy, macro_f1
from src.models.classification import TaskDefinition

MAX_CONSECUTIVE_ERRORS = 5  # stop a run early (e.g. wrong API key) instead of failing every sample

PREDICTION_COLUMNS = [
    "index", "text", "gold", "pred", "correct", "error",
    "input_tokens", "output_tokens", "reasoning_tokens", "latency_s",
]
SUMMARY_COLUMNS = [
    "usecase", "run", "provider", "model", "temperature", "reasoning_effort", "max_retries",
    "samples", "completed", "errors", "aborted", "accuracy", "macro_f1",
    "input_tokens", "output_tokens", "reasoning_tokens", "mean_latency_s",
]


@dataclass(frozen=True)
class Sample:
    text: str
    label: str


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_samples(dataset: DatasetConfig) -> List[Sample]:
    with open(resolve(dataset.path), encoding="utf-8", newline="") as f:
        rows = [row for row in csv.reader(f, delimiter=dataset.delimiter) if any(c.strip() for c in row)]
    header = rows[0]
    if len(header) == 1 and dataset.delimiter in header[0]:
        header = header[0].split(dataset.delimiter)  # header written as one quoted cell
    header = [c.strip() for c in header]
    for column in (dataset.text_column, dataset.label_column):
        if column not in header:
            raise ValueError(f"{dataset.path}: column {column!r} not found; columns are {header}")
    text_i, label_i = header.index(dataset.text_column), header.index(dataset.label_column)

    samples = []
    for line_no, row in enumerate(rows[1:], start=2):
        if len(row) <= max(text_i, label_i):
            raise ValueError(f"{dataset.path}: data row {line_no - 1} has only {len(row)} columns: {row}")
        samples.append(Sample(text=row[text_i], label=row[label_i].strip()))

    if dataset.max_samples is not None and dataset.max_samples < len(samples):
        samples = random.Random(dataset.seed).sample(samples, dataset.max_samples)
    return samples


def validate_labels(samples: List[Sample], task: TaskDefinition, dataset_path: Path) -> None:
    unknown = sorted({s.label for s in samples} - set(task.criteria))
    if unknown:
        raise ValueError(
            f"{dataset_path}: labels {unknown} are not classes of the {task.key!r} task {list(task.criteria)}"
        )


def _sum_or_none(values: list) -> Optional[int]:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def run_samples(usecase, samples: List[Sample], on_progress: Callable[[int, int], None] = lambda i, n: None):
    """Predict every sample; returns (records, aborted). A failed call is recorded, not raised."""
    records, consecutive_errors, aborted = [], 0, False
    for i, sample in enumerate(samples, start=1):
        record = {"index": i, "text": sample.text, "gold": sample.label, "pred": None, "error": None,
                  "input_tokens": None, "output_tokens": None, "reasoning_tokens": None}
        started = time.perf_counter()
        try:
            result = usecase.predict(sample.text)
            record.update(pred=result.label, input_tokens=result.usage.input_tokens,
                          output_tokens=result.usage.output_tokens,
                          reasoning_tokens=result.usage.reasoning_tokens)
            consecutive_errors = 0
        except Exception as e:  # keep evaluating the remaining samples
            record["error"] = f"{type(e).__name__}: {e}"
            consecutive_errors += 1
        record["latency_s"] = round(time.perf_counter() - started, 3)
        record["correct"] = record["pred"] == sample.label
        records.append(record)
        on_progress(i, len(samples))
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            aborted = True
            break
    return records, aborted


def summarize(usecase_name: str, run_index: int, run: RunConfig, total: int, records: list, aborted: bool) -> dict:
    gold = [r["gold"] for r in records]
    pred = [r["pred"] for r in records]
    return {
        "usecase": usecase_name, "run": run_index, "provider": run.provider, "model": run.model,
        "temperature": getattr(run, "temperature", None),
        "reasoning_effort": getattr(run, "reasoning_effort", None),
        "max_retries": run.max_retries,
        "samples": total, "completed": len(records),
        "errors": sum(r["error"] is not None for r in records), "aborted": aborted,
        "accuracy": round(accuracy(gold, pred), 4) if records else None,
        "macro_f1": round(macro_f1(gold, pred), 4) if records else None,
        "input_tokens": _sum_or_none([r["input_tokens"] for r in records]),
        "output_tokens": _sum_or_none([r["output_tokens"] for r in records]),
        "reasoning_tokens": _sum_or_none([r["reasoning_tokens"] for r in records]),
        "mean_latency_s": round(sum(r["latency_s"] for r in records) / len(records), 3) if records else None,
    }


def _write_csv(path: Path, columns: list, rows: list) -> None:
    # utf-8-sig so Excel shows Thai text correctly
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _file_part(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in text)


def run_evaluation(
    config: EvalConfig,
    *,
    dry_run: bool = False,
    classifier_factory: Callable[[RunConfig], Classifier] = build_classifier,
    log: Callable[[str], None] = print,
):
    """Evaluate every run of every usecase in `config`; returns (output_dir, summary_rows).

    Datasets and labels are validated for all usecases before the first API call.
    With dry_run=True nothing is called or written and (None, []) is returned.
    """
    plan = []
    for name, usecase_config in config.usecases.items():
        task = USECASES[name](classifier=None).task
        samples = load_samples(usecase_config.dataset)
        validate_labels(samples, task, usecase_config.dataset.path)
        plan.append((name, usecase_config, samples))

    total_calls = sum(len(samples) * len(uc.runs) for _, uc, samples in plan)
    log(f"Plan: {total_calls} API calls")
    for name, usecase_config, samples in plan:
        log(f"  {name}: {len(samples)} samples x {len(usecase_config.runs)} runs")
        for i, run in enumerate(usecase_config.runs, start=1):
            log(f"    #{i} {run.provider}:{run.model}")
    if dry_run:
        return None, []

    output_dir = resolve(config.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for name, usecase_config, samples in plan:
        for i, run in enumerate(usecase_config.runs, start=1):
            label = f"{name} #{i} {run.provider}:{run.model}"
            usecase = USECASES[name](classifier_factory(run))
            records, aborted = run_samples(
                usecase, samples, on_progress=lambda done, n: log(f"  [{label}] {done}/{n}")
            )
            _write_csv(output_dir / f"{name}__{i}_{run.provider}_{_file_part(run.model)}.csv",
                       PREDICTION_COLUMNS, records)
            summary_rows.append(summarize(name, i, run, len(samples), records, aborted))
            _write_csv(output_dir / "summary.csv", SUMMARY_COLUMNS, summary_rows)  # survives a later crash
            if aborted:
                log(f"  [{label}] aborted after {MAX_CONSECUTIVE_ERRORS} consecutive errors: {records[-1]['error']}")
    return output_dir, summary_rows
