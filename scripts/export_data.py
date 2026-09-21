"""Export only release-approved fields from the original experiment snapshot."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

MODELS = [
    {"id": "base_nonthinking", "label": "Base · non-thinking", "shortLabel": "Base non-thinking", "trainingDataset": None, "checkpoint": None, "mode": "nonthinking"},
    {"id": "base_thinking", "label": "Base · thinking", "shortLabel": "Base thinking", "trainingDataset": None, "checkpoint": None, "mode": "thinking"},
    {"id": "joint_nonthinking", "label": "Joined SFT · non-thinking", "shortLabel": "Joined SFT", "trainingDataset": "joint", "checkpoint": 300, "mode": "nonthinking"},
    {"id": "disjoint_nonthinking", "label": "Disjoint SFT · non-thinking", "shortLabel": "Disjoint SFT", "trainingDataset": "disjoint", "checkpoint": 600, "mode": "nonthinking"},
]
COUNTS = {"joint": (858, 683, 175), "disjoint": (829, 675, 154)}
SOURCES = {"ct_rate", "mr_rate", "rexgradient"}


def group_for_rank(rank: int) -> str:
    if type(rank) is not int or not 1 <= rank <= 50:
        raise ValueError("Invalid persona rank")
    return "id" if rank <= 40 else "ood"


def text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing report text")
    return value


def score(value: object) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Invalid numeric evaluation score")
    return float(value)


def token_count(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Invalid token count")
    return value


def boolean(value: object) -> bool:
    if type(value) is not bool:
        raise ValueError("Invalid boolean diagnostic")
    return value


def public_output(row: dict) -> dict:
    if row["parser_status"] != "valid" or not row["eligible_for_scoring"]:
        raise ValueError("Only complete valid candidates may be exported")
    result = {
        "text": text(row["generated_impression"]),
        "tokens": token_count(row["impression_token_count"]),
        "rougeL": score(row["rouge_l"]["f1"]),
        "checksPassed": boolean(row["deterministic_metrics"]["passed"]),
        "truncated": boolean(row["impression_truncated"]),
        "thinkingFallback": boolean(row["scratchpad_fallback"]),
    }
    for family in ("green", "style"):
        aggregate = row[f"{family}_aggregate"]
        if aggregate["all_three_valid"] is not True or aggregate["valid_count"] != 3:
            raise ValueError("All three judge replicates are required")
        result[family] = {"mean": score(aggregate["mean"]), "sd": score(aggregate["std"])}
    return result


def build_dataset(artifacts: Path, dataset: str, snapshot: dict) -> dict:
    reports = {}
    shared = ("findings", "reference_impression", "persona_rank", "source_dataset", "reference_token_count")
    expected_decoding = {
        "nonthinking": {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
        "thinking": {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
    }
    for model in MODELS:
        cell = f"{model['id']}__{dataset}"
        summary = snapshot["matrix"][cell]
        if not summary["all_four_families_complete"]:
            raise ValueError(f"Incomplete snapshot cell: {cell}")
        seen = set()
        path = artifacts / "evaluation" / cell / "evaluation_records.jsonl"
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                key = row["example_id"]
                if key in seen:
                    raise ValueError(f"Duplicate example in {cell}")
                seen.add(key)
                if row["experiment_dataset"] != dataset or row["split"] != "test":
                    raise ValueError("Unexpected dataset or split")
                if row["decoding_parameters"] != expected_decoding[model["mode"]]:
                    raise ValueError("Refusing to mix decoding profiles")
                if key not in reports:
                    if model != MODELS[0]:
                        raise ValueError("Model outputs have different example identities")
                    reports[key] = {field: row[field] for field in shared} | {"outputs": {}}
                elif any(reports[key][field] != row[field] for field in shared):
                    raise ValueError("Reference text or persona changed between models")
                reports[key]["outputs"][model["id"]] = public_output(row)
        if set(reports) != seen or len(seen) != summary["number_eligible"]:
            raise ValueError(f"Incomplete model coverage: {cell}")
        for family in ("green", "style"):
            mean = sum(reports[key]["outputs"][model["id"]][family]["mean"] for key in seen) / len(seen)
            if not math.isclose(mean, summary[f"{family}_mean"], abs_tol=1e-10):
                raise ValueError("Per-report scores do not match frozen snapshot")
    cases = []
    for index, key in enumerate(sorted(reports), 1):
        row = reports[key]
        if row["source_dataset"] not in SOURCES:
            raise ValueError("Unknown dataset source")
        cases.append({
            "id": f"{'joined' if dataset == 'joint' else 'disjoint'}-{index:04d}",
            "source": row["source_dataset"],
            "personaRank": row["persona_rank"],
            "personaGroup": group_for_rank(row["persona_rank"]),
            "findings": text(row["findings"]),
            "reference": text(row["reference_impression"]),
            "referenceTokens": token_count(row["reference_token_count"]),
            "outputs": row["outputs"],
        })
    counts = (len(cases), sum(c["personaGroup"] == "id" for c in cases), sum(c["personaGroup"] == "ood" for c in cases))
    if counts != COUNTS[dataset] or {c["personaRank"] for c in cases} != set(range(1, 51)):
        raise ValueError("Unexpected report or persona coverage")
    return {
        "schemaVersion": 1, "dataset": dataset,
        "label": "Joined" if dataset == "joint" else "Disjoint",
        "snapshotCompletedAt": snapshot["completed_at_utc"], "decodingProfile": "original-experiment",
        "personaGrouping": "Test catalog ranks 1-40 = ID; 41-50 = OOD. Cross-dataset evaluation does not change this dashboard label.",
        "models": MODELS, "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/data"))
    parser.add_argument("--public-release-approved", action="store_true", required=True,
                        help="Required acknowledgment that the clinical text is approved for public redistribution")
    args = parser.parse_args()
    snapshot = json.loads((args.artifacts / "deadline_20260921_0600/results_summary.json").read_text())
    if snapshot["status"] != "complete_snapshot" or snapshot["complete_matrix_cells"] != 12:
        raise ValueError("A completed frozen snapshot is required")
    datasets = [build_dataset(args.artifacts, name, snapshot) for name in COUNTS]
    catalog = {
        "schemaVersion": 1, "snapshotCompletedAt": snapshot["completed_at_utc"],
        "decodingProfile": "original-experiment", "modelId": "Qwen/Qwen3.5-2B",
        "models": MODELS, "totalReports": sum(len(d["cases"]) for d in datasets),
        "datasets": [{"id": d["dataset"], "label": d["label"], "reports": len(d["cases"]),
                      "idReports": sum(c["personaGroup"] == "id" for c in d["cases"]),
                      "oodReports": sum(c["personaGroup"] == "ood" for c in d["cases"])} for d in datasets],
        "releaseApproval": {"date": "2026-09-21", "scope": "Owner confirmed non-identifying records and approved public redistribution."},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [("catalog", catalog)] + [(d["dataset"], d) for d in datasets]:
        target = args.output / f"{name}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(target)
    print(json.dumps({"reports": catalog["totalReports"], "models": len(MODELS),
                      "outputs": catalog["totalReports"] * len(MODELS), "directory": str(args.output)}))


if __name__ == "__main__":
    main()
