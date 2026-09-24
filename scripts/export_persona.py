"""Export final persona-conditioned comparisons for the already approved public cohort."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from scripts.export_data import COUNTS, SOURCES, group_for_rank, public_output

PROFILE = "qwen-text-nonthinking"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DECODING = {
    "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 20,
    "presence_penalty": 2.0, "min_p": 0.0, "repetition_penalty": 1.0,
}
MODELS = [
    {"id": "base", "label": "Base - non-thinking", "shortLabel": "Base",
     "trainingDataset": None, "checkpoint": None, "epoch": 0, "conditioned": False, "mode": "nonthinking"},
    *[
        {"id": f"{dataset}-{kind}", "label": f"{dataset.title()} {kind}",
         "shortLabel": f"{dataset.title()} {kind}", "trainingDataset": dataset,
         "checkpoint": step, "epoch": 3, "conditioned": kind == "conditioned", "mode": "nonthinking"}
        for dataset, step in (("joint", 923), ("disjoint", 927))
        for kind in ("unconditioned", "conditioned")
    ],
]
IDENTITY_FIELDS = (
    "findings", "reference_impression", "reference_token_count", "persona_rank", "persona_partition",
    "persona_id", "persona_policy_sha256", "source_dataset", "source_case_id", "modality",
)


def context(model: dict, dataset: str, rank: int) -> dict:
    group_for_rank(rank)
    training = model["trainingDataset"]
    dataset_id = training == dataset
    persona_id = dataset_id and rank <= 40
    return {
        "conditioningToken": (
            f"<|persona_{rank:03d}|>" if persona_id else "<|persona_unknown|>"
        ) if model["conditioned"] else None,
        "datasetExposure": "not_applicable" if training is None else "in_distribution" if dataset_id else "ood",
        "personaExposure": "not_applicable" if training is None else "in_distribution" if persona_id else "ood",
    }


def final_cells(index: dict) -> dict:
    if index["status"] != "complete" or index["replicates_per_family"] != 3:
        raise ValueError("A complete three-judge evaluation index is required")
    cells = {}
    models = {model["id"]: model for model in MODELS}
    for entry in index["cells"].values():
        if entry["model"] != "base" and not (entry.get("is_final") and not entry.get("is_token_control")):
            continue
        key = (entry["model"], entry["test_dataset"])
        if key in cells or key[0] not in models or key[1] not in COUNTS:
            raise ValueError("Duplicate or unexpected final cell")
        model = models[key[0]]
        control = "headline" if model["conditioned"] else "unconditioned"
        if (entry["epoch"] != model["epoch"] or entry["training_dataset"] != model["trainingDataset"]
                or entry["control"] != control or entry.get("global_step") != model["checkpoint"]
                or entry["full_summary"]["all_four_families_complete"] is not True):
            raise ValueError("Unexpected checkpoint, control, or incomplete full-test evaluation")
        cells[key] = entry
    if set(cells) != {(m["id"], dataset) for m in MODELS for dataset in COUNTS}:
        raise ValueError("All ten final full-test cells are required")
    return cells


def validate_row(row: dict, model: dict, dataset: str) -> dict:
    expected = context(model, dataset, row["persona_rank"])
    partition = "in_distribution" if row["persona_rank"] <= 40 else "ood"
    if (row["experiment_dataset"] != dataset or row["split"] != "test"
            or row["persona_partition"] != partition or row["thinking_mode"] is not False
            or row["decoding_parameters"] != DECODING):
        raise ValueError("Unexpected split, persona partition, or decoding profile")
    if (row.get("conditioning_token") != expected["conditioningToken"]
            or row["persona_exposure"] != expected["personaExposure"]
            or row.get("dataset_exposure", "not_applicable") != expected["datasetExposure"]):
        raise ValueError("Incorrect conditioning token or model exposure")
    identity = row["model_checkpoint_id"]
    if model["trainingDataset"] is None:
        if (identity["revision"] != REVISION or identity["model_id"] != "Qwen/Qwen3.5-2B"
                or identity["decoding_profile"] != PROFILE):
            raise ValueError("Unexpected frozen base identity")
    elif (identity["epoch"] != 3 or identity["global_step"] != model["checkpoint"]
          or identity["training_dataset"] != model["trainingDataset"]
          or identity["settings"]["decoding_profile"] != PROFILE
          or identity["stage"] != ("inference" if model["conditioned"] else "unconditioned")
          or row["token_control"] != ("headline" if model["conditioned"] else "unconditioned")):
        raise ValueError("Only final epoch-3 headline checkpoints may be exported")
    return public_output(row) | expected


def build_dataset(cells: dict, dataset: str, approved: dict) -> dict:
    reports = {}
    for model in MODELS:
        entry = cells[model["id"], dataset]
        seen = set()
        with Path(entry["full_records"]).open() as stream:
            for line in stream:
                row = json.loads(line)
                key = row["example_id"]
                if key in seen:
                    raise ValueError("Duplicate test example")
                seen.add(key)
                identity = {field: row[field] for field in IDENTITY_FIELDS}
                if key not in reports:
                    if model != MODELS[0]:
                        raise ValueError("Model outputs have different example identities")
                    reports[key] = {"identity": identity, "outputs": {}}
                if reports[key]["identity"] != identity:
                    raise ValueError("Reference or semantic persona changed between models")
                reports[key]["outputs"][model["id"]] = validate_row(row, model, dataset)
        summary = entry["full_summary"]
        if set(reports) != seen or len(seen) != summary["number_eligible"] or not seen:
            raise ValueError("Incomplete model coverage")
        outputs = [reports[key]["outputs"][model["id"]] for key in sorted(seen)]
        measured = {
            "green": (sum(o["green"]["mean"] for o in outputs) / len(outputs), summary["green_mean"]),
            "style": (sum(o["style"]["mean"] for o in outputs) / len(outputs), summary["style_mean"]),
            "rouge": (sum(o["rougeL"] for o in outputs) / len(outputs), summary["rouge_l"]["f1"]),
            "checks": (sum(o["checksPassed"] for o in outputs) / len(outputs), summary["deterministic_pass_rate"]),
        }
        if any(not math.isclose(actual, recorded, abs_tol=1e-10, rel_tol=0) for actual, recorded in measured.values()):
            raise ValueError("Export scores differ from the completed experiment")
    if approved["dataset"] != dataset or len(approved["cases"]) != len(reports):
        raise ValueError("The already approved public cohort must match")
    cases = []
    for key, previous in zip(sorted(reports), approved["cases"]):
        row = reports[key]["identity"]
        shared = {
            "source": row["source_dataset"], "personaRank": row["persona_rank"],
            "personaGroup": group_for_rank(row["persona_rank"]), "findings": row["findings"],
            "reference": row["reference_impression"], "referenceTokens": row["reference_token_count"],
        }
        if row["source_dataset"] not in SOURCES or any(previous[field] != value for field, value in shared.items()):
            raise ValueError("Refusing to publish text outside the already approved public cohort")
        cases.append({"id": previous["id"], **shared, "outputs": reports[key]["outputs"]})
    counts = (len(cases), sum(c["personaGroup"] == "id" for c in cases), sum(c["personaGroup"] == "ood" for c in cases))
    if counts != COUNTS[dataset] or {c["personaRank"] for c in cases} != set(range(1, 51)):
        raise ValueError("Unexpected report or persona coverage")
    return {
        "schemaVersion": 1, "dataset": dataset, "label": dataset.title(), "decodingProfile": PROFILE,
        "personaGrouping": "Catalog ranks 1-40 = ID; 41-50 = OOD. Actual model exposure is reported separately.",
        "models": MODELS, "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--approved-data", type=Path, default=Path("docs/data"))
    parser.add_argument("--output", type=Path, default=Path("docs/data/persona"))
    parser.add_argument("--public-release-approved", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.approved_data.resolve():
        raise ValueError("The original data release must not be overwritten")
    index_bytes = (args.run / "artifacts/evaluation/index.json").read_bytes()
    summary_bytes = (args.run / "artifacts/results/results_summary.json").read_bytes()
    if json.loads(summary_bytes)["status"] != "complete":
        raise ValueError("The completed experiment summary is required")
    cells = final_cells(json.loads(index_bytes))
    datasets = [build_dataset(cells, dataset, json.loads((args.approved_data / f"{dataset}.json").read_text()))
                for dataset in COUNTS]
    catalog = {
        "schemaVersion": 1, "experiment": "persona-conditioned-20260922",
        "exportedAt": datetime.now(timezone.utc).isoformat(), "decodingProfile": PROFILE,
        "decodingParameters": DECODING, "modelId": "Qwen/Qwen3.5-2B", "baseRevision": REVISION,
        "models": MODELS, "totalReports": sum(len(d["cases"]) for d in datasets),
        "datasets": [{"id": d["dataset"], "label": d["label"], "reports": len(d["cases"]),
                      "idReports": sum(c["personaGroup"] == "id" for c in d["cases"]),
                      "oodReports": sum(c["personaGroup"] == "ood" for c in d["cases"])} for d in datasets],
        "sourceIndexSha256": hashlib.sha256(index_bytes).hexdigest(),
        "sourceSummarySha256": hashlib.sha256(summary_bytes).hexdigest(),
        "releaseApproval": {"date": "2026-09-24", "scope": "Owner requested conditioned generations on the public dashboard; reference cohort matches the September 21 approved release."},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [("catalog", catalog), *[(d["dataset"], d) for d in datasets]]:
        target = args.output / f"{name}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.replace(target)
    print(json.dumps({"reports": catalog["totalReports"], "models": len(MODELS),
                      "outputs": catalog["totalReports"] * len(MODELS)}))


if __name__ == "__main__":
    main()
