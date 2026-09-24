import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import export_persona as export


class PersonaExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = {"status": "complete", "replicates_per_family": 3, "cells": {}}
        self.approved = {}
        counts = patch.dict(export.COUNTS, {"joint": (50, 40, 10), "disjoint": (50, 40, 10)})
        counts.start()
        self.addCleanup(counts.stop)
        for dataset in export.COUNTS:
            cases = [{
                "id": f"public-{rank:04d}", "source": "ct_rate", "personaRank": rank,
                "personaGroup": "id" if rank <= 40 else "ood", "findings": "Synthetic findings.",
                "reference": "Synthetic reference.", "referenceTokens": 4,
            } for rank in range(1, 51)]
            self.approved[dataset] = {"dataset": dataset, "cases": cases}
            for model in export.MODELS:
                name = f"{model['id']}__{dataset}"
                records = self.root / f"{name}.jsonl"
                records.write_text("".join(json.dumps(self.row(model, dataset, rank)) + "\n" for rank in range(1, 51)))
                self.index["cells"][name] = {
                    "model": model["id"], "test_dataset": dataset, "training_dataset": model["trainingDataset"],
                    "epoch": model["epoch"], "global_step": model["checkpoint"], "is_final": True,
                    "is_token_control": False, "control": "headline" if model["conditioned"] else "unconditioned",
                    "full_records": str(records),
                    "full_summary": {"all_four_families_complete": True, "number_eligible": 50,
                                     "green_mean": 0.8, "style_mean": 0.8,
                                     "rouge_l": {"f1": 0.5}, "deterministic_pass_rate": 1.0},
                }

    def row(self, model, dataset, rank):
        context = export.context(model, dataset, rank)
        return {
            "example_id": f"private-example-{rank:02d}", "source_case_id": f"PRIVATE-source-{rank}",
            "persona_id": f"PRIVATE-persona-{rank}", "persona_policy_sha256": "PRIVATE-policy",
            "modality": "CT", "experiment_dataset": dataset, "split": "test", "persona_rank": rank,
            "persona_partition": "in_distribution" if rank <= 40 else "ood", "source_dataset": "ct_rate",
            "findings": "Synthetic findings.", "reference_impression": "Synthetic reference.",
            "reference_token_count": 4, "generated_impression": "Synthetic generation.",
            "impression_token_count": 4, "impression_truncated": False, "scratchpad_fallback": False,
            "rouge_l": {"f1": 0.5}, "deterministic_metrics": {"passed": True},
            "parser_status": "valid", "eligible_for_scoring": True,
            "green_aggregate": {"all_three_valid": True, "valid_count": 3, "mean": 0.8, "std": 0.1},
            "style_aggregate": {"all_three_valid": True, "valid_count": 3, "mean": 0.8, "std": 0.1},
            "thinking_mode": False, "decoding_parameters": dict(export.DECODING),
            "conditioning_token": context["conditioningToken"], "dataset_exposure": context["datasetExposure"],
            "persona_exposure": context["personaExposure"],
            "token_control": "headline" if model["conditioned"] else "unconditioned",
            "model_checkpoint_id": {
                "revision": export.REVISION, "model_id": "Qwen/Qwen3.5-2B", "decoding_profile": export.PROFILE,
                "epoch": model["epoch"], "global_step": model["checkpoint"], "training_dataset": model["trainingDataset"],
                "settings": {"decoding_profile": export.PROFILE},
                "stage": "inference" if model["conditioned"] else "unconditioned",
                "checkpoint": "/PRIVATE/weights",
            },
            "generated_reasoning": "PRIVATE reasoning", "green_judgments": ["PRIVATE judge"],
            "patient_study_ids": {"patient": "PRIVATE"}, "source_path": "/PRIVATE/source",
        }

    def build(self, dataset="joint"):
        return export.build_dataset(export.final_cells(self.index), dataset, self.approved[dataset])

    def test_complete_release_preserves_cohort_and_only_exports_allowed_fields(self):
        for dataset in export.COUNTS:
            result = self.build(dataset)
            self.assertEqual(len(result["cases"]), 50)
            self.assertEqual(result["decodingProfile"], export.PROFILE)
            self.assertNotIn("PRIVATE", json.dumps(result))
            for case, approved in zip(result["cases"], self.approved[dataset]["cases"]):
                self.assertEqual({k: v for k, v in case.items() if k != "outputs"}, approved)
                self.assertEqual(set(case["outputs"]), {m["id"] for m in export.MODELS})
                for output in case["outputs"].values():
                    self.assertEqual(set(output), {
                        "text", "tokens", "rougeL", "checksPassed", "truncated", "thinkingFallback",
                        "green", "style", "conditioningToken", "datasetExposure", "personaExposure",
                    })
            cross = result["cases"][0]["outputs"][f"{'disjoint' if dataset == 'joint' else 'joint'}-conditioned"]
            self.assertEqual(cross["conditioningToken"], "<|persona_unknown|>")
            self.assertEqual(cross["datasetExposure"], "ood")
            self.assertEqual(cross["personaExposure"], "ood")
            own = result["cases"][0]["outputs"][f"{dataset}-conditioned"]
            self.assertEqual(own["conditioningToken"], "<|persona_001|>")
            self.assertEqual(result["cases"][40]["outputs"][f"{dataset}-conditioned"]["conditioningToken"], "<|persona_unknown|>")

    def test_wrong_token_or_exposure_rejected(self):
        model = next(m for m in export.MODELS if m["id"] == "joint-conditioned")
        for field, value in [("conditioning_token", "<|persona_001|>"), ("persona_exposure", "in_distribution"), ("dataset_exposure", "in_distribution")]:
            row = self.row(model, "disjoint", 1)
            row[field] = value
            with self.assertRaisesRegex(ValueError, "token or model exposure"):
                export.validate_row(row, model, "disjoint")

    def test_old_decoding_or_wrong_checkpoint_rejected(self):
        model = next(m for m in export.MODELS if m["id"] == "joint-conditioned")
        row = self.row(model, "joint", 1)
        row["decoding_parameters"]["temperature"] = 0.7
        with self.assertRaisesRegex(ValueError, "decoding profile"):
            export.validate_row(row, model, "joint")
        row = self.row(model, "joint", 1)
        row["model_checkpoint_id"]["global_step"] = 300
        with self.assertRaisesRegex(ValueError, "epoch-3"):
            export.validate_row(row, model, "joint")

    def test_incomplete_duplicate_or_control_cells_rejected(self):
        for mutate in [
            lambda index: index["cells"].pop("base__joint"),
            lambda index: index["cells"].update(duplicate=copy.deepcopy(index["cells"]["base__joint"])),
            lambda index: index["cells"]["joint-conditioned__joint"].update(control="wrong"),
            lambda index: index["cells"]["joint-conditioned__joint"].update(epoch=2.5),
        ]:
            index = copy.deepcopy(self.index)
            mutate(index)
            with self.assertRaises(ValueError):
                export.final_cells(index)

    def test_all_four_aggregate_metrics_are_checked(self):
        summary = self.index["cells"]["joint-conditioned__joint"]["full_summary"]
        for field in ["green_mean", "style_mean", "deterministic_pass_rate", "rouge_l"]:
            original = summary[field]
            summary[field] = {"f1": 0.2} if field == "rouge_l" else 0.2
            with self.assertRaisesRegex(ValueError, "scores differ"):
                self.build()
            summary[field] = original

    def test_unapproved_reference_or_changed_semantic_identity_rejected(self):
        self.approved["joint"]["cases"][0]["findings"] = "Another report."
        with self.assertRaisesRegex(ValueError, "approved public cohort"):
            self.build()
        self.approved["joint"]["cases"][0]["findings"] = "Synthetic findings."
        path = Path(self.index["cells"]["joint-conditioned__joint"]["full_records"])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["persona_policy_sha256"] = "changed"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "semantic persona"):
            self.build()


if __name__ == "__main__":
    unittest.main()
