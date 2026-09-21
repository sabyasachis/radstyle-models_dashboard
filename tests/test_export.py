import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("export_data", Path(__file__).parents[1] / "scripts/export_data.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class ExportTests(unittest.TestCase):
    def row(self):
        aggregate = {"all_three_valid": True, "valid_count": 3, "mean": 0.8, "std": 0.1}
        return {
            "parser_status": "valid", "eligible_for_scoring": True,
            "generated_impression": "Example output", "impression_token_count": 4,
            "rouge_l": {"f1": 0.5}, "deterministic_metrics": {"passed": False},
            "impression_truncated": False, "scratchpad_fallback": False,
            "green_aggregate": dict(aggregate), "style_aggregate": dict(aggregate),
            "generated_reasoning": "MUST NOT EXPORT", "source_path": "PRIVATE",
            "patient_study_ids": {"private": "PRIVATE"}, "green_judgments": ["PRIVATE"],
        }

    def test_allowlist_excludes_private_fields(self):
        result = export.public_output(self.row())
        self.assertEqual(set(result), {"text", "tokens", "rougeL", "checksPassed", "truncated", "thinkingFallback", "green", "style"})
        self.assertNotIn("PRIVATE", str(result))
        self.assertNotIn("MUST NOT EXPORT", str(result))

    def test_incomplete_judging_is_rejected(self):
        row = self.row()
        row["green_aggregate"]["valid_count"] = 2
        with self.assertRaises(ValueError):
            export.public_output(row)

    def test_nonfinite_scores_are_rejected(self):
        for value in [float("nan"), float("inf"), True, -0.1, 1.1, "0.8"]:
            with self.assertRaises(ValueError):
                export.score(value)

    def test_partition_is_rank_only(self):
        self.assertEqual(export.group_for_rank(40), "id")
        self.assertEqual(export.group_for_rank(41), "ood")
        for value in [0, 51, True, "40"]:
            with self.assertRaises(ValueError):
                export.group_for_rank(value)


if __name__ == "__main__":
    unittest.main()
