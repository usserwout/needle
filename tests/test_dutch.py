import json
import tempfile
import unittest
from pathlib import Path


TOOLS = [
    {
        "name": "set_alarm",
        "description": "Set an alarm.",
        "parameters": {
            "type": "object",
            "properties": {"time": {"type": "string"}},
            "required": ["time"],
        },
    }
]


class DutchTrainingTests(unittest.TestCase):
    def test_validation_accepts_grounded_example_and_rejects_unknown_argument(self):
        from needle.dutch import validate_example

        valid = {
            "query": "Zet een alarm voor 07:30.",
            "tools": TOOLS,
            "answers": [{"name": "set_alarm", "arguments": {"time": "07:30"}}],
            "reasoning": "time='07:30' uit de vraag",
        }
        self.assertEqual(validate_example(valid), [])

        invalid = {
            **valid,
            "answers": [{"name": "set_alarm", "arguments": {"time": "08:00"}}],
        }
        errors = validate_example(invalid, require_grounding=True)
        self.assertTrue(any("not grounded" in error for error in errors))

    def test_semantic_metrics_ignore_json_key_order_and_report_no_call(self):
        from needle.dutch import score_predictions

        rows = [
            {"answers": [{"name": "set_alarm", "arguments": {"time": "07:30", "label": "werk"}}],
             "task_family": "action"},
            {"answers": [], "task_family": "negative"},
        ]
        predictions = [
            [{"name": "set_alarm", "arguments": {"label": "werk", "time": "07:30"}}],
            [],
        ]
        report = score_predictions(rows, predictions, bootstrap_samples=20, seed=4)
        self.assertEqual(report["exact_call_accuracy"], 1.0)
        self.assertEqual(report["no_call"]["f1"], 1.0)
        self.assertIn("action", report["slices"]["task_family"])

    def test_massive_builder_keeps_partitions_and_uses_english_schemas(self):
        from needle.dutch import build_massive_examples

        rows = [
            {"id": "a", "partition": "train", "locale": "nl-NL", "scenario": "alarm",
             "intent": "alarm_set", "utt": "Zet een alarm om zeven uur",
             "annot_utt": "Zet een alarm om [time : zeven uur]"},
            {"id": "b", "partition": "train", "locale": "nl-NL", "scenario": "alarm",
             "intent": "alarm_cancel", "utt": "Annuleer mijn alarm",
             "annot_utt": "Annuleer mijn alarm"},
            {"id": "c", "partition": "test", "locale": "nl-NL", "scenario": "alarm",
             "intent": "alarm_set", "utt": "Alarm om acht uur",
             "annot_utt": "Alarm om [time : acht uur]"},
        ]
        examples = build_massive_examples(rows, partition="train", limit=10, seed=2)
        self.assertEqual(len(examples), 2)
        self.assertTrue(all(row["split"] == "train" for row in examples))
        alarm = next(row for row in examples if row["id"] == "massive:a")
        self.assertEqual(alarm["answers"][0]["name"], "set_alarm")
        self.assertEqual(alarm["answers"][0]["arguments"], {"time": "zeven uur"})
        self.assertTrue(all(tool["name"].isascii() for tool in alarm["tools"]))

    def test_jsonl_validation_finds_cross_split_duplicates(self):
        from needle.dutch import validate_jsonl

        row = {
            "query": "Zet een alarm om 07:30",
            "tools": TOOLS,
            "answers": [{"name": "set_alarm", "arguments": {"time": "07:30"}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(
                json.dumps({**row, "split": "train"}) + "\n" +
                json.dumps({**row, "split": "test"}) + "\n",
                encoding="utf-8",
            )
            report = validate_jsonl(path, require_grounding=True)
        self.assertEqual(report["invalid_examples"], 0)
        self.assertEqual(len(report["cross_split_duplicates"]), 1)

    def test_multi_file_validation_fails_group_and_near_duplicate_leaks(self):
        from needle.dutch import validate_many_jsonl

        row = {
            "query": "Zet een alarm om 07:30", "tools": TOOLS,
            "answers": [{"name": "set_alarm", "arguments": {"time": "07:30"}}],
            "group_id": "same-request",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            test = root / "test.jsonl"
            train.write_text(json.dumps({**row, "split": "train"}) + "\n", encoding="utf-8")
            test.write_text(json.dumps({**row, "query": "Alarm om 07:30 zet", "split": "test"}) + "\n",
                            encoding="utf-8")
            report = validate_many_jsonl([train, test], require_grounding=True)
        self.assertGreater(report["leakage_examples"], 0)
        self.assertEqual(len(report["cross_split_group_leaks"]), 1)

    def test_validation_reports_malformed_schema_without_crashing(self):
        from needle.dutch import validate_example

        errors = validate_example({
            "query": "Zet een alarm om 07:30", "tools": [{"name": "set_alarm", "parameters": []}],
            "answers": [{"name": "set_alarm", "arguments": {}}],
        })
        self.assertTrue(any("parameters must be an object" in error for error in errors))

    def test_release_gates_enforce_quality_and_quantization_thresholds(self):
        from needle.dutch import release_gates

        report = {
            "exact_call_accuracy": 0.86, "exact_call_ci95": [0.83, 0.89],
            "tool_selection_accuracy": 0.92, "hallucinated_argument_rate": 0.01,
            "no_call": {"precision": 0.91, "recall": 0.92},
            "all_generated_calls_schema_valid": True,
            "slices": {"task_family": {"action": {"examples": 10, "exact_call_accuracy": 0.80}}},
        }
        self.assertTrue(release_gates(report, compact_report={"exact_call_accuracy": 0.85})["passed"])
        self.assertFalse(release_gates(report, compact_report={"exact_call_accuracy": 0.84})["passed"])

    def test_dataset_builder_writes_disjoint_training_artifacts(self):
        from needle.dutch import create_dutch_dataset, validate_jsonl

        rows = [
            {"id": str(index), "partition": partition, "locale": locale,
             "scenario": "alarm", "intent": intent, "utt": utterance,
             "annot_utt": annotation}
            for index, (partition, locale, intent, utterance, annotation) in enumerate([
                ("train", "nl-NL", "alarm_set", "Zet alarm om zeven uur", "Zet alarm om [time : zeven uur]"),
                ("train", "nl-NL", "alarm_cancel", "Annuleer alarm", "Annuleer alarm"),
                ("dev", "nl-NL", "alarm_set", "Alarm om acht uur", "Alarm om [time : acht uur]"),
                ("test", "nl-NL", "alarm_set", "Alarm om negen uur", "Alarm om [time : negen uur]"),
                ("train", "en-US", "alarm_set", "Set alarm at seven", "Set alarm at [time : seven]"),
                ("dev", "en-US", "alarm_set", "Alarm at eight", "Alarm at [time : eight]"),
            ])
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nl = root / "nl-NL.jsonl"
            en = root / "en-US.jsonl"
            nl.write_text("\n".join(json.dumps(row) for row in rows if row["locale"] == "nl-NL") + "\n")
            en.write_text("\n".join(json.dumps(row) for row in rows if row["locale"] == "en-US") + "\n")
            outputs = create_dutch_dataset(
                nl, root / "out", massive_en_path=en, seed=3,
                action_count=2, extraction_count=4, negative_count=2,
                multi_count=2, english_count=2,
            )
            train = [json.loads(line) for line in Path(outputs["train"]).read_text().splitlines()]
            test = [json.loads(line) for line in Path(outputs["test"]).read_text().splitlines()]
            validation = validate_jsonl(outputs["train"], require_grounding=True)
        self.assertEqual(len(train), 11)
        self.assertEqual(validation["invalid_examples"], 0)
        self.assertTrue(all(row["split"] == "train" for row in train))
        self.assertTrue(any(row["task_family"] == "extraction" for row in train))
        self.assertTrue(any(row["locale"] == "en-US" for row in train))
        self.assertTrue(all(row["split"] == "test" for row in test))

    def test_augmented_examples_are_varied_grounded_and_schema_valid(self):
        from needle.dutch import make_augmented_examples, validate_example

        rows = make_augmented_examples(500, seed=7)
        self.assertEqual(len(rows), 500)
        self.assertGreaterEqual(len({row["query"] for row in rows}), 450)
        self.assertEqual({"nl-NL", "nl-BE", "nl-noisy"}, {row["locale"] for row in rows})
        self.assertTrue({"action", "extraction", "negative", "multi_call"}.issubset(
            {row["task_family"] for row in rows}))
        targets = {row["answers"][0]["name"] for row in rows if row["answers"]}
        self.assertTrue({"move_robot", "analyze_sentiment", "launch_game",
                         "extract_receipt", "book_ticket", "transfer_money",
                         "send_email", "schedule_doctor_visit", "control_lights"}.issubset(targets))
        self.assertTrue(all(not validate_example(row, require_grounding=True) for row in rows))

    def test_deterministic_candidates_do_not_leak_target_position(self):
        from needle.dutch import make_extraction_examples, make_multi_call_examples

        extraction = make_extraction_examples(60, split="train")
        extraction_positions = {
            next(i for i, tool in enumerate(row["tools"])
                 if tool["name"] == row["answers"][0]["name"])
            for row in extraction
        }
        self.assertGreater(len(extraction_positions), 1)
        self.assertNotEqual(extraction_positions, {0})

        multi = make_multi_call_examples(60, split="train")
        multi_positions = {
            next(i for i, tool in enumerate(row["tools"])
                 if tool["name"] == row["answers"][0]["name"])
            for row in multi
        }
        self.assertGreater(len(multi_positions), 1)
        self.assertNotEqual(multi_positions, {0})

    def test_augmentation_dialects_are_not_tied_to_task_family(self):
        from collections import defaultdict
        from needle.dutch import make_augmented_examples

        rows = make_augmented_examples(1_000, seed=11)
        locales_by_family = defaultdict(set)
        for row in rows:
            locales_by_family[row["task_family"]].add(row["locale"])
        expected = {"nl-NL", "nl-BE", "nl-noisy"}
        for family in ("action", "extraction", "negative", "multi_call"):
            self.assertEqual(locales_by_family[family], expected)

    def test_multi_call_answer_order_follows_timer_first_request(self):
        from needle.dutch import make_multi_call_examples

        row = next(row for row in make_multi_call_examples(8, split="train")
                   if row["id"].endswith(":4"))
        self.assertEqual([answer["name"] for answer in row["answers"]],
                         ["set_timer", "set_alarm"])

    def test_augmentation_merges_training_file_and_writes_manifest(self):
        from needle.dutch import augment_training_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train.jsonl"
            output = root / "train-augmented.jsonl"
            source.write_text(json.dumps({
                "query": "Zet een alarm om 07:30", "tools": TOOLS,
                "answers": [{"name": "set_alarm", "arguments": {"time": "07:30"}}],
            }) + "\n", encoding="utf-8")
            manifest = augment_training_file(source, output, count=25, seed=2)
            rows = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 26)
            self.assertEqual(manifest["added_examples"], 25)
            self.assertTrue(Path(manifest["manifest"]).exists())

    def test_dataset_builder_can_include_full_augmentation_in_one_output(self):
        from needle.dutch import create_dutch_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "nl-NL.jsonl"
            english = root / "en-US.jsonl"
            rows = [
                {"id": "1", "partition": "train", "locale": "nl-NL",
                 "scenario": "alarm", "intent": "alarm_set",
                 "utt": "Zet een alarm om zeven uur",
                 "annot_utt": "Zet een alarm om [time : zeven uur]"},
                {"id": "2", "partition": "dev", "locale": "nl-NL",
                 "scenario": "alarm", "intent": "alarm_set",
                 "utt": "Zet een alarm om acht uur",
                 "annot_utt": "Zet een alarm om [time : acht uur]"},
                {"id": "3", "partition": "test", "locale": "nl-NL",
                 "scenario": "alarm", "intent": "alarm_set",
                 "utt": "Zet een alarm om negen uur",
                 "annot_utt": "Zet een alarm om [time : negen uur]"},
            ]
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            english.write_text(json.dumps({**rows[0], "locale": "en-US",
                                           "utt": "Set an alarm at seven",
                                           "annot_utt": "Set an alarm at [time : seven]"}) + "\n",
                               encoding="utf-8")
            outputs = create_dutch_dataset(
                source, root / "out", massive_en_path=english, seed=1,
                action_count=1, extraction_count=0, negative_count=0,
                multi_count=0, english_count=0, augmentation_count=24,
            )
            train = [json.loads(line) for line in Path(outputs["train"]).read_text().splitlines()]
        self.assertEqual(len(train), 25)
        self.assertTrue(any(row["source"] == "deterministic-dutch-augmentation-v1" for row in train))


if __name__ == "__main__":
    unittest.main()
