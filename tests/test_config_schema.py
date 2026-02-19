import tempfile
import unittest
from pathlib import Path

from config_schema import TrainConfig


class TrainConfigTests(unittest.TestCase):
    def test_parses_lora_targets_string(self):
        cfg = TrainConfig.model_validate(
            {
                "model_id": "Qwen/Qwen2-0.5B",
                "dataset_source": "hf",
                "hf_dataset_id": "wikitext",
                "hf_dataset_subset": "wikitext-2-raw-v1",
                "hf_split": "train",
                "text_column": "text",
                "lora_target_modules": "q_proj,k_proj , v_proj",
            }
        )
        self.assertEqual(cfg.lora_target_modules, ["q_proj", "k_proj", "v_proj"])

    def test_rejects_streaming_with_packing(self):
        with self.assertRaises(ValueError):
            TrainConfig.model_validate(
                {
                    "model_id": "Qwen/Qwen2-0.5B",
                    "dataset_source": "hf",
                    "hf_dataset_id": "wikitext",
                    "streaming": True,
                    "pack_sequences": True,
                    "text_column": "text",
                }
            )

    def test_rejects_fp16_and_bf16(self):
        with self.assertRaises(ValueError):
            TrainConfig.model_validate(
                {
                    "model_id": "Qwen/Qwen2-0.5B",
                    "dataset_source": "hf",
                    "hf_dataset_id": "wikitext",
                    "fp16": True,
                    "bf16": True,
                    "text_column": "text",
                }
            )

    def test_local_dataset_path_must_exist(self):
        with tempfile.TemporaryDirectory() as td:
            data_path = Path(td) / "data.txt"
            data_path.write_text("hello\nworld\n", encoding="utf-8")

            cfg = TrainConfig.model_validate(
                {
                    "model_id": "Qwen/Qwen2-0.5B",
                    "dataset_source": "local",
                    "local_dataset_path": str(data_path),
                    "local_file_type": "txt",
                    "text_column": "text",
                }
            )
            self.assertEqual(cfg.dataset_source, "local")
            self.assertEqual(Path(cfg.local_dataset_path), data_path)


if __name__ == "__main__":
    unittest.main()

