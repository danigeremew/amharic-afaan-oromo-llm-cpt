import tempfile
import unittest
from pathlib import Path

from config_loader import load_train_config


class TrainConfigLoaderTests(unittest.TestCase):
    def test_loads_yaml_config(self):
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.yaml"
            cfg_path.write_text(
                "\n".join(
                    [
                        'run_name: "test_run"',
                        'output_dir: "pretraining/outputs"',
                        'model_id: "Qwen/Qwen2-0.5B"',
                        'dataset_source: "hf"',
                        'hf_dataset_id: "wikitext"',
                        'hf_dataset_subset: "wikitext-2-raw-v1"',
                        'hf_split: "train"',
                        'text_column: "text"',
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_train_config(cfg_path)
            self.assertEqual(cfg.run_name, "test_run")
            self.assertEqual(cfg.dataset_source, "hf")
            self.assertEqual(cfg.hf_dataset_id, "wikitext")


if __name__ == "__main__":
    unittest.main()

