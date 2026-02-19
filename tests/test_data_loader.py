import tempfile
import unittest
from pathlib import Path

from data_loader import load_text_dataset_from_local, normalize_to_text_column


def _has_datasets() -> bool:
    try:
        import datasets  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(_has_datasets(), "datasets is not installed")
class DataLoaderTests(unittest.TestCase):
    def test_loads_local_txt(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.txt"
            p.write_text("hello\nworld\n", encoding="utf-8")

            ds = load_text_dataset_from_local(str(p), file_type="txt")
            ds = normalize_to_text_column(ds, "text")
            self.assertIn("text", ds.column_names)
            self.assertGreater(len(ds), 0)


if __name__ == "__main__":
    unittest.main()

