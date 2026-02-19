from __future__ import annotations

from pathlib import Path
from typing import Any


def _require_datasets() -> Any:
    try:
        import datasets  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: datasets. Install pretraining deps with "
            "`python -m pip install -r requirements-pretraining.txt`."
        ) from e
    return datasets


def _expand_data_files(path: Path, file_type: str) -> list[str]:
    if path.is_file():
        return [str(path)]

    if path.is_dir():
        exts: set[str] = set()
        if file_type == "jsonl":
            exts = {".jsonl", ".json"}
        elif file_type == "csv":
            exts = {".csv"}
        elif file_type == "parquet":
            exts = {".parquet"}
        elif file_type == "txt":
            exts = {".txt"}

        files = []
        for p in sorted(path.rglob("*")):
            if not p.is_file():
                continue
            if exts and p.suffix.lower() not in exts:
                continue
            files.append(str(p))

        if not files:
            raise ValueError(f"No files found under directory: {path}")
        return files

    raise ValueError(f"Invalid local_dataset_path: {path}")


def load_text_dataset_from_hf(
    dataset_id: str,
    subset: str | None,
    split: str,
    streaming: bool,
) -> Any:
    datasets = _require_datasets()
    kwargs: dict[str, Any] = {"split": split, "streaming": streaming}
    if subset:
        return datasets.load_dataset(dataset_id, subset, **kwargs)
    return datasets.load_dataset(dataset_id, **kwargs)


def load_text_dataset_from_local(
    local_dataset_path: str,
    file_type: str,
) -> Any:
    datasets = _require_datasets()
    path = Path(local_dataset_path)
    data_files = _expand_data_files(path, file_type=file_type)

    if file_type == "jsonl":
        return datasets.load_dataset("json", data_files=data_files, split="train")
    if file_type == "csv":
        return datasets.load_dataset("csv", data_files=data_files, split="train")
    if file_type == "parquet":
        return datasets.load_dataset("parquet", data_files=data_files, split="train")
    if file_type == "txt":
        return datasets.load_dataset("text", data_files=data_files, split="train")

    raise ValueError(f"Unsupported local_file_type: {file_type}")


def normalize_to_text_column(dataset: Any, text_column: str) -> Any:
    # datasets.Dataset / datasets.IterableDataset expose column_names.
    column_names = getattr(dataset, "column_names", None)
    if column_names is None:  # pragma: no cover
        raise ValueError("Dataset object missing column_names; unsupported dataset type.")

    if text_column not in column_names:
        raise ValueError(f"text_column '{text_column}' not found. Available columns: {column_names}")

    keep_cols = [text_column]
    remove_cols = [c for c in column_names if c not in keep_cols]

    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)

    if text_column != "text":
        dataset = dataset.rename_column(text_column, "text")

    return dataset
