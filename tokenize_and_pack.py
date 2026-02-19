from __future__ import annotations

from typing import Any


def tokenize_dataset(
    dataset: Any,
    tokenizer: Any,
    seq_len: int,
    *,
    pack_sequences: bool,
    streaming: bool,
    num_proc: int,
) -> Any:
    if streaming and pack_sequences:
        raise ValueError("streaming=true is not compatible with pack_sequences=true.")

    if pack_sequences:
        return _tokenize_and_pack(dataset, tokenizer, seq_len, num_proc=num_proc)

    return _tokenize_no_pack(dataset, tokenizer, seq_len, streaming=streaming, num_proc=num_proc)


def _tokenize_no_pack(dataset: Any, tokenizer: Any, seq_len: int, *, streaming: bool, num_proc: int) -> Any:
    def _tok(batch: dict[str, list[str]]) -> dict[str, Any]:
        # Truncate to seq_len for stable memory usage.
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=seq_len,
            padding=False,
        )

    map_kwargs: dict[str, Any] = {"batched": True, "remove_columns": ["text"]}
    if not streaming:
        map_kwargs["num_proc"] = max(1, int(num_proc))

    return dataset.map(_tok, **map_kwargs)


def _tokenize_and_pack(dataset: Any, tokenizer: Any, seq_len: int, *, num_proc: int) -> Any:
    def _tok(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=False, padding=False)

    tokenized = dataset.map(
        _tok,
        batched=True,
        remove_columns=["text"],
        num_proc=max(1, int(num_proc)),
    )

    # Standard HF pattern: concatenate then chunk into fixed-size blocks.
    def _group_texts(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated: dict[str, list[int]] = {}
        for key, sequences in examples.items():
            flattened: list[int] = []
            for seq in sequences:
                flattened.extend(seq)
            concatenated[key] = flattened

        total_length = len(concatenated["input_ids"])
        if total_length < seq_len:
            return {"input_ids": [], "attention_mask": []}

        total_length = (total_length // seq_len) * seq_len
        result: dict[str, list[list[int]]] = {}
        for key, values in concatenated.items():
            values = values[:total_length]
            result[key] = [values[i : i + seq_len] for i in range(0, total_length, seq_len)]
        return result

    packed = tokenized.map(
        _group_texts,
        batched=True,
        num_proc=max(1, int(num_proc)),
    )

    # Remove empty groups produced when a batch doesn't meet seq_len.
    packed = packed.filter(lambda ex: len(ex["input_ids"]) == seq_len)
    return packed

