from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _empty_str_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


class TrainConfig(BaseModel):
    run_name: str = Field(default="cpt_run_001")
    output_dir: str = Field(default="pretraining/outputs")
    seed: int = Field(default=42)

    model_id: str
    trust_remote_code: bool = Field(default=False)
    resume_from_checkpoint: str | None = Field(default=None)

    dataset_source: Literal["hf", "local"] = Field(default="hf")

    hf_dataset_id: str | None = Field(default=None)
    hf_dataset_subset: str | None = Field(default=None)
    hf_split: str = Field(default="train")
    streaming: bool = Field(default=False)

    local_dataset_path: str | None = Field(default=None)
    local_file_type: Literal["jsonl", "csv", "parquet", "txt"] = Field(default="jsonl")
    text_column: str = Field(default="text")

    seq_len: int = Field(default=512, ge=16)
    pack_sequences: bool = Field(default=True)
    num_proc: int = Field(default=1, ge=1)

    use_4bit: bool = Field(default=True)
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    lora_target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    learning_rate: float = Field(default=2e-4, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    lr_scheduler_type: str = Field(default="cosine")
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    max_steps: int = Field(default=1000, ge=1)
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=200, ge=1)
    save_total_limit: int = Field(default=2, ge=1)
    fp16: bool = Field(default=True)
    bf16: bool = Field(default=False)
    gradient_checkpointing: bool = Field(default=True)

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model_id must be a non-empty string (Hugging Face model ID).")
        return value

    @field_validator("resume_from_checkpoint", mode="before")
    @classmethod
    def _normalize_resume_path(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("resume_from_checkpoint must be a string path or empty.")
        return _empty_str_to_none(value)

    @field_validator("hf_dataset_id", "hf_dataset_subset", "local_dataset_path", mode="before")
    @classmethod
    def _normalize_optional_str(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("Expected a string.")
        return _empty_str_to_none(value)

    @field_validator("lora_target_modules", mode="before")
    @classmethod
    def _parse_lora_targets(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",")]
            return [p for p in parts if p]
        if isinstance(value, list):
            normalized: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    raise TypeError("lora_target_modules list items must be strings.")
                item = item.strip()
                if item:
                    normalized.append(item)
            return normalized
        raise TypeError("lora_target_modules must be a comma-separated string or list of strings.")

    @model_validator(mode="after")
    def _validate_dataset_settings(self) -> "TrainConfig":
        if self.dataset_source == "hf":
            if not self.hf_dataset_id:
                raise ValueError("hf_dataset_id is required when dataset_source=hf.")
        else:
            if not self.local_dataset_path:
                raise ValueError("local_dataset_path is required when dataset_source=local.")

            path = Path(self.local_dataset_path)
            if not path.exists():
                raise ValueError(f"local_dataset_path does not exist: {path}")

        if self.streaming and self.pack_sequences:
            raise ValueError("streaming=true is not compatible with pack_sequences=true (disable packing).")

        if self.fp16 and self.bf16:
            raise ValueError("Set only one: fp16=true or bf16=true (not both).")

        return self

