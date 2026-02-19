from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from config_loader import load_train_config  # noqa: E402
from data_loader import (  # noqa: E402
    load_text_dataset_from_hf,
    load_text_dataset_from_local,
    normalize_to_text_column,
)
from tokenize_and_pack import tokenize_dataset  # noqa: E402


def _require_torch_and_transformers() -> tuple[Any, Any]:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: torch. Install a CUDA-enabled PyTorch build.") from e

    try:
        import transformers  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: transformers. Install it with `python -m pip install transformers`."
        ) from e

    return torch, transformers


def _require_peft() -> tuple[Any, Any, Any]:
    try:
        from peft import (  # type: ignore
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: peft. Install pretraining deps with "
            "`python -m pip install -r requirements-pretraining.txt`."
        ) from e

    return LoraConfig, get_peft_model, prepare_model_for_kbit_training


def _set_seed(seed: int, torch_mod: Any) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    torch_mod.manual_seed(seed)
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(seed)


def _has_tensorboard() -> bool:
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401

        return True
    except Exception:
        return False


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None

    def _step(p: Path) -> int:
        try:
            return int(p.name.split("-")[-1])
        except Exception:
            return -1

    checkpoints.sort(key=_step)
    return checkpoints[-1]


@dataclass(frozen=True)
class _Runtime:
    torch: Any
    transformers: Any


def _build_tokenizer(runtime: _Runtime, model_id: str, trust_remote_code: bool) -> Any:
    tokenizer = runtime.transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    return tokenizer


def _build_model(runtime: _Runtime, cfg: Any, dtype: Any) -> Any:
    torch_mod = runtime.torch
    transformers = runtime.transformers

    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(cfg.trust_remote_code),
        "device_map": "auto",
    }

    if cfg.use_4bit:
        try:
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            kwargs["quantization_config"] = bnb_config
        except Exception as e:
            raise RuntimeError(
                "use_4bit=true but failed to configure 4-bit loading. "
                "Make sure bitsandbytes is installed and CUDA is working."
            ) from e

    model = transformers.AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)

    if getattr(model, "config", None) is not None:
        model.config.use_cache = False

    if cfg.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    LoraConfig, get_peft_model, prepare_model_for_kbit_training = _require_peft()
    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(cfg.gradient_checkpointing),
        )

    lora_cfg = LoraConfig(
        r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        target_modules=list(cfg.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    if torch_mod.cuda.is_available():
        free_gb = torch_mod.cuda.mem_get_info()[0] / (1024**3)
        print(f"CUDA available: {torch_mod.cuda.get_device_name(0)} (free ~{free_gb:.2f} GB)")
    else:
        print("CUDA not available; training will run on CPU (very slow).")

    return model


def _load_and_prepare_dataset(cfg: Any, tokenizer: Any) -> Any:
    if cfg.dataset_source == "hf":
        ds = load_text_dataset_from_hf(
            dataset_id=cfg.hf_dataset_id,
            subset=cfg.hf_dataset_subset,
            split=cfg.hf_split,
            streaming=bool(cfg.streaming),
        )
    else:
        ds = load_text_dataset_from_local(
            local_dataset_path=cfg.local_dataset_path,
            file_type=cfg.local_file_type,
        )

    ds = normalize_to_text_column(ds, cfg.text_column)

    tokenized = tokenize_dataset(
        dataset=ds,
        tokenizer=tokenizer,
        seq_len=int(cfg.seq_len),
        pack_sequences=bool(cfg.pack_sequences),
        streaming=bool(cfg.streaming),
        num_proc=int(cfg.num_proc),
    )

    return tokenized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous pretraining (CPT) with LoRA/QLoRA.")
    parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
        help="Path to YAML/JSON config.",
    )
    parser.add_argument(
        "--max_steps_override",
        type=int,
        default=None,
        help="Override max_steps for a quick smoke run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)

    torch_mod, transformers = _require_torch_and_transformers()
    runtime = _Runtime(torch=torch_mod, transformers=transformers)
    _set_seed(int(cfg.seed), torch_mod)

    dtype = torch_mod.float32
    if cfg.bf16:
        dtype = torch_mod.bfloat16
    elif cfg.fp16:
        dtype = torch_mod.float16

    run_out_dir = Path(cfg.output_dir) / cfg.run_name
    run_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name: {cfg.run_name}")
    print(f"Output dir: {run_out_dir}")
    print(f"Model: {cfg.model_id}")
    print(
        f"Dataset: {cfg.dataset_source} "
        f"({cfg.hf_dataset_id if cfg.dataset_source == 'hf' else cfg.local_dataset_path})"
    )

    tokenizer = _build_tokenizer(runtime, cfg.model_id, bool(cfg.trust_remote_code))
    model = _build_model(runtime, cfg, dtype=dtype)

    tokenized = _load_and_prepare_dataset(cfg, tokenizer)

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    max_steps = int(args.max_steps_override) if args.max_steps_override else int(cfg.max_steps)
    report_to: list[str] = ["tensorboard"] if _has_tensorboard() else []

    training_args = transformers.TrainingArguments(
        output_dir=str(run_out_dir),
        overwrite_output_dir=False,
        max_steps=max_steps,
        per_device_train_batch_size=int(cfg.per_device_train_batch_size),
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        learning_rate=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        warmup_ratio=float(cfg.warmup_ratio),
        lr_scheduler_type=str(cfg.lr_scheduler_type),
        logging_steps=int(cfg.logging_steps),
        save_steps=int(cfg.save_steps),
        save_total_limit=int(cfg.save_total_limit),
        fp16=bool(cfg.fp16),
        bf16=bool(cfg.bf16),
        gradient_checkpointing=bool(cfg.gradient_checkpointing),
        report_to=report_to,
        logging_dir=str(run_out_dir / "logs"),
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
        save_safetensors=True,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    resume_path = cfg.resume_from_checkpoint
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
    else:
        latest = _find_latest_checkpoint(run_out_dir)
        if latest is not None:
            resume_path = str(latest)
            print(f"Auto-resuming from latest checkpoint: {resume_path}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(str(run_out_dir))
    tokenizer.save_pretrained(str(run_out_dir))

    print("Training complete.")
    print(f"Saved adapter/tokenizer to: {run_out_dir}")


if __name__ == "__main__":
    main()
