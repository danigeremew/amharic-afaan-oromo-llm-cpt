from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and export GGUF using llama.cpp.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2-0.5B", help="HF base model id.")
    parser.add_argument("--adapter", type=Path, default=Path("outputs/cpt_run_001"), help="LoRA adapter directory.")
    parser.add_argument(
        "--merged-out",
        type=Path,
        default=Path("outputs/merged_hf/cpt_run_001"),
        help="Directory to save merged HF model.",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        required=True,
        help="Path to local llama.cpp repository.",
    )
    parser.add_argument(
        "--f16-gguf",
        type=Path,
        default=Path("outputs/gguf/cpt_run_001-f16.gguf"),
        help="Output path for f16 GGUF.",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="",
        help="Optional quantization type (examples: Q8_0, Q6_K, Q5_K_M, Q4_K_M).",
    )
    parser.add_argument(
        "--quant-gguf",
        type=Path,
        default=Path("outputs/gguf/cpt_run_001-q4_k_m.gguf"),
        help="Output path for quantized GGUF (used only when --quant is set).",
    )
    return parser.parse_args()


def _find_converter(llama_cpp_dir: Path) -> Path:
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found in: {llama_cpp_dir}")
    return converter


def _find_quantize_exe(llama_cpp_dir: Path) -> Path:
    candidates = [
        llama_cpp_dir / "build" / "bin" / "llama-quantize.exe",
        llama_cpp_dir / "build" / "bin" / "Release" / "llama-quantize.exe",
        llama_cpp_dir / "build" / "bin" / "quantize.exe",
        llama_cpp_dir / "build" / "bin" / "Release" / "quantize.exe",
    ]
    for path in candidates:
        if path.exists():
            return path

    fallback = shutil.which("llama-quantize") or shutil.which("llama-quantize.exe")
    if fallback:
        return Path(fallback)

    raise FileNotFoundError(
        "Could not find llama-quantize executable. Build llama.cpp first (cmake --build build --config Release)."
    )


def merge_adapter(base_model: str, adapter_dir: Path, merged_out: Path) -> None:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_dir}")

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
    print(f"Loading adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    print("Merging LoRA adapter into base model...")
    merged = model.merge_and_unload()

    print(f"Saving merged HF model to: {merged_out}")
    merged_out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_out), safe_serialization=True)

    tokenizer = None
    if (adapter_dir / "tokenizer_config.json").exists() or (adapter_dir / "tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(str(merged_out))


def convert_to_gguf(converter: Path, merged_out: Path, f16_gguf: Path) -> None:
    f16_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(converter),
        str(merged_out),
        "--outfile",
        str(f16_gguf),
        "--outtype",
        "f16",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def quantize_gguf(quantize_exe: Path, f16_gguf: Path, quant_gguf: Path, quant_type: str) -> None:
    quant_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(quantize_exe),
        str(f16_gguf),
        str(quant_gguf),
        quant_type,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    converter = _find_converter(args.llama_cpp_dir)
    merge_adapter(args.base_model, args.adapter, args.merged_out)
    convert_to_gguf(converter, args.merged_out, args.f16_gguf)

    if args.quant:
        quantize_exe = _find_quantize_exe(args.llama_cpp_dir)
        quantize_gguf(quantize_exe, args.f16_gguf, args.quant_gguf, args.quant.upper())

    print("Done.")
    print(f"Merged HF: {args.merged_out}")
    print(f"GGUF f16: {args.f16_gguf}")
    if args.quant:
        print(f"GGUF {args.quant.upper()}: {args.quant_gguf}")


if __name__ == "__main__":
    main()
