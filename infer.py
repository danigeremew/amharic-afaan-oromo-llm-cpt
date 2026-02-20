from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a base model + LoRA adapter.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2-0.5B", help="HF base model id.")
    parser.add_argument("--adapter", type=Path, default=Path("outputs/cpt_run_001"), help="Path to LoRA adapter dir.")
    parser.add_argument("--prompt", type=str, default="", help="Single prompt to generate from.")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--no-sample", action="store_true", help="Disable sampling (greedy decoding).")
    parser.add_argument("--interactive", action="store_true", help="Start interactive prompt loop.")
    return parser.parse_args()


def load_tokenizer(base_model: str, adapter_path: Path):
    if (adapter_path / "tokenizer_config.json").exists() or (adapter_path / "tokenizer.json").exists():
        return AutoTokenizer.from_pretrained(str(adapter_path))
    return AutoTokenizer.from_pretrained(base_model)


def load_model(base_model: str, adapter_path: Path):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model


def generate_text(model, tokenizer, prompt: str, args: argparse.Namespace) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.no_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main() -> None:
    args = parse_args()

    if not args.adapter.exists():
        raise FileNotFoundError(f"Adapter path not found: {args.adapter}")

    tokenizer = load_tokenizer(args.base_model, args.adapter)
    model = load_model(args.base_model, args.adapter)

    if args.interactive:
        print("Interactive mode. Type /exit to quit.")
        while True:
            prompt = input("\nPrompt> ").strip()
            if not prompt:
                continue
            if prompt.lower() in {"/exit", "exit", "quit"}:
                break
            print("\n" + generate_text(model, tokenizer, prompt, args))
        return

    if not args.prompt.strip():
        raise ValueError("Provide --prompt for one-shot generation, or use --interactive.")

    print(generate_text(model, tokenizer, args.prompt, args))


if __name__ == "__main__":
    main()
