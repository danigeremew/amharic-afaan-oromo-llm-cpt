# Pretraining (CPT)

This repo includes a **continuous pretraining (CPT)** pipeline and local inference/export helpers.

## What you edit

Edit `train_config.yaml`:

- `model_id`: Hugging Face *trainable* base checkpoint (example: `Qwen/Qwen2-0.5B`)
- Dataset (Hugging Face):
  - `dataset_source: "hf"`
  - `hf_dataset_id`, `hf_dataset_subset`, `hf_split`
  - `text_column` (set this if the dataset uses a different field than `"text"`)
- Dataset (local files):
  - `dataset_source: "local"`
  - `local_dataset_path` (file or directory)
  - `local_file_type` (`jsonl|csv|parquet|txt`)
  - `text_column`

## Install (one time)

From repo root:

```powershell
python -m pip install -r requirements-pretraining.txt
```

Notes:
- Your current Python already has `torch`, `transformers`, `accelerate`, and `bitsandbytes` installed, but `datasets`/`peft` are required for CPT.
- If you prefer a venv, create/activate one first, then run the same installs.

## Run CPT

Smoke test (quick):

```powershell
python run_cpt.py --max_steps_override 20
```

Full run:

```powershell
python run_cpt.py
```

Custom config path:

```powershell
python run_cpt.py --config path\to\train_config.yaml
```

Outputs are written under `outputs/<run_name>/` and should include LoRA adapter files (not a fully merged model).

## Run inference (adapter)

One prompt:

```powershell
python infer.py --adapter outputs/cpt_run_001 --base-model Qwen/Qwen2-0.5B --prompt "አማርኛ ስለ ቴክኖሎጂ አጭር ጽሑፍ ጻፍ።"
```

Interactive chat:

```powershell
python infer.py --adapter outputs/cpt_run_001 --base-model Qwen/Qwen2-0.5B --interactive
```

## Export GGUF (optional)

This requires a local `llama.cpp` checkout that has been built.

Export merged FP16 GGUF:

```powershell
python export_gguf.py --adapter outputs/cpt_run_001 --base-model Qwen/Qwen2-0.5B --llama-cpp-dir C:\path\to\llama.cpp
```

Export and quantize (example `Q4_K_M`):

```powershell
python export_gguf.py --adapter outputs/cpt_run_001 --base-model Qwen/Qwen2-0.5B --llama-cpp-dir C:\path\to\llama.cpp --quant Q4_K_M --quant-gguf outputs/gguf/cpt_run_001-q4_k_m.gguf
```

## Important limitations

- **Ollama models are not directly trainable here.** Use a Hugging Face checkpoint as `model_id`. (You can export later for serving.)
- `streaming: true` is supported, but `pack_sequences: true` requires `streaming: false`.

