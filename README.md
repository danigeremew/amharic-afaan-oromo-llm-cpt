# Pretraining (CPT)

This folder adds a **continuous pretraining (CPT)** pipeline to the repo. It is separate from the retrieval-only chatbot in `app/`.

## What you edit

Edit `pretraining/config/train_config.yaml`:

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

## Important limitations

- **Ollama models are not directly trainable here.** Use a Hugging Face checkpoint as `model_id`. (You can export later for serving.)
- `streaming: true` is supported, but `pack_sequences: true` requires `streaming: false`.

