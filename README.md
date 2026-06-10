# SFT LLM Distil

Small experimental pipeline for fine-tuning and distilling Qwen models on legal QA data. The main goal was to map and evaluate old document repositories from my Git history by turning repository/document content into grouped yes/no questions, generating answers with a compact model, and comparing them against labeled outputs.

## What It Uses

- **Hugging Face Datasets** to load the QA dataset (`corejur/aviacao_qa_20_v3`).
- **Transformers** and **TRL** for causal language model training.
- **Unsloth** for faster/leaner Qwen loading and fine-tuning.
- **PEFT / LoRA** for adapter-based training.
- **BitsAndBytes** with 4-bit loading to reduce GPU memory usage.
- **Flash Attention 2** for faster inference/training where available.
- **Weights & Biases** for experiment tracking.
- **scikit-learn** for validation reports.

## Repository Flow

1. **Prepare QA data**

   ```bash
   python download_qa_data.py
   ```

   Downloads the Hugging Face dataset and creates train/test JSON files under `data/qa_aviacao/`.

2. **Fine-tune a student model**

   ```bash
   python sft_completion.py
   ```

   Fine-tunes `Qwen/Qwen2-0.5B-Instruct` using completion-only supervised fine-tuning with LoRA adapters.

3. **Run distillation**

   ```bash
   python completions_only_distil.py
   ```

   Distills behavior from a larger Qwen teacher model into the smaller student model using custom KL-based losses from `losses.py` and `custom_trainer.py`.

4. **Generate answers**

   ```bash
   python generate_qa_inference.py
   ```

   Runs the trained adapter over the test prompts and writes model outputs to `outputs/`.

5. **Validate results**

   ```bash
   python validate_qa_responses.py
   ```

   Compares generated JSON answers against the labeled answers and prints classification metrics.

## Environment

Create a Python environment and install the base dependencies:

```bash
pip install -r requirements.txt
```

The scripts expect a `.env` file with:

```env
HF_KEY=your_huggingface_token
WANDB_API_KEY=your_wandb_key
WANDB_PROJECT=your_wandb_project
```

## Notes

The code is research-oriented and uses hardcoded model names, dataset paths, and output paths. It was built for quick iteration while mapping legacy repositories/documents into structured QA evaluations, not as a packaged library.
