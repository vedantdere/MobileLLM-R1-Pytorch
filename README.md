MobileLLM HF Loader + Inference 🚀

Simple, fast utilities to run MobileLLM with Hugging Face–formatted weights. Load from the Hub or a local folder, then generate with a single command.

Features
- Load HF checkpoints (.safetensors or .bin) and config.json
- Run via CLI or Python API
- Pull models directly from the Hugging Face Hub
- Tied-weights handling for lm_head ↔ embed_tokens
- Deterministic decoding and EOS-aware stopping

Table of Contents
- Install
- Quick Start (CLI)
- Python API
- Options & Tips
- Troubleshooting
- Entrypoints

Install
- Requires Python 3.9+ and a working PyTorch install.
- Install directly from git (replace with your repo URL):

```bash
pip install "git+https://github.com/your-org/your-repo.git#egg=mobilellm-hf"
```

Dependencies
- Required: `torch`, `transformers>=4.43`, `huggingface_hub>=0.23`
- Optional: `safetensors` (for `.safetensors` checkpoints)

Quick Start (CLI)
- From a Hugging Face model id:

```bash
mobilellm-hf-infer \
  --model-id org/model \
  --prompt "Hello, MobileLLM!" \
  --max-new-tokens 64 \
  --device cpu
```

- From a local HF directory:

```bash
mobilellm-hf-infer \
  --model-path /path/to/hf_dir \
  --prompt "Summarize this in one sentence:" \
  --device cuda
```

- Useful flags:

```text
--dtype float32|float16|bfloat16
--revision <branch|tag|commit>      # with --model-id
--local-files-only                  # use cache only, no network
--strict-load                       # enforce exact key matching
--temperature 0.0                   # deterministic
--top-k 40                          # sample from top-k tokens
```

Python API
Minimal example:

```python
from hf_inference import load_mobilellm, generate_text

# Load from local HF directory
model, tokenizer = load_mobilellm("/path/to/hf_dir", device="cuda", dtype="bfloat16")

print(generate_text(model, tokenizer, "What is 2+2?", max_new_tokens=32))
```

Or with the Hub in the test utility (which resolves a local path first):

```bash
mobilellm-hf-test --model-id org/model --prompt "Write a haiku about code" --max-new-tokens 24
```

Options & Tips
- Precision: `--dtype float32` for CPU correctness; `bfloat16` is a good default on modern GPUs; `float16` for memory savings.
- Reproducibility: set `--temperature 0` for greedy decoding (no randomness).
- EOS stopping: generation halts early if an EOS id is produced.
- Tokenizer: the loader tries `AutoTokenizer` from the same directory; provide full tokenizer files for best results.

Troubleshooting
- Missing key: `lm_head.weight`
  - The loader ties `lm_head.weight` to `embed_tokens.weight` when `tie_word_embeddings` is enabled, and will synthesize the key during load.
- Output ignores prompt / looks random
  - Use `--strict-load` to catch state_dict mismatches.
  - Try `--temperature 0` (deterministic).
  - Ensure tokenizer files are present and compatible with the checkpoint.
- Private/Gated models from Hub
  - Login: `huggingface-cli login` or set `HF_TOKEN` environment variable.

Entrypoints
- `mobilellm-hf-infer`: run inference from a prompt (Hub or local path)
- `mobilellm-hf-test`: smoke test; falls back to token-id generation if tokenizer is missing

Example Outputs

```bash
$ mobilellm-hf-infer --model-id org/model --prompt "What is 2+2?" --temperature 0 --max-new-tokens 8
Answer: 4
```

License
- Add your license text here.

