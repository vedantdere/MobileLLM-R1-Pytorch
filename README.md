Pure 💧 Pytorch 🔥 Reimplementation of the MobileLLM-R1-Pytorch

Beginner friendly: if you know `nn.Module`, you’re good. This repo shows a clean PyTorch implementation of a MobileLLM-style decoder and a tiny Hugging Face loader so you can run real checkpoints with minimal code.

What’s Inside
- Plain PyTorch modules in `layers.py` and `attn_implementation.py`
- A simple config in `config.py` (no magic)
- HF-compatible loader + CLI in `hf_inference.py`
- A tiny smoke test in `test_hf_inference.py`

Install (git+)
- Python 3.9+ and a working PyTorch install.
- Replace the URL with your repo:

```bash
pip install "git+https://github.com/your-org/your-repo.git#egg=mobilellm-hf"
```

Run It (CLI)
- From the Hugging Face Hub:

```bash
mobilellm-hf-infer \
  --model-id org/model \
  --prompt "Hello, MobileLLM!" \
  --max-new-tokens 64 \
  --device cpu
```

- From a local folder (HF format):

```bash
mobilellm-hf-infer \
  --model-path /path/to/hf_dir \
  --prompt "Summarize in one line:" \
  --device cuda
```

Useful Flags

```text
--dtype float32|float16|bfloat16
--revision <branch|tag|commit>      # with --model-id
--local-files-only                  # use cache only, no network
--strict-load                       # enforce exact key matching
--temperature 0.0                   # greedy, deterministic
--top-k 40                          # sample from top-k tokens
```

Python in 10 Lines

```python
from hf_inference import load_mobilellm, generate_text

model, tokenizer = load_mobilellm("/path/to/hf_dir", device="cuda", dtype="bfloat16")
print(generate_text(model, tokenizer, "What is 2+2?", max_new_tokens=32))
```

If You’re Curious (but still simple)
- Core blocks are standard `nn.Module`s:

```python
# layers.py (simplified)
class Llama4ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Llama4TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask, position_ids, past_key_values,
                inputs_embeds, labels, use_cache, cache_position, **kwargs):
        hidden = self.model(
            input_ids, attention_mask, position_ids, past_key_values,
            inputs_embeds, use_cache, cache_position, **kwargs
        )[0]
        return self.lm_head(hidden)
```

Bring Your Own Weights
- Put HF-style files in a folder: `config.json`, `model.safetensors` or `pytorch_model*.bin`, plus tokenizer files.
- Then run:

```bash
mobilellm-hf-infer --model-path /path/to/hf_dir --prompt "Hi" --temperature 0
```

Smoke Test

```bash
# Hub
mobilellm-hf-test --model-id org/model --prompt "Write a haiku about code" --max-new-tokens 24

# Local
mobilellm-hf-test --model-path /path/to/hf_dir --device cpu
```

Tips
- Prefer `bfloat16` on modern GPUs, `float32` on CPU.
- Use `--temperature 0` for reproducible outputs.
- Loader ties `lm_head` to embeddings when `tie_word_embeddings` is enabled.

Troubleshooting
- Outputs ignore the prompt? Try `--strict-load` and ensure tokenizer files match the checkpoint.
- Private/gated models: `huggingface-cli login` or set `HF_TOKEN`.

Entrypoints
- `mobilellm-hf-infer`: prompt-based generation (Hub or local)
- `mobilellm-hf-test`: quick smoke test; falls back to token-IDs if needed

License
- Add your license text here.
