# Pure 💧 Pytorch 🔥 Reimplementation of the MobileLLM-R1-Pytorch

![MobileLLM overview](assets/asset1.png)

Beginner friendly: if you know `nn.Module`, you’re good. This repo shows a clean PyTorch implementation of a MobileLLM-style decoder and a tiny Hugging Face loader so you can run real checkpoints with minimal code.

## What’s Inside ✨
- Plain PyTorch modules in `layers.py` and `attn_implementation.py`
- A simple config in `config.py` (no magic)
- HF-compatible loader + CLI in `hf_inference.py`

## Install (git+) 📦
- Python 3.9+ and a working PyTorch install.
- Replace the URL with your repo:

```bash
pip install "git+https://github.com/vedantdere/MobileLLM-R1-Pytorch.git"
```

## Run It (CLI) 🚀
- From the Hugging Face Hub:

```bash
mobilellm-hf-infer \
  --model-id facebook/MobileLLM-R1-360M \
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

## Useful Flags 🧰

```text
--dtype float32|float16|bfloat16
--revision <branch|tag|commit>      # with --model-id
--local-files-only                  # use cache only, no network
--strict-load                       # enforce exact key matching
--temperature 0.0                   # greedy, deterministic
--top-k 40                          # sample from top-k tokens
```

## MobileLLM in 10 Lines 🐍

```python
from hf_inference import load_mobilellm, generate_text

model, tokenizer = load_mobilellm("/path/to/hf_dir", device="cuda", dtype="bfloat16")
print(generate_text(model, tokenizer, "What is 2+2?", max_new_tokens=32))
```

## If You’re Curious (but still simple) 🔍
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



## Tips 💡
- Prefer `bfloat16` on modern GPUs, `float32` on CPU.
- Use `--temperature 0` for reproducible outputs.
- Loader ties `lm_head` to embeddings when `tie_word_embeddings` is enabled.

## Troubleshooting 🛠️
- Outputs ignore the prompt? Try `--strict-load` and ensure tokenizer files match the checkpoint.
- Private/gated models: `huggingface-cli login` or set `HF_TOKEN`.

## Entrypoints 🏁
- `mobilellm-hf-infer`: prompt-based generation (Hub or local)


Special thanks to the Hugging Face Transformers team for their incredible open-source contributions, which continue to set the standard for accessibility and innovation in NLP. This reimplementation draws heavy inspiration from their integration but is re-imagined in a simpler, beginner-friendly way to make it easier for newcomers to understand.

A huge appreciation also goes to Meta for their groundbreaking research and open-sourcing efforts, which have accelerated progress in large-scale language models and made cutting-edge technology available to the community.
