# Conda Environments

Two environments are used across the three stages:

| Environment       | Used by                                | Key packages                                          |
|-------------------|----------------------------------------|-------------------------------------------------------|
| `l_embed_new`     | Stage 1 (devis) and Stage 2 (embeds)   | Python 3.10 · PyTorch 2.5.1 + CUDA 12.4 · transformers 4.51.3 · accelerate 1.6.0 · flash-attn |
| `llava_embed`     | Stage 3 (PID estimation)               | Python 3.10 · PyTorch 2.1.2 · numpy · matplotlib · tqdm |

The two stages are kept separate because the VLM inference code needs a newer
`transformers` + CUDA 12.4 + flash-attention build, while the PID estimator is
a self-contained PyTorch script and works on the older stack.

## Re-create

```bash
conda env create -f envs/l_embed_new.yml
conda env create -f envs/llava_embed.yml
```

If `flash-attn` fails to install from the yml (prebuilt wheels may not match
your CUDA/python combo), install it manually after the base env is built:

```bash
conda activate l_embed_new
pip install flash-attn==<version> --no-build-isolation
```

The exact version is listed inside `l_embed_new.yml` under the `pip:` section.

## Exports were generated with

```bash
conda env export --no-builds > envs/<name>.yml
```

Build strings are stripped (`--no-builds`) so the environments are a little more
portable across cluster images; package versions are still pinned exactly.
