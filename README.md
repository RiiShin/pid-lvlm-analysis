# pid-lvlm-analysis

Official repository for **"A Comprehensive Information-Decomposition Analysis of Large Vision-Language Models"** (ICLR 2026).

Pipeline for decomposing multi-modal information in VLMs into Partial Information
Decomposition (PID) components — Redundancy, Unique (Visual), Unique (Language), Synergy —
across multiple model families (Qwen2-VL, Qwen2.5-VL, InternVL2.5, InternVL3,
LLaMA 3.2 Vision, LLaVA-OV, InstructBLIP, Fuyu) and benchmarks (MMBench, POPE, PMC-VQA).

## Pipeline Overview

The analysis runs in **three stages**:

```
Stage 1 (devis)       Stage 2 (embeds)                       Stage 3 (pid)
────────────────      ────────────────────────────           ────────────────
Compute per-model     For every sample, run 3 inferences     Train CE-alignment
mean/std of visual    and keep features + probabilities:     model on collected
and text token          • full V+L (vl_prob)                 features/probs and
embeddings over the     • language-only (l_prob,             output PID values
train split             image masked by noise sampled          (R, U1, U2, S)
                        from image mean/std)
                      • visual-only (v_prob, text
                        masked by noise sampled from
                        text mean/std)
```

Outputs:
- Stage 1 → `.pt` files containing `{mean, std}` tensors, one per (model, modality).
- Stage 2 → one JSON per model, each sample annotated with `v_feature`, `l_feature`, and the three probability vectors (`v_prob`, `l_prob`, `vl_prob`, plus the un-normalized `*_orig_prob` variants).
- Stage 3 → per-model PID values (Redundancy, Unique-V, Unique-L, Synergy) printed to stdout / logs.

## Repository Layout

```
models_scripts/
├── scripts/
│   ├── devis/      # Bash launchers for Stage 1 (one per dataset/model-family combo)
│   └── embeds/     # Bash launchers for Stage 2 (one per dataset/model combo)
└── general_models/
    ├── devis/      # Python: *_devi.py — computes per-dim mean/std of visual and text embeddings
    └── embeds/     # Python: *_embed.py — three-way inference, dumps features+probs to JSON

pid/
├── mmbench/   # batch_vlm_final_mmbench_drop.py  (num_labels=4, + uniform-failure fix-up)
├── pope/      # batch_vlm_final_pope_drop.py     (num_labels=2)
└── pmc/       # batch_vlm_final_drop.py          (num_labels=4)

utils/
└── output_uni_acc.py  # helper: compute per-modality accuracy from the Stage-2 JSONs

envs/
├── l_embed_new.yml   # env for Stages 1 & 2 (Torch 2.5 + CUDA 12.4, transformers 4.51, flash-attn)
├── llava_embed.yml   # env for Stage 3 (Torch 2.1, numpy, tqdm)
└── README.md         # setup notes
```

## Setup

The pipeline uses two conda environments:

```bash
conda env create -f envs/l_embed_new.yml   # Stages 1 & 2 (devis + embeds)
conda env create -f envs/llava_embed.yml   # Stage 3 (PID)
```

See `envs/README.md` for details and flash-attn installation notes.

## Required Data Layout

The bash launchers assume the following directory layout on disk (paths are configurable via CLI flags on each python script):

```
data/
├── mmbench/
│   ├── mmbench_en_train.json
│   └── mmbench_en_val.json
├── pope/
│   ├── pope_{train,val}.json
│   └── pope_images/...
└── pmc/
    ├── pmc_{train,val}.json
    └── pmc_images/...

results/
├── devis_vector/{dataset}/           # Stage 1 outputs (*.pt)
└── embeds/{dataset}/{train,val}/     # Stage 2 outputs (*.json)
```

Each input JSON is a list of items with:
```json
{
  "image": "<path-or-base64>",
  "num_options": 4,
  "conversations": [
    {"from": "human", "value": "...<image>..."},
    {"from": "gpt",   "value": "A"}
  ]
}
```

## Running the Pipeline

Paths below use placeholders — edit the bash launchers to match your cluster.

### Stage 1 — deviation statistics

```bash
cd models_scripts/general_models/devis
python qwen2_small_devi.py \
    --read_dir   <DATA>/mmbench/mmbench_en_ \
    --image_dir  ""  \
    --vector_save_dir <RESULTS>/devis_vector/mmbench \
    --model_size 2 \
    --data_mode  train
```

Produces `<RESULTS>/devis_vector/mmbench/Qwen_Qwen2-VL-2B-Instruct_train_{text,image}_stats.pt`.

Batch wrappers live in `models_scripts/scripts/devis/` — they launch several model
sizes on different GPUs with `nohup`.

### Stage 2 — three-way inference

```bash
cd models_scripts/general_models/embeds
python qwen2_small_embed.py \
    --read_dir   <DATA>/mmbench/mmbench_en_ \
    --image_dir  ""  \
    --save_dir   <RESULTS>/embeds/mmbench \
    --num_choices 4 \
    --model_qwen_size 2 \
    --data_mode  train \
    --devi_vector_dir <RESULTS>/devis_vector/mmbench
```

Reads the `.pt` stats from Stage 1 and writes
`<RESULTS>/embeds/mmbench/train/qwen2_2.json`.

### Stage 3 — PID

```bash
cd pid/mmbench
python batch_vlm_final_mmbench_drop.py \
    --directory <RESULTS>/embeds/mmbench/ \
    --file_name qwen2_2.json
```

Prints `Redundancy, Unique1, Unique2, Synergy` (in bits).

## Notes

- `num_choices` must match the benchmark: MMBench = 4, POPE = 2, PMC = 4.
- Stage-2 scripts expect stats files named `<clean-model-name>_train_{text,image}_stats.pt`
  (produced by Stage 1 with `--data_mode train`), regardless of which mode is being inferred.
- For MMBench, the Stage-3 script post-processes uniform-failure distributions to the
  canonical `[0.25, 0.25, 0.25, 0.25]` before PID estimation.
- All scripts fix `torch.manual_seed(42)` for reproducibility.
