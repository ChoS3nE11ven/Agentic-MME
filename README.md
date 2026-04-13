# Agentic-MME: What Agentic Capability Really Brings to Multimodal Intelligence?

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge)](https://agenticmme.github.io/)
[![Paper](https://img.shields.io/badge/Paper-arXiv%202604.03016-red?style=for-the-badge)](https://arxiv.org/abs/2604.03016)
[![HF Paper](https://img.shields.io/badge/HF-Paper-yellow?style=for-the-badge)](https://huggingface.co/papers/2604.03016)
[![HF Dataset](https://img.shields.io/badge/HF-Dataset-orange?style=for-the-badge)](https://huggingface.co/datasets/Crystal1047/Agentic-MME)

[![Tasks](https://img.shields.io/badge/Tasks-418-0A7B83?style=flat-square)](https://huggingface.co/datasets/Crystal1047/Agentic-MME)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-2000%2B-1D3557?style=flat-square)](https://arxiv.org/pdf/2604.03016)
[![Domains](https://img.shields.io/badge/Domains-6-457B9D?style=flat-square)](https://arxiv.org/pdf/2604.03016)
[![Levels](https://img.shields.io/badge/Levels-3-E76F51?style=flat-square)](https://arxiv.org/pdf/2604.03016)

[[Project Page](https://agenticmme.github.io/)] [[Paper PDF](https://arxiv.org/pdf/2604.03016)] [[Dataset](https://huggingface.co/datasets/Crystal1047/Agentic-MME)]

</div>

<p align="center">
  <img src="./assets/case.png" alt="Agentic-MME case study" width="920"/>
</p>

---

## News

- **`2026.04.06`** Agentic-MME released on arXiv.
- **`2026.04.06`** Official benchmark code and dataset usage pipeline released.

---

## Contents

- [Agentic-MME Overview](#agentic-mme-overview)
- [Why Agentic-MME Is Different](#why-agentic-mme-is-different)
- [Dataset at a Glance](#dataset-at-a-glance)
- [Dataset Usage](#dataset-usage)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Evaluation](#evaluation)
- [Sharded Runs](#sharded-runs)
- [Available Tools](#available-tools)
- [Output Files](#output-files)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## Agentic-MME Overview

Agentic-MME is a **process-verified benchmark** for evaluating multimodal agentic capabilities.

Unlike final-answer-only benchmarks, Agentic-MME explicitly evaluates whether a model can:

1. **see better** through active visual operations (Visual Expansion), and
2. **know better** through open-web retrieval (Knowledge Expansion),

while maintaining correct multi-turn reasoning under realistic interaction budgets.

> Highlights:
> - 🧠 Process-level auditing instead of answer-only grading
> - 🖼️ Visual operation correctness and artifact verification
> - 🌐 Search strategy and evidence quality verification
> - ⏱️ Efficiency diagnosis via overthinking vs. human trajectories

---

## Why Agentic-MME Is Different

### 1) Two core capability dimensions

- 🖼️ **Visual Expansion**: active image manipulation (crop/rotate/enhance/...) to reveal latent visual cues.
- 🌐 **Knowledge Expansion**: open-web retrieval (search/lens/webpage reading) to obtain external evidence beyond parametric memory.

### 2) Progressive task difficulty (not a flat score)

- `Level 1`: single decisive visual operation.
- `Level 2`: short multi-step visual + retrieval workflow.
- `Level 3`: iterative, interleaved visual-retrieval synergy under ambiguity.

### 3) Process-level dual-axis scoring

| Axis | What it evaluates | Typical signals |
|---|---|---|
| **S-axis (Strategy)** | Whether retrieval plans/actions are correct and useful | query intent, keyword quality, URL/evidence correctness |
| **V-axis (Visual)** | Whether visual operations are correctly executed and produce valid visual evidence | intermediate artifacts and checkpoint pass/fail |

V-axis is further decomposed for diagnosis:

- `V-tool`: whether expected visual operations were invoked.
- `V-true`: whether produced visual artifacts actually satisfy the checkpoint.

### 4) Efficiency beyond correctness

- ⏱️ **Overthinking** is measured relative to human reference trajectories, capturing redundant or excessive tool usage.

---

## Dataset at a Glance

### Core statistics

| Property | Value |
|---|---:|
| Total tasks | **418** |
| Difficulty levels | **3** |
| Major domains / sub-categories | **6 / 35** |
| Stepwise checkpoints | **2,000+** |
| Human annotation cost | **10+ person-hours per task (avg.)** |
| Total images / tool calls (human trajectories) | **430 / 899** |
| Avg. image resolution | **1952 × 1747** |
| Small-cue cases (`<10%` image area) | **226 (43.1%)** |
| External-search-required tasks | **29.4%** |
| Avg. prompt length / answer length | **31.9 / 1.5 tokens** |

### Difficulty distribution

| Level | Share | Avg checkpoints / task | Avg tool calls / task | Characterization |
|---|---:|---:|---:|---|
| **L1 (Easy)** | 48.6% | 2.89 | 1.21 | single decisive visual operation |
| **L2 (Mid)** | 32.1% | 4.64 | 2.42 | short multi-step visual + retrieval workflow |
| **L3 (Hard)** | 19.4% | 6.67 | 4.07 | advanced synergistic, interleaved reasoning |

### Domain distribution (6 major domains)

| Domain | Share |
|---|---:|
| **Diagram** | 21.3% |
| **Finance** | 19.9% |
| **Society** | 19.4% |
| **Life** | 14.4% |
| **Culture** | 12.9% |
| **Science** | 12.2% |

---

## Dataset Usage

### 🤗 Load from Hugging Face

```python
from datasets import load_dataset

ds = load_dataset("Crystal1047/Agentic-MME", split="train")
print(ds)
print(ds.features)
```

Current split/fields:

- Split: `train` (`418` examples)
- Fields:
  - `image`
  - `task_id`
  - `question`
  - `match_type`
  - `golden_answer`
  - `question_images`
  - `visual_clues`
  - `search_evidence`

### 🗂️ Official dataset folder layout

```text
<dataset_root>/
├── image_cause/
├── images/
├── json/
└── search_url/
```

Folder purpose:

| Folder | Purpose | Used by runner |
|---|---|---|
| `json/` | Task configuration files (`*.json`) | ✅ Required |
| `images/` | Input images referenced by task IDs | ✅ Required |
| `image_cause/` | Auxiliary image evidence/metadata | ⚪ Optional (analysis/inspection) |
| `search_url/` | Retrieval evidence metadata | ⚪ Optional (analysis/inspection) |

### 🧭 Recommended runner invocation with this layout

```bash
python general/run_general_script_openai.py \
  --task_dir <dataset_root>/json \
  --dataset_root <dataset_root> \
  --images_dir <dataset_root>/images \
  --model gpt-4o \
  --api_config configs/api.json
```

Notes:

- `--images_dir` is optional, but passing it explicitly avoids ambiguity.
- `--dataset_root` should point to the folder that directly contains `json/` and `images/`.
- If omitted, `dataset_root` is inferred from the task JSON path.

---

## Environment Setup

```bash
git clone https://github.com/ChoS3nE11ven/Agentic-MME.git
cd Agentic-MME
conda create -n agenticmme python=3.9 -y
conda activate agenticmme
pip install -r requirements.txt
```

For local models (Thyme/DeepEyes):

```bash
pip install torch transformers accelerate
```

---

## Configuration

### 🔐 OpenAI API

```bash
cp configs/api.json.example configs/api.json
```

Then edit `configs/api.json` with your key/base URL.

### 🌐 Web retrieval config

Edit `configs/search_config.json`:

- `serper_api_key`: [Serper.dev](https://serper.dev) key for Google Search + Google Lens
- `imgbb_api_key`: [ImgBB](https://imgbb.com) key for image upload
- `jina_api_key`: [Jina Reader](https://jina.ai/reader) key for webpage content extraction

If retrieval is not needed, simply omit `--enable_search`.

---

## Running Experiments

### 🚀 Quick sanity check (single task)

#### General mode + OpenAI

```bash
python general/run_general_script_openai.py \
  --task_json <task_json> \
  --dataset_root <dataset_root> \
  --images_dir <dataset_root>/images \
  --model gpt-4o \
  --api_config configs/api.json \
  --enable_search \
  --search_config configs/search_config.json \
  --max_rounds 15 \
  --max_tool_calls 15
```

#### Atomic mode + OpenAI

```bash
python atomic/run_atomic_tools_openai.py \
  --task_json <task_json> \
  --dataset_root <dataset_root> \
  --images_dir <dataset_root>/images \
  --model gpt-4o-mini \
  --api_config configs/api.json \
  --enable_search \
  --search_config configs/search_config.json \
  --max_rounds 15 \
  --max_tool_calls 15
```

### 📦 Batch run (general mode)

#### OpenAI

```bash
python general/run_general_script_openai.py \
  --task_dir <dataset_root>/json \
  --dataset_root <dataset_root> \
  --images_dir <dataset_root>/images \
  --model gpt-4o \
  --api_config configs/api.json \
  --enable_search \
  --search_config configs/search_config.json \
  --max_rounds 15 \
  --max_tool_calls 15 \
  --skip_existing
```

#### Local models

```bash
# Thyme
python -m general.run_general_script_thyme \
  --task_dir <dataset_root>/json \
  --dataset_root <dataset_root> \
  --images_dir <dataset_root>/images \
  --model_path /path/to/thyme-model \
  --enable_search \
  --search_config configs/search_config.json \
  --skip_existing

# DeepEyes
python -m general.run_general_script_deepeyes \
  --task_dir <dataset_root>/json \
  --dataset_root <dataset_root> \
  --images_dir <dataset_root>/images \
  --model_path /path/to/deepeyes-model \
  --enable_search \
  --search_config configs/search_config.json \
  --skip_existing
```

### ⚙️ Batch run (atomic mode)

#### OpenAI

```bash
python atomic/run_atomic_tools_openai.py \
  --task_dir <dataset_root>/json \
  --dataset_root <dataset_root> \
  --images_dir <dataset_root>/images \
  --model gpt-4o-mini \
  --api_config configs/api.json \
  --enable_search \
  --search_config configs/search_config.json \
  --max_rounds 15 \
  --max_tool_calls 15 \
  --skip_existing
```

#### Local models

```bash
# Thyme
python atomic/run_atomic_tools_thyme.py \
  --task_dir <dataset_root>/json \
  --dataset_root <dataset_root> \
  --images_dir <dataset_root>/images \
  --model_path /path/to/thyme-model \
  --enable_search \
  --search_config configs/search_config.json \
  --skip_existing

# DeepEyes
python atomic/run_atomic_tools_deepeyes.py \
  --task_dir <dataset_root>/json \
  --dataset_root <dataset_root> \
  --images_dir <dataset_root>/images \
  --model_path /path/to/deepeyes-model \
  --enable_search \
  --search_config configs/search_config.json \
  --skip_existing
```

### 🧩 Important arguments

| Argument | Default | Description |
|---|---|---|
| `--task_dir` | - | Directory of task JSONs |
| `--task_json` | - | Single task JSON path |
| `--dataset_root` | inferred | Root used to resolve dataset files |
| `--images_dir` | - | Optional explicit image directory |
| `--out_dir` | auto (`runs/general/{model}` or `runs/atomic/{model}`) | Output directory |
| `--model` | script-dependent | OpenAI model name |
| `--model_path` | script-dependent | Local model path/ID |
| `--api_config` | - | API config JSON |
| `--enable_search` | `false` | Enable search/lens/webpage tools |
| `--search_config` | - | Search config JSON |
| `--max_rounds` | `15` | Max dialogue rounds |
| `--max_tool_calls` | `15` | Max tool invocations |
| `--skip_existing` | `false` | Skip completed tasks |
| `--task_delay` | `2.0` | Delay between tasks to avoid rate limits |
| `--max_retries` | `3` | Retry times for rate-limit errors |
| `--max_tasks` | `0` | Process first N tasks (`0` = all) |
| `--shard` | `0` | Shard index |
| `--num_shards` | `1` | Total shard count |

---

## Evaluation

### 📊 Step 1: score runs

```bash
python evaluation/eval_runs_search.py \
  --runs_dir runs/general/gpt-4o \
  --api_config configs/api.json
```

Default summary output:

```text
runs/scores/general_gpt-4o_scored.json
```

When running with shards (`--num_shards > 1`), each shard writes:

```text
runs/scores/general_gpt-4o_shard{shard_idx}_scored.json
```

### 🔍 Step 2: optional V-axis breakdown

```bash
python evaluation/analyze_v.py \
  runs/scores/general_gpt-4o_scored.json \
  <dataset_root>/json
```

This script breaks V-axis into `tool-use` and `visual-check`, and reports `Overall / L1 / L2 / L3`.

### 🧮 Step 3: aggregate from per-task results

```bash
python evaluation/aggregate_shards.py \
  --runs_dir runs/general/gpt-4o
```

`aggregate_shards.py` reads each run folder's `result_scored.json` and produces:

```text
runs/scores/general_gpt-4o_aggregated.json
```

---

## Sharded Runs

All major run/eval scripts support `--shard` and `--num_shards`.

> Sharding strategy: contiguous blocks (not interleaved sampling).

### 1) Launch N shards in parallel (run stage)

```bash
NUM_SHARDS=8
MODEL=gpt-4o
DATASET_ROOT=<dataset_root>

mkdir -p logs
for SHARD in $(seq 0 $((NUM_SHARDS-1))); do
  python general/run_general_script_openai.py \
    --task_dir ${DATASET_ROOT}/json \
    --dataset_root ${DATASET_ROOT} \
    --images_dir ${DATASET_ROOT}/images \
    --model ${MODEL} \
    --api_config configs/api.json \
    --enable_search \
    --search_config configs/search_config.json \
    --skip_existing \
    --shard ${SHARD} \
    --num_shards ${NUM_SHARDS} \
    > logs/run_shard_${SHARD}.log 2>&1 &
done
wait
```

### 2) Launch N shards in parallel (evaluation stage)

```bash
NUM_SHARDS=8
MODEL=gpt-4o
RUNS_DIR=runs/general/${MODEL}

mkdir -p logs
for SHARD in $(seq 0 $((NUM_SHARDS-1))); do
  python evaluation/eval_runs_search.py \
    --runs_dir ${RUNS_DIR} \
    --api_config configs/api.json \
    --skip_existing \
    --shard ${SHARD} \
    --num_shards ${NUM_SHARDS} \
    > logs/eval_shard_${SHARD}.log 2>&1 &
done
wait
```

### 3) Aggregate all shard outputs

```bash
python evaluation/aggregate_shards.py \
  --runs_dir runs/general/gpt-4o
```

You can apply exactly the same pattern to atomic mode by replacing script paths and `runs/atomic/<model>`.

---

## Available Tools

### 🖼️ Atomic image tools (visual expansion)

| Tool | Description |
|---|---|
| `crop` | Crop region by normalized coordinates |
| `rotate` | Rotate image by angle |
| `flip` | Horizontal/vertical/both flip |
| `resize` | Resize image |
| `enhance` | Adjust brightness/contrast/sharpness |
| `grayscale` | Convert to grayscale |
| `autocontrast` | Auto contrast |
| `blur` | Gaussian blur |
| `sharpen` | Sharpen |
| `denoise` | Denoise |
| `edge_detect` | Canny/Sobel/simple edge detection |
| `invert` | Invert color |
| `equalize` | Histogram equalization |
| `threshold` | Binarization |

### 🌐 Retrieval tools (knowledge expansion)

| Tool | Description |
|---|---|
| `google_search` | Text web search |
| `google_lens_search` | Reverse image search |
| `fetch_webpage` | Fetch webpage content |

`bbox_2d` uses normalized `[x1, y1, x2, y2]` in `[0,1000]`, where `(0,0)` is top-left and `(1000,1000)` is bottom-right.

---

## Output Files

Each task output is stored under:

```text
runs/{mode}/{model}/{task_id}/
```

Typical files:

| File | Description |
|---|---|
| `orig.png` / `orig_*.png` | Copied original input image(s) |
| `tool_images/` | Intermediate generated images |
| `model_answer.txt` | Final model answer |
| `conversation.json` | Multi-turn full conversation |
| `tool_use_list.json` | Tool call logs for evaluation |
| `run_meta.json` | Run metadata and summary |
| `result_scored.json` | Per-task scored result |
| `raw_model_output_turn_*.txt` | Raw per-turn model output |
| `model_code_turn_*.py` | Executed code per turn (General mode) |

---

## Project Structure

```text
Agentic-MME/
├── atomic/
│   ├── run_atomic_tools_openai.py
│   ├── run_atomic_tools_thyme.py
│   └── run_atomic_tools_deepeyes.py
├── general/
│   ├── run_general_script_openai.py
│   ├── run_general_script_thyme.py
│   └── run_general_script_deepeyes.py
├── evaluation/
│   ├── eval_runs_search.py
│   ├── analyze_v.py
│   └── aggregate_shards.py
├── configs/
│   ├── api.json.example
│   └── search_config.json
├── assets/
│   └── case.png
├── common_utils.py
├── dataset_utils.py
├── ast_ops.py
├── atomic_toolbox.py
├── search_toolbox.py
├── search_tools.py
├── verifiers.py
└── requirements.txt
```

---

## Citation

If you find Agentic-MME useful, please cite:

```bibtex
@article{wei2026agentic,
  title={Agentic-MME: What Agentic Capability Really Brings to Multimodal Intelligence?},
  author={Wei, Qianshan and Yang, Yishan and Wang, Siyi and Chen, Jinglin and Wang, Binyu and Wang, Jiaming and Chen, Shuang and Li, Zechen and Shi, Yang and Tang, Yuqi and others},
  journal={arXiv preprint arXiv:2604.03016},
  year={2026}
}
```
