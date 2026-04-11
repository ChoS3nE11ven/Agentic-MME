# Agentic-MME: What Agentic Capability Really Brings to Multimodal Intelligence?

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://agenticmme.github.io/)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2604.03016)
[![HF Paper](https://img.shields.io/badge/Paper-HuggingFace-yellow)](https://huggingface.co/papers/2604.03016)
[![HF Dataset](https://img.shields.io/badge/Dataset-HuggingFace-orange)](https://huggingface.co/datasets/Crystal1047/Agentic-MME)

</div>

This is the official repository for [***Agentic-MME***](https://arxiv.org/pdf/2604.03016), a comprehensive benchmark designed to evaluate the agentic capabilities of Multimodal Large Language Models (MLLMs). As MLLMs evolve from passive observers into active agents, they increasingly solve real-world problems through **Visual Expansion** (invoking visual tools to transform images) and **Knowledge Expansion** (leveraging open-web search). However, existing evaluations fail to capture the synergy between these capabilities or verify whether tools are actually invoked, applied correctly, and used efficiently.

Agentic-MME addresses these gaps with 418 real-world tasks across 6 domains and 3 difficulty levels, featuring over 2,000 stepwise checkpoints (averaging 10+ person-hours of manual annotation per task). Our framework supports both sandboxed code execution and structured tool APIs, enabling true process-level verification through dual-axis evaluation (S-axis for strategy, V-axis for visual operations) and an overthinking metric relative to human trajectories.

![Case Study](./assets/case.png)

## Key Features

- **Multi-turn Dialogue**: Models can call tools, execute code, and observe results multiple times
- **Multi-image Input**: Support for single and multi-image tasks
- **Web Search**: Integrated Google Search and Google Lens (via Serper.dev API)
- **Process Evaluation**: S/V two-axis evaluation system (Strategy, Visual)
- **Parallel Processing**: Support for multi-process parallel processing of large-scale datasets

## Running Modes

| Mode | Script | Description |
|------|--------|-------------|
| **General** | `run_general_script_*.py` | Model writes Python code freely to process images |
| **Atomic** | `run_atomic_tools_*.py` | Model calls predefined tools via function calling |

## Supported Models

| Model | Type | Configuration |
|-------|------|---------------|
| **OpenAI API** | API | `configs/api.json` |
| **Thyme** | Local (Qwen2.5-VL-7B) | `--model_path` |
| **DeepEyes** | Local (Qwen2.5-VL-7B) | `--model_path` |

## :gear: Environment Setup

```bash
git clone https://github.com/ChoS3nE11ven/Agentic-MME.git
cd Agentic-MME
conda create -n agenticmme python=3.9 -y
conda activate agenticmme
pip install -r requirements.txt
```

### For Local Models (Thyme/DeepEyes)

If you plan to run local models, install the additional model dependencies:

```bash
pip install torch transformers accelerate
```

## Configuration Files

### OpenAI API

Copy `configs/api.json.example` to `configs/api.json`, then fill in your key and optional base URL.

### Web Search

Edit `configs/search_config.json` in place if you want search-enabled runs.

- **serper_api_key**: [Serper.dev](https://serper.dev) API key for Google Search and Google Lens
- **imgbb_api_key**: [ImgBB](https://imgbb.com) API key for uploading local images
- **jina_api_key**: [Jina Reader](https://jina.ai/reader) API key for webpage content extraction

If you do not need web search, simply omit `--enable_search`.

## Dataset Layout

The runner expects task JSON files and images to follow the same numeric ID.

Recommended layout:

```text
<dataset_root>/
├── json/
│   ├── 0001.json
│   ├── 0002.json
│   └── ...
└── image/
    ├── image_0001.png
    ├── image_0002.png
    ├── image_0162_1.png
    ├── image_0162_2.png
    └── ...
```

Image resolution rules used by the code:

- `0009.json` -> `image_0009.png`
- `0162.json` -> `image_0162_1.png`, `image_0162_2.png`, ...
- Images can live under `image/` or `images/`
- If `--dataset_root` is omitted, it defaults to the parent of your JSON directory
- If `--images_dir` is provided, it is searched before `dataset_root/image` and `dataset_root/images`

In practice, if your tasks are in `../merged/json`, then `../merged` should usually be passed as `--dataset_root`.

## :rocket: Running Experiments

### Quick Start: Single Task

Use `--task_json` first if you want to confirm your setup before launching a full benchmark run.

**General mode + OpenAI API**

```bash
python general/run_general_script_openai.py \
    --task_json <task_json> \
    --dataset_root <dataset_root> \
    --model gpt-4o \
    --api_config configs/api.json \
    --enable_search \
    --search_config configs/search_config.json \
    --max_rounds 15 \
    --max_tool_calls 15
```

**Atomic mode + OpenAI API**

```bash
python atomic/run_atomic_tools_openai.py \
    --task_json <task_json> \
    --dataset_root <dataset_root> \
    --model gpt-4o-mini \
    --api_config configs/api.json \
    --enable_search \
    --search_config configs/search_config.json \
    --max_rounds 15 \
    --max_tool_calls 15
```

### Batch Run: General Mode

**OpenAI API**

```bash
python general/run_general_script_openai.py \
    --task_dir ../merged/json \
    --dataset_root ../merged \
    --model gpt-4o \
    --api_config configs/api.json \
    --enable_search \
    --search_config configs/search_config.json \
    --max_rounds 15 \
    --max_tool_calls 15 \
    --skip_existing
```

**Local Models**

```bash
# Thyme
python -m general.run_general_script_thyme \
    --task_dir ../merged/json \
    --dataset_root ../merged \
    --model_path /path/to/thyme-model \
    --enable_search \
    --search_config configs/search_config.json \
    --skip_existing

# DeepEyes
python -m general.run_general_script_deepeyes \
    --task_dir ../merged/json \
    --dataset_root ../merged \
    --model_path /path/to/deepeyes-model \
    --enable_search \
    --search_config configs/search_config.json \
    --skip_existing
```

### Batch Run: Atomic Mode

**OpenAI API**

```bash
python atomic/run_atomic_tools_openai.py \
    --task_dir ../merged/json \
    --dataset_root ../merged \
    --api_config configs/api.json \
    --model gpt-4o-mini \
    --enable_search \
    --search_config configs/search_config.json \
    --max_rounds 15 \
    --max_tool_calls 15 \
    --skip_existing
```

**Local Models**

```bash
# Thyme
python atomic/run_atomic_tools_thyme.py \
    --task_dir ../merged/json \
    --dataset_root ../merged \
    --model_path /path/to/thyme-model \
    --enable_search \
    --search_config configs/search_config.json \
    --skip_existing

# DeepEyes
python atomic/run_atomic_tools_deepeyes.py \
    --task_dir ../merged/json \
    --dataset_root ../merged \
    --model_path /path/to/deepeyes-model \
    --enable_search \
    --search_config configs/search_config.json \
    --skip_existing
```

### Important Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task_dir` | - | Directory containing task JSON files |
| `--task_json` | - | Run a single task JSON |
| `--dataset_root` | inferred from `task_json`/`task_dir` | Dataset root used to resolve images |
| `--images_dir` | - | Optional explicit image directory |
| `--out_dir` | auto-generated | Output directory for run results |
| `--model` | `gpt-4.1` or `gpt-4o-mini` | OpenAI model name |
| `--model_path` | script-dependent | Local model path or model ID |
| `--api_config` | - | OpenAI API configuration JSON |
| `--enable_search` | `false` | Enable Google Search / Google Lens / webpage fetching |
| `--search_config` | - | Search configuration JSON |
| `--max_rounds` | `15` | Maximum dialogue turns |
| `--max_tool_calls` | `15` | Maximum tool calls |
| `--skip_existing` | `false` | Skip completed runs |
| `--max_tasks` | `0` | Stop after N tasks (`0` means no limit) |
| `--shard` | `0` | Shard index for sharded runs |
| `--num_shards` | `1` | Total shard count |

## :bar_chart: Evaluation

### Step 1: Score the Runs

Use `eval_runs_search.py` as the main evaluation entry point. It evaluates each run directory, writes a per-task `result_scored.json`, and also writes an overall summary JSON under `runs/scores/`.

```bash
python evaluation/eval_runs_search.py \
    --runs_dir runs/general/gpt-4o \
    --api_config configs/api.json
```

Typical output summary path:

```text
runs/scores/general_gpt-4o_scored.json
```

### Evaluation Dimensions

This framework uses an **S-V two-axis evaluation system**:

#### S-axis (Strategy)

S-axis checks whether the model used the appropriate search actions for the task.

- `google_search`: text search
- `google_lens_search`: reverse image search

#### V-axis (Visual)

V-axis checks whether the model performed the right visual operations and whether the resulting image state matches the task requirement.

It contains two different checkpoint types:

1. `tool-use` checkpoints
These verify whether the expected visual operation was invoked, such as `crop`, `resize`, `rotate`, `flip`, or `enhance`.

2. `visual-check` checkpoints
These use a judge model to inspect generated images and determine whether the visual result satisfies the requirement.

If you prefer the paper-style terminology, these are the two categories usually closest to:

- `V-tool`: tool-use checkpoints
- `V-true`: visual-check checkpoints

### Step 2: Optional V-axis Breakdown

Use `analyze_v.py` after `eval_runs_search.py` if you want a more detailed V-axis checkpoint breakdown.

```bash
python evaluation/analyze_v.py \
    runs/scores/general_gpt-4o_scored.json \
    ../merged/json
```

What this script does:

- Reads the scored summary JSON produced by `eval_runs_search.py`
- Separates V-axis checkpoints into `tool-use` and `visual-check`
- Reports pass rates for `Overall`, `L1`, `L2`, and `L3`
- Optionally uses the task config directory to count incomplete runs as failed V checkpoints

What this script does **not** do:

- It does not rescore runs
- It does not replace the main `eval_runs_search.py` evaluation
- It does not measure final-answer accuracy

### Step 3: Optional Aggregation After Sharded Evaluation

If you ran evaluation in shards, use `aggregate_shards.py` to aggregate per-run `result_scored.json` files from the run directory.

```bash
python evaluation/aggregate_shards.py \
    --runs_dir runs/general/gpt-4o
```

This script is mainly useful when `eval_runs_search.py` was run with `--num_shards > 1`, because each shard summary only contains a subset of runs.

## Advanced: Sharded Runs

Both experiment scripts and the evaluation script support `--shard` and `--num_shards`.

The task list is split into contiguous blocks, not interleaved sampling. For example, with `--num_shards 8`, shard `0` processes the first block of tasks, shard `1` processes the next block, and so on.

Example:

```bash
python general/run_general_script_openai.py \
    --task_dir ../merged/json \
    --dataset_root ../merged \
    --model gpt-4o \
    --api_config configs/api.json \
    --enable_search \
    --search_config configs/search_config.json \
    --skip_existing \
    --shard 0 \
    --num_shards 8
```

Launch the same command again with `--shard 1`, `--shard 2`, ..., up to `--shard 7` if you want all eight shards.

The same pattern also works for `evaluation/eval_runs_search.py`.

## :wrench: Available Tools

### Image Processing Tools (Atomic Mode)

| Tool | Description |
|------|-------------|
| `crop` | Crop image region using normalized coordinates |
| `rotate` | Rotate image by specified angle |
| `flip` | Flip image (horizontal/vertical/both) |
| `resize` | Resize image to specified dimensions |
| `enhance` | Adjust brightness, contrast, sharpness |
| `grayscale` | Convert to grayscale |
| `autocontrast` | Auto contrast adjustment |
| `blur` | Apply Gaussian blur |
| `sharpen` | Sharpen image |
| `denoise` | Remove noise |
| `edge_detect` | Edge detection (canny/sobel/simple) |
| `invert` | Invert colors |
| `equalize` | Histogram equalization |
| `threshold` | Binarization |

### Search Tools

| Tool | Description |
|------|-------------|
| `google_search` | Text-based web search |
| `google_lens_search` | Reverse image search |
| `fetch_webpage` | Fetch webpage content |

**Note:** `bbox_2d` uses normalized coordinates `[x1, y1, x2, y2]` with range `0-1000`, where `(0, 0)` is top-left and `(1000, 1000)` is bottom-right.

## :file_folder: Output Files

Each task is saved under:

```text
runs/{mode}/{model}/{task_id}/
```

Typical contents:

| File | Description |
|------|-------------|
| `orig.png` or `orig_*.png` | Original input image(s) copied into the run directory |
| `tool_images/` | Generated intermediate images |
| `model_answer.txt` | Model's final answer |
| `conversation.json` | Full multi-turn conversation history |
| `tool_use_list.json` | Tool call records used for evaluation |
| `run_meta.json` | Run metadata and summary statistics |
| `result_scored.json` | Per-task evaluation result written by `eval_runs_search.py` |
| `raw_model_output_turn_*.txt` | Raw model output per turn |
| `model_code_turn_*.py` | Executed code per turn in General mode |

## Project Structure

```text
Agentic-MME/
├── atomic/                        # Atomic mode scripts
│   ├── run_atomic_tools_openai.py
│   ├── run_atomic_tools_thyme.py
│   └── run_atomic_tools_deepeyes.py
├── general/                       # General mode scripts
│   ├── run_general_script_openai.py
│   ├── run_general_script_thyme.py
│   └── run_general_script_deepeyes.py
├── evaluation/                    # Evaluation scripts
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

## :page_with_curl: Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{wei2026agentic,
  title={Agentic-MME: What Agentic Capability Really Brings to Multimodal Intelligence?},
  author={Wei, Qianshan and Yang, Yishan and Wang, Siyi and Chen, Jinglin and Wang, Binyu and Wang, Jiaming and Chen, Shuang and Li, Zechen and Shi, Yang and Tang, Yuqi and others},
  journal={arXiv preprint arXiv:2604.03016},
  year={2026}
}
```
