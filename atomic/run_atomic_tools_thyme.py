#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thyme Local Model Runner for Atomic Tools Mode
Adapted from run_atomic_tools_openai.py to use local Thyme model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import warnings
from transformers import AutoProcessor
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*AutoModelForVision2Seq.*deprecated.*")
warnings.filterwarnings("ignore", message=".*fast processor.*")
warnings.filterwarnings("ignore", message=".*model of type.*")

# Try to import the correct model class
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    MODEL_CLASS = Qwen2_5_VLForConditionalGeneration
except ImportError:
    try:
        from transformers import Qwen2VLForConditionalGeneration
        MODEL_CLASS = Qwen2VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForVision2Seq
        MODEL_CLASS = AutoModelForVision2Seq

from common_utils import ensure_dir, read_json, write_json, image_to_data_url, utc_ts, get_adaptive_image_params
from dataset_utils import resolve_dataset_root, resolve_image_path
from atomic_toolbox import (
    AtomicState, build_atomic_tools_schema,
    tool_crop, tool_rotate, tool_flip, tool_resize, tool_enhance,
    tool_grayscale, tool_autocontrast, tool_blur, tool_sharpen,
    tool_denoise, tool_edge_detect, tool_invert, tool_equalize, tool_threshold
)

# Search tools
from search_tools import SearchTools, load_search_config


# All image tool names
IMAGE_TOOLS = {
    "crop", "rotate", "flip", "resize", "enhance",
    "grayscale", "autocontrast", "blur", "sharpen",
    "denoise", "edge_detect", "invert", "equalize", "threshold"
}

# Maximum number of images that can be downloaded per task
# MAX_DOWNLOAD_IMAGES = 5  # Disabled: download_image tool is commented out

# Thyme-specific system prompt (uses XML tool_log instead of function calling)
# Simplified for 7B models - clear structure, simple examples
THYME_SYSTEM_PROMPT = r'''You are a multimodal reasoning agent that solves visual questions step by step.

You have access to:
1. **Image tools** (via XML tool_log): crop, rotate, flip, resize, enhance, grayscale, autocontrast, blur, sharpen, denoise, edge_detect, invert, equalize, threshold
2. **Search tools** (via XML tool_log): google_search, google_lens_search, fetch_webpage

## Image Management (Simple!)
- Image 0 = original input image
- Image 1, 2, 3... = processed images (created by tools)
- Each tool creates a NEW image with a new index
- After each tool, you'll see: "[Image 1: crop - description]"

## Workflow (ReAct Pattern)

For each step:
1. **Think**: What do I need to do?
2. **Act**: Use tools (image tools OR search tools)
3. **Observe**: Look at the results
4. **Repeat** until you have the answer
5. **Answer**: Give your final answer

## Response Format (Use XML blocks)

<think>
Your reasoning. What do you see? What do you need to do next?
</think>

<tool_log>
[
  {"tool_name": "crop", "arguments": {"image_index": 0, "bbox_2d": [250, 250, 750, 750], "label": "center region"}},
  {"tool_name": "google_search", "arguments": {"query": "your search query"}}
]
</tool_log>

<answer>
Your final answer. Only use when you're done with all tools.
</answer>

## Image Tools - Simple Examples

**crop** - Cut out a region (coordinates: 0-1000 scale)
```
<tool_log>
[{"tool_name": "crop", "arguments": {"image_index": 0, "bbox_2d": [200, 300, 800, 700], "label": "text area"}}]
</tool_log>
```
- bbox_2d: [x1, y1, x2, y2] where 0=left/top, 1000=right/bottom
- [250, 250, 750, 750] = center 50% of image
- **IMPORTANT**: Cropped region must have aspect ratio < 200 (width/height or height/width)
  - ✓ Good: [100, 100, 900, 500] (aspect ratio = 1.6)
  - ✗ Bad: [100, 100, 900, 105] (aspect ratio = 160, too extreme)
  - If you need a narrow region, make it slightly wider/taller to keep aspect ratio reasonable

**rotate** - Rotate the image
```
<tool_log>
[{"tool_name": "rotate", "arguments": {"image_index": 0, "angle": -90, "label": "rotate right"}}]
</tool_log>
```
- angle: positive=counterclockwise, negative=clockwise
- -90 = rotate right, 90 = rotate left

**flip** - Mirror the image
```
<tool_log>
[{"tool_name": "flip", "arguments": {"image_index": 0, "direction": "horizontal", "label": "mirror"}}]
</tool_log>
```
- direction: "horizontal" (left↔right), "vertical" (top↔bottom), "both"

**enhance** - Adjust brightness/contrast/sharpness
```
<tool_log>
[{"tool_name": "enhance", "arguments": {"image_index": 0, "brightness": 1.5, "contrast": 1.2}}]
</tool_log>
```
- 1.0 = no change, >1.0 = increase, <1.0 = decrease

**resize** - Change image size
```
<tool_log>
[{"tool_name": "resize", "arguments": {"image_index": 0, "width": 800, "height": 600}}]
</tool_log>
```

**Other tools** (simple, no extra parameters needed):
- grayscale, autocontrast, sharpen, invert, equalize
- blur, denoise, edge_detect, threshold

## Search Tools - Simple Examples

**google_search** - Search the web
```
<tool_log>
[{"tool_name": "google_search", "arguments": {"query": "Taylor Swift 2025 concert"}}]
</tool_log>
```

**google_lens_search** - Reverse image search (identify objects, logos, text)
```
<tool_log>
[{"tool_name": "google_lens_search", "arguments": {"image_index": 0}}]
</tool_log>
```

**fetch_webpage** - Read a webpage
```
<tool_log>
[{"tool_name": "fetch_webpage", "arguments": {"url": "https://example.com/article"}}]
</tool_log>
```

# **download_image** - Download an image from a URL (max 5 per task) [DISABLED]
# ```
# <tool_log>
# [{"tool_name": "download_image", "arguments": {"url": "https://example.com/image.jpg"}}]
# </tool_log>
# ```
# - Downloaded images are saved as downloaded_image_1.png, downloaded_image_2.png, etc.
# - You can then use image tools on them (e.g., crop downloaded_image_1)

## Multiple Tools in One Turn

You can call multiple tools at once:
```
<tool_log>
[
  {"tool_name": "crop", "arguments": {"image_index": 0, "bbox_2d": [100, 100, 500, 500], "label": "top left"}},
  {"tool_name": "crop", "arguments": {"image_index": 0, "bbox_2d": [500, 500, 900, 900], "label": "bottom right"}},
  {"tool_name": "google_search", "arguments": {"query": "search query"}}
]
</tool_log>
```

## Critical Rules (IMPORTANT!)

1. **Do NOT mix action and answer**:
   - If you use <tool_log>, do NOT include <answer> in the same response
   - Wait for results first, then answer
   - <answer> only appears when you're completely done

2. **After tools, you'll see**:
   - Image tools: "[Image 1: crop - description]" + the actual image
   - Search tools: Text results
   - Review these before continuing

3. **Image indexing is simple**:
   - Image 0 = original (always)
   - Image 1, 2, 3... = processed images (in order)
   - Use the right image_index for each tool

4. **Crop coordinates (0-1000 scale)**:
   - [0, 0, 1000, 1000] = entire image
   - [250, 250, 750, 750] = center 50%
   - [0, 0, 500, 500] = top-left quarter
   - [500, 500, 1000, 1000] = bottom-right quarter
   - **Aspect ratio limit**: Ensure (x2-x1)/(y2-y1) < 200 and (y2-y1)/(x2-x1) < 200
   - Avoid extremely narrow or tall crops (e.g., 800x4 or 4x800)

5. **ALWAYS use tools, NEVER guess**:
   - Use image tools to examine the image carefully
   - Use search tools to find facts
   - Do NOT make up information!

## Tips for Success

✅ **DO**:
- Think step by step in <think>
- Use simple, clear tool calls
- Wait for results before answering
- Use image_index correctly (0 for original, 1+ for processed)

❌ **DON'T**:
- Don't put <answer> with <tool_log> in same response
- Don't guess facts - search for them
- Don't use wrong image_index
- Don't forget to specify required arguments

## Example Workflow

```
Turn 1:
<think>I need to crop the text region and search for information.</think>
<tool_log>
[
  {"tool_name": "crop", "arguments": {"image_index": 0, "bbox_2d": [100, 200, 900, 400], "label": "text"}},
  {"tool_name": "google_search", "arguments": {"query": "relevant search"}}
]
</tool_log>

Turn 2 (after seeing results):
<think>The cropped image shows X, and search says Y. Now I can answer.</think>
<answer>
The answer is Z based on the cropped image and search results.
</answer>
```

## Important

- Keep it simple: one step at a time
- Use tools to gather information
- Think before acting
- Answer only when confident
- Always specify image_index in tool arguments
- Do NOT output <answer> tags until you have used tools and gathered information
- Do NOT guess - use tools to find the correct information!
'''



# ============================================================================
# Thyme Model Wrapper (OpenAI-compatible interface)
# ============================================================================

class ThymeModelWrapper:
    """Wrapper for Thyme model to provide OpenAI-compatible interface"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize Thyme model"""
        print(f"Loading Thyme model from {model_path}...")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model
        self.model = MODEL_CLASS.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.model.eval()
        self.device = device
        print(f"Thyme model loaded successfully! (using {MODEL_CLASS.__name__})")
    
    def chat_completions_create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ):
        """OpenAI-compatible chat completion interface"""
        
        # Convert OpenAI message format to Qwen format
        messages_for_processor = []
        all_images = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                messages_for_processor.append({
                    "role": "system",
                    "content": content if isinstance(content, str) else ""
                })
            
            elif role == "user":
                if isinstance(content, str):
                    messages_for_processor.append({
                        "role": "user",
                        "content": content
                    })
                elif isinstance(content, list):
                    user_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "image_url":
                                image_url = item.get("image_url", {}).get("url", "")
                                if image_url.startswith("data:image"):
                                    import base64
                                    import io
                                    header, data = image_url.split(",", 1)
                                    img_data = base64.b64decode(data)
                                    img = Image.open(io.BytesIO(img_data))
                                    all_images.append(img)
                                    user_content.append({"type": "image", "image": img})
                            elif item.get("type") == "text":
                                text_content = item.get('text', '')
                                if text_content:
                                    user_content.append({"type": "text", "text": text_content})
                    
                    if user_content:
                        messages_for_processor.append({
                            "role": "user",
                            "content": user_content
                        })
            
            elif role == "assistant":
                if isinstance(content, str) and content:
                    messages_for_processor.append({
                        "role": "assistant",
                        "content": content
                    })
        
        # Use apply_chat_template
        text = self.processor.apply_chat_template(
            messages_for_processor,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Prepare inputs
        if all_images:
            inputs = self.processor(
                text=[text],
                images=all_images,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        # Decode response
        generated_text = self.processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        # Create OpenAI-compatible response object
        class Message:
            def __init__(self, content):
                self.content = content
                self.tool_calls = None
        
        class Choice:
            def __init__(self, message):
                self.message = message
        
        class Usage:
            def __init__(self):
                self.prompt_tokens = 0
                self.completion_tokens = 0
                self.total_tokens = 0
        
        class Response:
            def __init__(self, content):
                self.choices = [Choice(Message(content))]
                self.usage = Usage()
        
        return Response(generated_text)


class ThymeClient:
    """OpenAI-compatible client wrapper for Thyme model"""
    
    def __init__(self, model_wrapper: ThymeModelWrapper):
        self.model_wrapper = model_wrapper
        self.chat = self
        self.completions = self
    
    def create(self, **kwargs):
        """OpenAI-compatible create method"""
        return self.model_wrapper.chat_completions_create(**kwargs)


def make_thyme_client(model_path: str, device: str = "cuda") -> ThymeClient:
    """Create a Thyme client with OpenAI-compatible interface"""
    wrapper = ThymeModelWrapper(model_path, device)
    return ThymeClient(wrapper)



# ============================================================================
# Tool Execution Functions
# ============================================================================

def dispatch_image_tool(state: AtomicState, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch image tool call to the appropriate function."""
    image_index = int(args.get("image_index", 0))
    label = str(args.get("label", ""))
    
    if name == "crop":
        bbox = args.get("bbox_2d", [0, 0, 1000, 1000])
        zoom_scale = float(args.get("zoom_scale", 1.0))
        return tool_crop(state, image_index, bbox, label, zoom_scale)
    if name == "rotate":
        return tool_rotate(state, image_index, float(args["angle"]), bool(args.get("expand", True)), label)
    if name == "flip":
        return tool_flip(state, image_index, str(args.get("direction", "horizontal")), label)
    if name == "resize":
        return tool_resize(state, image_index, args.get("width"), args.get("height"), args.get("scale"), label)
    if name == "enhance":
        return tool_enhance(state, image_index, args.get("brightness"), args.get("contrast"), args.get("sharpness"), label)
    if name == "grayscale":
        return tool_grayscale(state, image_index, label)
    if name == "autocontrast":
        return tool_autocontrast(state, image_index, float(args.get("cutoff", 0)), label)
    if name == "blur":
        return tool_blur(state, image_index, int(args.get("radius", 2)), label)
    if name == "sharpen":
        return tool_sharpen(state, image_index, label)
    if name == "denoise":
        return tool_denoise(state, image_index, int(args.get("strength", 10)), label)
    if name == "edge_detect":
        return tool_edge_detect(state, image_index, str(args.get("method", "canny")), label)
    if name == "invert":
        return tool_invert(state, image_index, label)
    if name == "equalize":
        return tool_equalize(state, image_index, label)
    if name == "threshold":
        return tool_threshold(state, image_index, int(args.get("value", 128)), str(args.get("mode", "binary")), label)
    raise ValueError(f"Unknown image tool: {name}")


def _resolve_image_for_lens(state: AtomicState, args: Dict[str, Any]) -> str:
    """Resolve image path for google_lens_search."""
    if "image_index" in args:
        idx = int(args["image_index"])
        return str(state.get_image(idx))
    
    # Legacy support
    image_ref = (args.get("image_ref") or "").strip().lower()
    if image_ref in {"current", "cur", "latest"}:
        return str(state.images[-1][0]) if state.images else str(state.orig_path)
    if image_ref in {"orig", "original", "input"}:
        return str(state.orig_path)
    
    return str(state.orig_path)


def parse_tool_log(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse <tool_log> XML block and return (tool_list, warnings)."""
    import re
    warnings = []
    
    # Extract tool_log block
    tool_log_re = re.compile(r"<tool_log>(.*?)</tool_log>", re.IGNORECASE | re.DOTALL)
    match = tool_log_re.search(text or "")
    
    if not match:
        return [], []
    
    raw = match.group(1).strip()
    if not raw:
        return [], []
    
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            # Flatten if model outputs [[...]]
            if len(obj) == 1 and isinstance(obj[0], list):
                obj = obj[0]
            
            tool_log = [x for x in obj if isinstance(x, dict)]
            if len(tool_log) != len(obj):
                non_dict_count = len(obj) - len(tool_log)
                warnings.append(f"Some tool_log items ({non_dict_count}) were not JSON objects")
            return tool_log, warnings
        else:
            warnings.append("tool_log must be a JSON array")
            return [], warnings
    except Exception as e:
        warnings.append(f"Failed to parse tool_log JSON: {e}")
        return [], warnings



# ============================================================================
# Main Rollout Function
# ============================================================================

def run_one(
    client: Any,
    task_json: Path,
    dataset_root: Path,
    images_dir: Optional[Path],
    out_dir: Path,
    model: str,
    temperature: float,
    max_rounds: int,
    max_tool_calls: int,
    enable_search: bool,
    search_cfg_path: Optional[Path],
    max_image_pixels: int = 2048 * 2048,
    image_quality: int = 95,
) -> Dict[str, Any]:
    """Run atomic tools mode for one task with multi-turn conversation."""
    
    task_cfg = read_json(task_json)
    img_path_result = resolve_image_path(task_json, task_cfg, dataset_root, images_dir)
    
    # Handle single or multiple images
    if isinstance(img_path_result, list):
        img_paths = img_path_result
    else:
        img_paths = [img_path_result]

    run_dir = ensure_dir(out_dir / task_json.stem)
    tool_images_dir = ensure_dir(run_dir / "tool_images")

    # Copy original image(s)
    orig_copies: List[Path] = []
    for i, img_path in enumerate(img_paths):
        if len(img_paths) == 1:
            orig_copy = run_dir / "orig.png"
        else:
            orig_copy = run_dir / f"orig_{i}.png"
        Image.open(img_path).save(orig_copy, "PNG")
        orig_copies.append(orig_copy)
    
    orig_copy = orig_copies[0]

    # Initialize atomic state
    state = AtomicState(orig_path=orig_copy, processed_dir=tool_images_dir)
    
    # Add additional original images
    for i, oc in enumerate(orig_copies[1:], start=1):
        state.images.append((oc, f"original input image {i+1}"))

    # Search tools
    search_tools: Optional[SearchTools] = None
    if enable_search:
        cfg = load_search_config(str(search_cfg_path) if search_cfg_path else None)
        if not cfg.cache_dir:
            cfg.cache_dir = str(ensure_dir(run_dir / "search_cache"))
        task_id = task_cfg.get("task_id", "") or run_dir.name
        search_tools = SearchTools(cfg, task_id=task_id)

    prompt = (task_cfg.get("input") or {}).get("prompt", "")

    # Build initial user message
    user_content: List[Dict[str, Any]] = []
    for i, oc in enumerate(orig_copies):
        user_content.append({"type": "image_url", "image_url": {"url": image_to_data_url(oc, max_image_pixels, image_quality, max_size_mb=8.0)}})
    
    if len(orig_copies) == 1:
        user_content.append({"type": "text", "text": f"[Image 0: original input]\n\n{prompt}"})
        history_text = f"[Image 0: original input - {img_paths[0].name}]\n\n{prompt}"
    else:
        img_labels = ", ".join([f"Image {i}" for i in range(len(orig_copies))])
        user_content.append({"type": "text", "text": f"[{img_labels}: original inputs]\n\n{prompt}"})
        history_text = f"[{img_labels}: original inputs - {', '.join([p.name for p in img_paths])}]\n\n{prompt}"

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": THYME_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    conversation_history: List[Dict[str, Any]] = [
        {"role": "system", "content": THYME_SYSTEM_PROMPT},
        {"role": "user", "content": history_text},
    ]

    tool_use_list: List[Dict[str, Any]] = []
    usage = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    tool_calls_total = 0
    download_count = 0  # Track downloaded images (max 5 per task)
    final_answer = ""
    all_warnings: List[str] = []

    for turn in range(max_rounds):
        # Call Thyme model
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=4000,
        )
        usage["api_calls"] += 1

        msg = resp.choices[0].message
        raw_content = msg.content or ""

        # Save raw output
        (run_dir / f"raw_model_output_turn_{turn}.txt").write_text(raw_content, encoding="utf-8")

        # Parse tool_log from XML
        tool_log, parse_warnings = parse_tool_log(raw_content)
        all_warnings.extend([f"[turn {turn}] {w}" for w in parse_warnings])

        # Force tool usage on first turn
        if turn == 0 and not tool_log:
            all_warnings.append(f"[turn {turn}] Model did not use tools in first turn, prompting to use tools...")
            messages.append({
                "role": "user",
                "content": "You MUST use tools to analyze the image and search for information. Do NOT provide an answer without using tools first. Please output a <tool_log> block with your tool calls now."
            })
            conversation_history.append({
                "role": "user",
                "content": "[System reminder: Please use tools first - output <tool_log> block]"
            })
            continue  # Retry this turn

        # Process tool calls
        if tool_log:
            messages.append({"role": "assistant", "content": raw_content})
            if raw_content:
                conversation_history.append({"role": "assistant", "content": raw_content})

            for tool_entry in tool_log:
                tool_calls_total += 1
                if tool_calls_total > max_tool_calls:
                    all_warnings.append(f"Exceeded max_tool_calls={max_tool_calls}")
                    break

                name = tool_entry.get("tool_name", "")
                args = tool_entry.get("arguments", {})

                # Track tool call
                conversation_history.append({
                    "role": "tool_call",
                    "content": json.dumps({"name": name, "arguments": args}, ensure_ascii=False),
                })

                # Execute tool
                if name in IMAGE_TOOLS:
                    # Image tool
                    try:
                        out = dispatch_image_tool(state, name, args)
                        new_index = out.get("new_image_index", len(state.images))
                        out_path = Path(out["output_path"])
                        
                        tool_use_list.append({
                            "index": len(tool_use_list),
                            "tool_name": name,
                            "raw_tool_name": name,
                            "arguments": args,
                            "output": out,
                            "turn": turn,
                            "timestamp": utc_ts(),
                        })

                        # Send new image to model
                        label = out.get("label", "")
                        img_info = f"[Image {new_index}: {name}"
                        if label:
                            img_info += f" - {label}"
                        img_info += "]"
                        
                        # Adaptive compression
                        img_count = sum(1 for m in messages if isinstance(m.get("content"), list) 
                                       and any(c.get("type") == "image_url" for c in m["content"] if isinstance(c, dict)))
                        adaptive_params = get_adaptive_image_params(img_count)
                        
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_to_data_url(
                                    out_path, 
                                    max_pixels=adaptive_params["max_pixels"],
                                    quality=adaptive_params["quality"],
                                    max_size_mb=adaptive_params["max_size_mb"]
                                )}},
                                {"type": "text", "text": img_info},
                            ],
                        })

                        conversation_history.append({
                            "role": "tool_response",
                            "content": json.dumps(out, ensure_ascii=False),
                        })
                        conversation_history.append({
                            "role": "user",
                            "content": f"{img_info} - {out_path.name}",
                        })
                        
                    except Exception as e:
                        error_out = {"ok": "false", "error": str(e)}
                        messages.append({
                            "role": "user",
                            "content": json.dumps(error_out, ensure_ascii=False),
                        })
                        conversation_history.append({
                            "role": "tool_response",
                            "content": json.dumps(error_out, ensure_ascii=False),
                        })
                        all_warnings.append(f"Tool {name} error: {e}")

                else:
                    # Search tool
                    try:
                        if not enable_search or search_tools is None:
                            out = {"ok": False, "error": "Search is not enabled"}
                        elif name == "google_search":
                            out = search_tools.google_search(
                                query=str(args.get("query", "")),
                                gl=args.get("gl"),
                                hl=args.get("hl"),
                            )
                        elif name == "google_lens_search":
                            img_path_for_lens = _resolve_image_for_lens(state, args)
                            out = search_tools.google_lens_search(image_path=img_path_for_lens)
                        elif name == "fetch_webpage":
                            out = search_tools.fetch_webpage(
                                url=str(args.get("url", "")),
                                max_chars=int(args.get("max_chars", 12000) or 12000),
                            )
                        # elif name == "download_image":
                        #     # Check download limit
                        #     if download_count >= MAX_DOWNLOAD_IMAGES:
                        #         out = {"ok": False, "error": f"Download limit reached. Maximum {MAX_DOWNLOAD_IMAGES} images per task."}
                        #     else:
                        #         url = str(args.get("url", ""))
                        #         if not url:
                        #             out = {"ok": False, "error": "download_image requires 'url' argument"}
                        #         else:
                        #             from search_tools import download_image_from_url
                        #             out = download_image_from_url(
                        #                 url=url,
                        #                 save_dir=str(tool_images_dir),
                        #                 timeout_s=30,
                        #             )
                        #             if isinstance(out, dict) and out.get("ok"):
                        #                 download_count += 1
                        #                 # Add downloaded image to state
                        #                 if "path" in out:
                        #                     downloaded_path = Path(out["path"])
                        #                     if downloaded_path.exists():
                        #                         new_idx = len(state.images)
                        #                         state.images.append((downloaded_path, f"downloaded from {url[:50]}"))
                        else:
                            out = {"ok": False, "error": f"Unknown tool: {name}"}
                    except Exception as search_err:
                        error_msg = str(search_err)
                        if "504" in error_msg or "timeout" in error_msg.lower():
                            out = {"ok": False, "error": "Search service timeout"}
                        elif "SSL" in error_msg or "ssl" in error_msg.lower():
                            out = {"ok": False, "error": "Network error accessing URL"}
                        else:
                            out = {"ok": False, "error": f"Search failed: {error_msg[:200]}"}
                        all_warnings.append(f"Search tool {name} error: {error_msg}")

                    tool_use_list.append({
                        "index": len(tool_use_list),
                        "tool_name": name,
                        "raw_tool_name": name,
                        "arguments": args,
                        "output": out,
                        "turn": turn,
                        "timestamp": utc_ts(),
                    })

                    # Extract content for model
                    if isinstance(out, dict) and "context" in out:
                        model_content = out["context"]
                    elif isinstance(out, dict) and "text" in out:
                        model_content = out["text"]
                    elif isinstance(out, dict) and "error" in out:
                        model_content = f"Error: {out['error']}"
                    else:
                        model_content = json.dumps(out, ensure_ascii=False)

                    messages.append({
                        "role": "user",
                        "content": model_content,
                    })

                    conversation_history.append({
                        "role": "tool_response",
                        "content": model_content,
                    })

            continue  # Next turn

        # No tool calls - final answer
        final_answer = raw_content.strip()
        messages.append({"role": "assistant", "content": raw_content})
        conversation_history.append({"role": "assistant", "content": raw_content})
        break

    # Request final answer if exhausted rounds
    if not final_answer and turn == max_rounds - 1:
        final_prompt = "Please provide your final answer now based on all the information gathered."
        messages.append({"role": "user", "content": final_prompt})
        conversation_history.append({"role": "user", "content": final_prompt})

        try:
            final_resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=2000,
            )
            usage["api_calls"] += 1

            final_raw = final_resp.choices[0].message.content or ""
            (run_dir / "raw_model_output_final.txt").write_text(final_raw, encoding="utf-8")
            conversation_history.append({"role": "assistant", "content": final_raw})
            final_answer = final_raw.strip()
        except Exception as e:
            all_warnings.append(f"Failed to get final answer: {e}")

    # Save outputs
    (run_dir / "model_answer.txt").write_text(final_answer, encoding="utf-8")
    write_json(run_dir / "tool_use_list.json", tool_use_list)

    # Save image index tracking
    image_list = state.get_image_list()
    write_json(run_dir / "image_index.json", image_list)

    # Save conversation history
    # Build tools schema for conversation record (same format as OpenAI)
    tools_schema = build_atomic_tools_schema()
    tools_json = json.dumps(tools_schema, ensure_ascii=False)
    
    conversation_record = {
        "task_id": task_cfg.get("task_id", ""),
        "tools": tools_json,
        "images": image_list,
        "messages": conversation_history,
        "final_answer": final_answer,
    }
    with open(run_dir / "conversation.json", "w", encoding="utf-8") as f:
        json.dump(conversation_record, f, ensure_ascii=False, indent=2)

    # Build tool statistics
    image_tool_hist: Dict[str, int] = {}
    search_tool_hist: Dict[str, int] = {}
    for ev in tool_use_list:
        tname = ev.get("tool_name", "")
        if tname in IMAGE_TOOLS:
            image_tool_hist[tname] = image_tool_hist.get(tname, 0) + 1
        elif tname in ("google_search", "google_lens_search", "fetch_webpage"):  # , "download_image"):
            search_tool_hist[tname] = search_tool_hist.get(tname, 0) + 1

    run_meta = {
        "task_id": task_cfg.get("task_id", ""),
        "task_file": str(task_json.resolve()),
        "mode": "atomic",
        "driver": "thyme",
        "model": model,
        "temperature": temperature,
        "usage": usage,
        "effective_tool_calls": len(tool_use_list),
        "total_images": len(state.images),
        "paths": {
            "run_dir": str(run_dir),
            "orig": str(orig_copy),
            "processed_dir": str(tool_images_dir),
        },
        "image_tool_analysis": {
            "image_tool_calls": sum(image_tool_hist.values()),
            "image_tool_hist": image_tool_hist,
        },
        "search_analysis": {
            "search_enabled": bool(enable_search),
            "search_tool_calls": sum(search_tool_hist.values()),
            "search_tool_hist": search_tool_hist,
        },
        "warnings": all_warnings,
    }
    write_json(run_dir / "run_meta.json", run_meta)
    return run_meta



# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_json", type=str, default="")
    ap.add_argument("--task_dir", type=str, default="")
    ap.add_argument("--dataset_root", type=str, default="")
    ap.add_argument("--images_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory. Default: runs/atomic/thyme")
    ap.add_argument("--model_path", type=str, required=True, help="Path to Thyme model")
    ap.add_argument("--temperature", type=float, default=0.0)
    
    # Multi-turn settings
    ap.add_argument("--max_rounds", type=int, default=15, help="Max model turns")
    ap.add_argument("--max_tool_calls", type=int, default=15, help="Max total tool calls")
    
    # Rate limiting
    ap.add_argument("--task_delay", type=float, default=0.0, help="Delay between tasks (seconds)")

    # Search options
    ap.add_argument("--enable_search", action="store_true", help="Enable web search tools")
    ap.add_argument("--search_config", type=str, default="", help="Path to search config JSON")

    # Image settings
    ap.add_argument("--max_image_pixels", type=int, default=2048*2048)
    ap.add_argument("--image_quality", type=int, default=95)
    
    # Skip and shard options
    ap.add_argument("--skip_existing", action="store_true", help="Skip tasks that already have run_meta.json")
    ap.add_argument("--shard", type=int, default=0, help="Shard index (0-based)")
    ap.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    ap.add_argument("--max_tasks", type=int, default=0, help="Max tasks to process (0 = unlimited)")

    args = ap.parse_args()

    # Create Thyme client
    client = make_thyme_client(args.model_path)

    tasks: List[Path] = []
    if args.task_json:
        tasks = [Path(args.task_json)]
    elif args.task_dir:
        tasks = sorted(Path(args.task_dir).glob("*.json"))
    else:
        raise ValueError("Provide --task_json or --task_dir")

    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    images_dir = Path(args.images_dir) if args.images_dir else None
    
    # Auto-generate out_dir
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("runs/atomic/thyme")
    
    search_cfg_path = Path(args.search_config) if args.search_config else None
    
    # Apply sharding
    if args.num_shards > 1:
        total = len(tasks)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start_idx = args.shard * shard_size
        end_idx = min(start_idx + shard_size, total)
        tasks = tasks[start_idx:end_idx]
        print(f"[Shard {args.shard}/{args.num_shards}] Tasks {start_idx}-{end_idx-1} ({len(tasks)} tasks)")

    all_meta = []
    skipped = 0
    processed = 0
    
    for idx, t in enumerate(tasks):
        # Check max_tasks limit
        if args.max_tasks > 0 and processed >= args.max_tasks:
            print(f"[LIMIT] Reached max_tasks={args.max_tasks}, stopping")
            break
        
        # Skip if already completed
        if args.skip_existing:
            run_dir = out_dir / t.stem
            has_meta = (run_dir / "run_meta.json").exists()
            has_answer = (run_dir / "model_answer.txt").exists()
            if has_meta and has_answer:
                skipped += 1
                print(f"[SKIP] {t.name} (already completed)")
                continue
            elif has_meta or has_answer or run_dir.exists():
                print(f"[RETRY] {t.name} (incomplete run, retrying)")
        
        try:
            ds = resolve_dataset_root(t, dataset_root)
            
            m = run_one(
                client=client,
                task_json=t,
                dataset_root=ds,
                images_dir=images_dir,
                out_dir=out_dir,
                model=args.model_path,
                temperature=args.temperature,
                max_rounds=args.max_rounds,
                max_tool_calls=args.max_tool_calls,
                enable_search=bool(args.enable_search),
                search_cfg_path=search_cfg_path,
                max_image_pixels=args.max_image_pixels,
                image_quality=args.image_quality,
            )
            all_meta.append(m)
            processed += 1
            print(f"[OK] {t.name} -> {out_dir / t.stem}")
            
            # Delay between tasks
            if idx < len(tasks) - 1 and args.task_delay > 0:
                time.sleep(args.task_delay)
                
        except Exception as e:
            print(f"[ERR] {t}: {e}")
            traceback.print_exc()

    ensure_dir(out_dir)
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Completed: {len(all_meta)}")
    print(f"Skipped: {skipped}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
