#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepEyes-7B Model Runner for General Mode (Multi-turn with Code Execution)
Adapted from run_general_script_deepeyes.py to use DeepEyes-7B model
DeepEyes is based on Qwen-2.5-VL-7B-Instruct with RL training
"""

from __future__ import annotations
import os
import json
import argparse
import re
import subprocess
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

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*AutoModelForVision2Seq.*deprecated.*")
warnings.filterwarnings("ignore", message=".*fast processor.*")
warnings.filterwarnings("ignore", message=".*model of type.*")

# Try to import the correct model class (DeepEyes uses Qwen2.5-VL architecture)
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

from common_utils import ensure_dir, image_to_data_url, read_json, safe_name, utc_ts, write_json, get_adaptive_image_params
from dataset_utils import resolve_dataset_root, resolve_image_path
from search_tools import SearchTools, load_search_config
from ast_ops import infer_ops_and_saves, infer_tool_events

# Import all parsing and utility functions from OpenAI script (same folder)
from .run_general_script_openai import (
    parse_model_output,
    list_transformed_images,
    list_all_output_images,
    exec_python_file,
    build_tool_use_list_from_code_and_outputs,
    write_replay_script,
    _normalize_tool_request,
    _expand_placeholders,
    _execute_tool,
    MAX_DOWNLOAD_IMAGES,
    SEARCH_TOOLS_SCHEMA,  # Use the SAME tool schema
)

# =============================================================================
# System prompt for DeepEyes (simplified for 7B model, adapted for XML tool_log)
# =============================================================================
DEEPEYES_SYSTEM_PROMPT = r'''You are a multimodal reasoning agent. You solve tasks step-by-step using a "Think, Act, Observe" loop.

# IMPORTANT: SERVER ENVIRONMENT NOTICE
You are running in a server environment without a display. This means:
- **DO NOT use `plt.show()`, `cv2.imshow()`, or any display functions** - they will cause errors.
- **DO NOT use `pytesseract` library or `!pip install` commands** - they are not available and will fail.
- **ALWAYS save processed images to files** - this is the ONLY way to preserve them for future turns.
- **Use the file naming protocol below** - saved images will automatically be provided to you in the next turn.
- If you process an image but don't save it, it will be LOST and you won't be able to use it later.

# CRITICAL: CORRECT IMAGE LOADING
**CRITICAL:** `image_0` is NOT a variable! You MUST load it first!
- **WRONG:** `cv2.rotate(image_0, ...)` or `np.array(image_0)` → NameError!
- **CORRECT:** First load: `img = cv2.imread("image_0.png")`, then use: `cv2.rotate(img, ...)`

**IMPORTANT:** Every `<code>` block MUST load the image first. You CANNOT directly use `image_0.crop(...)` - you must first do `img = Image.open("image_0.png")` then `img.crop(...)`.

You MUST use standard image loading functions with the correct filename STRING.

**CORRECT (Use these):**
```python
# Using PIL (recommended for most operations)
from PIL import Image
img = Image.open("image_0.png")  # ✓ Use the STRING "image_0.png"

**WRONG (Never do this):**
```python
import numpy as np
img_array = np.array(image_0)  # ❌ WRONG: image_0 is NOT a variable!
# This will cause: NameError: name 'image_0' is not defined
```

**Why this is wrong:**
- `image_0` (without quotes) is treated as a variable name, which doesn't exist
- You MUST use the string `"image_0.png"` (with quotes) as the filename
- Always load images with `Image.open("image_0.png")` or `cv2.imread("image_0.png")`

# 1. THE "MUTUAL EXCLUSION" RULE (CRITICAL)
In any single turn, you can do **EXACTLY ONE** of the following:
- **OPTION A: ACT**. Generate a tool call (`<code>` or `<tool_log>`).
  - **FORBIDDEN**: Do NOT include `<answer>` in this turn.
  - **FORBIDDEN**: Do NOT speculate or guess the final answer in your `<think>` trace.
- **OPTION B: ANSWER**. Generate the final `<answer>`.
  -  **FORBIDDEN**: Do NOT use tools in this turn.

# 2. THINKING SCOPE
- When **Acting**: Your `<think>` should ONLY explain *why* you need the tool and *what* you expect to see. **Do NOT deduce the final answer yet.**
- When **Answering**: Your `<think>` should synthesize the *observed* evidence from previous steps.

# 1. AVAILABLE TOOLS

## Search Tools (Use via <tool_log> You must strictly follow these schemas. All search tools MUST be wrapped in `<tool_log>` tags)
- **google_search**: Check facts, dates, events, news.
  Example: `[{"tool_name": "google_search", "arguments": {"query": "US President in 1983"}}]`
- **google_lens_search**: Identify objects/text in an image.
  Example: `[{"tool_name": "google_lens_search", "arguments": {"image_path": "image_1.png"}}]`
- **fetch_webpage**: Read full content of a URL.
  Example:  [{"tool_name": "fetch_webpage", "arguments": {"url": "https://example.com/article"}}]

## Code Execution (Use via <code>)
- **Python**: Use `PIL` to crop, resize, or enhance images. Avoid using `cv2` package when possible.
- **Restriction**: Do NOT use code to calculate facts (e.g. dates).

# 2. FILE NAMING PROTOCOL (CRITICAL)
The system handles complex paths for you. You MUST follow these simple rules:
- **Input**: The original image is ALWAYS named `"image_0.png"`.
- **Output**: Save your processed image as `"image_1.png"` (for step 1), `"image_2.png"` (for step 2), etc.
- **Reference**: To use a processed image in Search tools, just use its name (e.g., `"image_1.png"`).
- **NO RANDOM**: Do NOT use `random` or dynamic filenames.
- **MUST SAVE**: Every processed image MUST be saved with `.save()` or `cv2.imwrite()`. Unsaved images are LOST forever.
- **NO DISPLAY**: Never use `plt.show()`, `cv2.imshow()`, or `Image.show()` - they don't work in server environments.

# 3. REACT WORKFLOW
1. **Think**: Analyze the current state. What do I know? What do I need?
2. **Act**: Choose ONE tool (Code OR Search) to get missing information.
3. **Stop**: Output the tool block and STOP. Wait for the system to give you the result.

# 4. EXAMPLES

## Example 1: Vision Processing (PIL)
<think>The text is too small. I need to crop the logo to read it.</think>
<code>
from PIL import Image

# ✓ CORRECT: Load with the STRING "image_0.png"
img = Image.open("image_0.png") 

# Process
cropped = img.crop((100, 100, 500, 500))

# CRITICAL: Save the processed image - this is the ONLY way to preserve it
cropped.save("image_1.png")
print("Saved image_1.png")
</code>

## Example 2: Fact Checking
<think>I see the logo says "Verizon". I need to find when it was founded.</think>
<tool_log>
[{"tool_name": "google_search", "arguments": {"query": "Verizon founding date"}}]
</tool_log>

**WRONG (Mixed):**
<think>I need to search. I think the answer is probably 2025.</think> ❌ (Guessing)
<tool_log>...</tool_log>
<answer>2025</answer> ❌ (Answering in same turn)

**WRONG (Not Saving):**
<code>
img = Image.open("image_0.png")
cropped = img.crop((100, 100, 500, 500))
# ❌ FORGOT TO SAVE - image will be lost!
</code>

**WRONG (Using Display):**
<code>
import matplotlib.pyplot as plt
img = Image.open("image_0.png")
plt.imshow(img)
plt.show()  # ❌ Will cause error in server environment
</code>

**WRONG (Incorrect Image Loading):**
<code>
import numpy as np
import cv2

# ❌ WRONG: image_0 without quotes is treated as a variable name
img_array = np.array(image_0)  # NameError: name 'image_0' is not defined

# Define crop region
left = 1600
top = 200
right = 1800
bottom = 300
cropped = img_array[top:bottom, left:right]
cv2.imwrite("image_1.png", cropped)
</code>

**CORRECT (Proper Image Loading - PIL):**
<code>
from PIL import Image

# ✓ CORRECT: Use the STRING "image_0.png" with quotes
img = Image.open("image_0.png")

# Define the crop box
left = 750
top = 50
right = 900
bottom = 150

# Crop the image
cropped_img = img.crop((left, top, right, bottom))

# Save the cropped image
cropped_img.save("image_1.png")
print("Cropped image_1.png")
</code>

**CORRECT (Proper Image Loading - OpenCV):**
<code>
import cv2

# ✓ CORRECT: Use the STRING "image_0.png" with quotes
img_array = cv2.imread("image_0.png")

# Define crop region
left = 1600
top = 200
right = 1800
bottom = 300

# Crop using array slicing
cropped = img_array[top:bottom, left:right]

# Save the cropped image
cv2.imwrite("image_1.png", cropped)
print("Cropped image_1.png")
</code>

'''


# ============================================================================
# DeepEyes Model Wrapper (OpenAI-compatible interface)
# ============================================================================

class DeepEyesModelWrapper:
    """Wrapper for DeepEyes model to provide OpenAI-compatible interface"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize DeepEyes model"""
        print(f"Loading DeepEyes model from {model_path}...")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model using the detected model class
        self.model = MODEL_CLASS.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.model.eval()
        self.device = device
        print(f"DeepEyes model loaded successfully! (using {MODEL_CLASS.__name__})")

    
    def chat_completions_create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_completion_tokens: int = 12000,
        **kwargs
    ):
        """OpenAI-compatible chat completion interface"""
        
        # Convert OpenAI message format to Qwen2.5-VL format
        messages_for_processor = []
        all_images = []
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                # System message - pass directly
                messages_for_processor.append({
                    "role": "system",
                    "content": content if isinstance(content, str) else ""
                })
            
            elif role == "user":
                if isinstance(content, str):
                    # Simple text message
                    messages_for_processor.append({
                        "role": "user",
                        "content": content
                    })
                elif isinstance(content, list):
                    # Multimodal content (text + images)
                    user_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "image_url":
                                # Extract base64 image data
                                image_url = item.get("image_url", {}).get("url", "")
                                if image_url.startswith("data:image"):
                                    # Decode base64 image
                                    import base64
                                    import io
                                    header, data = image_url.split(",", 1)
                                    img_data = base64.b64decode(data)
                                    img = Image.open(io.BytesIO(img_data))
                                    all_images.append(img)
                                    # Add image to content
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
                # Assistant message
                if isinstance(content, str) and content:
                    messages_for_processor.append({
                        "role": "assistant",
                        "content": content
                    })
        
        # Use apply_chat_template to format properly with system prompt
        text = self.processor.apply_chat_template(
            messages_for_processor,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Debug: Print the formatted text to verify system prompt is included
        print(f"\n=== DEBUG: Formatted prompt (first 500 chars) ===")
        print(text[:500])
        print(f"=== Total length: {len(text)} chars ===\n")
        
        # Prepare inputs
        if all_images:
            # Multi-modal input
            inputs = self.processor(
                text=[text],
                images=all_images,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
        else:
            # Text-only input
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True
            ).to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_completion_tokens,
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
                self.tool_calls = None  # DeepEyes doesn't use native function calling
        
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



class DeepEyesClient:
    """OpenAI-compatible client wrapper for DeepEyes model"""
    
    def __init__(self, model_wrapper: DeepEyesModelWrapper):
        self.model_wrapper = model_wrapper
        self.chat = self
        self.completions = self
    
    def create(self, **kwargs):
        """OpenAI-compatible create method"""
        return self.model_wrapper.chat_completions_create(**kwargs)


def make_deepeyes_client(model_path: str, device: str = "cuda") -> DeepEyesClient:
    """Create a DeepEyes client with OpenAI-compatible interface"""
    wrapper = DeepEyesModelWrapper(model_path, device)
    return DeepEyesClient(wrapper)

def _clean_deepeyes_code(code: str, tool_images_dir: Path, turn: int = 0) -> str:
    """
    Clean DeepEyes code with Robust Runtime Hijacking.
    Fixes:
    1. SyntaxError caused by regex replacement of strings.
    2. NameError in header injection.
    3. FileNotFoundError by hijacking Open/Save at runtime.
    """
    
    # -------------------------------------------------------------------------
    # 1. 静态文本清洗 (Global String Sanitization)
    # -------------------------------------------------------------------------
    # 移除常见的幻觉路径前缀，变成相对路径或空字符串，防止 os.path 报错
    bad_prefixes = [
        "/mnt/data/temp_processed_images/",
        "/mnt/data/", 
        "/d/data/temp_processed_images/",
        "/d/data/",
        "/tmp/"
    ]
    for bad in bad_prefixes:
        code = code.replace(bad, "")

    # 2. 基础格式清理
    code = re.sub(r'^```python\s*\n', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n```\s*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'^```\s*\n', '', code, flags=re.MULTILINE)

    # 3. JSON 防呆 (注释掉非代码行)
    if 'tool_name' in code:
        lines = code.split('\n')
        code = '\n'.join([f"# {l}" if ('tool_name' in l or 'google_search' in l or l.strip().startswith('[{')) else l for l in lines])

    # -------------------------------------------------------------------------
    # [核心逻辑] 运行时劫持注入 (The Nuclear Fix)
    # -------------------------------------------------------------------------
    target_filename = f"image_{turn + 1}.png"
    
    header = f'''
import os
import sys
from PIL import Image

try:
    # 1. 定义环境
    # [FIX] 增加默认值，防止环境变量缺失导致 Crash
    _real_save_dir = os.environ.get('PROCESSED_IMAGE_SAVE_PATH', '.')
    
    # [FIX] 修复 NameError: 必须用字符串 'ORIGINAL_IMAGE_PATH'
    _real_input_0 = os.environ.get('ORIGINAL_IMAGE_PATH', 'image_0.png')
    
    # 本轮强制输出路径
    _force_save_path = os.path.join(_real_save_dir, '{target_filename}')

    # ---------------------------------------------------------
    # 2. 劫持 Image.open (解决读取报错)
    # ---------------------------------------------------------
    _orig_open = Image.open
    
    def _safe_open(fp, mode="r"):
        fp_str = str(fp)
        # 规则 A: 只要包含 image_0，强制读原图
        # 无论它是 "image_0.png" 还是 "/mnt/.../image_0.png"
        if "image_0" in fp_str:
            return _orig_open(_real_input_0, mode)
            
        # 规则 B: 只要包含 image_X (且不是0)，强制去 save_dir 找
        if "image_" in fp_str:
            clean_name = os.path.basename(fp_str)
            return _orig_open(os.path.join(_real_save_dir, clean_name), mode)
             
        # 兜底
        return _orig_open(fp, mode)
        
    Image.open = _safe_open

    # ---------------------------------------------------------
    # 3. 劫持 Image.Image.save (解决写入报错)
    # ---------------------------------------------------------
    _orig_save = Image.Image.save
    
    def _safe_save(self, fp, format=None, **params):
        print(f"[System Hijack] Redirecting PIL save to: {{_force_save_path}}")
        try:
            # 强制写入正确路径，忽略模型传入的 fp
            _orig_save(self, _force_save_path, format, **params)
        except Exception as e:
            print(f"[System Hijack Error] {{e}}")
            
    Image.Image.save = _safe_save

    # ---------------------------------------------------------
    # 4. 劫持 cv2.imread (解决读取报错) - COMMENTED OUT
    # ---------------------------------------------------------
    # try:
    #     import cv2
    #     _orig_imread = cv2.imread
    #     
    #     def _safe_imread(filename, flags=None):
    #         fp_str = str(filename)
    #         # 规则 A: 只要包含 image_0，强制读原图
    #         if "image_0" in fp_str:
    #             if flags is not None:
    #                 return _orig_imread(_real_input_0, flags)
    #             else:
    #                 return _orig_imread(_real_input_0)
    #                 
    #         # 规则 B: 只要包含 image_X (且不是0)，强制去 save_dir 找
    #         if "image_" in fp_str:
    #             clean_name = os.path.basename(fp_str)
    #             real_path = os.path.join(_real_save_dir, clean_name)
    #             if flags is not None:
    #                 return _orig_imread(real_path, flags)
    #             else:
    #                 return _orig_imread(real_path)
    #              
    #         # 兜底
    #         if flags is not None:
    #             return _orig_imread(filename, flags)
    #         else:
    #             return _orig_imread(filename)
    #         
    #     cv2.imread = _safe_imread
    # except ImportError:
    #     pass  # cv2 not imported, skip hijacking

    # ---------------------------------------------------------
    # 5. 劫持 cv2.imwrite (解决写入报错)
    # ---------------------------------------------------------
    try:
        import cv2
        _orig_imwrite = cv2.imwrite
        
        def _safe_imwrite(filename, img, params=None):
            print(f"[System Hijack] Redirecting cv2.imwrite to: {{_force_save_path}}")
            try:
                # 强制写入正确路径，忽略模型传入的 filename
                if params is not None:
                    return _orig_imwrite(_force_save_path, img, params)
                else:
                    return _orig_imwrite(_force_save_path, img)
            except Exception as e:
                print(f"[System Hijack Error] {{e}}")
                return False
                
        cv2.imwrite = _safe_imwrite
    except ImportError:
        pass  # cv2 not imported, skip hijacking

except Exception as e:
    print(f"Setup error: {{e}}")
'''
    # 注入到代码最前面
    if 'import ' in code:
        code = header + code
    else:
        code = header + code

    # -------------------------------------------------------------------------
    # [正则辅助]
    # -------------------------------------------------------------------------
    
    # [变更] 移除了之前替换 "image_0.png" -> "_real_input_0" 的正则。
    # 原因：这会导致 SyntaxError (如 unmatched ')')。
    # 既然我们在 header 里劫持了 Image.open，代码里保留字符串 "image_0.png" 是完全安全的。
    
    # 仅仅为了保险，如果代码里有显式的 input_path = ... 赋值，我们可以尝试注释掉
    # 但不做激进替换，避免破坏语法结构。

    return code.strip()


# ============================================================================
# Main Rollout Function (IDENTICAL logic to OpenAI version)
# ============================================================================

def run_one_rollout(
    client: Any,
    task_json: Path,
    dataset_root: Path,
    images_dir: Optional[Path],
    out_dir: Path,
    model: str,
    temperature: float,
    python_exe: str,
    enable_search: bool = False,
    search_cfg_path: Optional[Path] = None,
    max_rounds: int = 4,
    max_tool_calls: int = 12,
    max_image_pixels: int = 2048 * 2048,
    image_quality: int = 95,
    use_native_tools: bool = False,  # DeepEyes doesn't support native tools, always False
) -> Dict[str, Any]:
    """Multi-turn runner that executes tools and feeds results back to the model.
    
    This function has IDENTICAL logic to run_general_script_openai.py
    The only difference is that DeepEyes doesn't support native function calling,
    so it always uses XML-based tool_log parsing.
    """

    task_cfg = read_json(task_json)
    img_path_result = resolve_image_path(task_json, task_cfg, dataset_root, images_dir)
    
    # Handle single or multiple images
    if isinstance(img_path_result, list):
        img_paths = img_path_result
    else:
        img_paths = [img_path_result]

    item_key = safe_name(task_json.stem)
    run_dir = ensure_dir(out_dir / item_key)
    tool_images_dir = ensure_dir(run_dir / "tool_images")

    # Copy original image(s) and save as image_0.png (or image_0.png, image_1.png for multiple)
    orig_copies: List[Path] = []
    for i, img_path in enumerate(img_paths):
        # Use image_N.png naming to match the index system
        orig_copy = run_dir / f"image_{i}.png"
        Image.open(img_path).save(orig_copy, "PNG")
        orig_copies.append(orig_copy)
    
    # Primary original image (for backward compatibility)
    orig_copy = orig_copies[0]

    prompt = (task_cfg.get("input") or {}).get("prompt", "")

    # Search tools - SAME as OpenAI
    search_tools: Optional[SearchTools] = None
    if enable_search:
        cfg = load_search_config(str(search_cfg_path) if search_cfg_path else None)
        if not getattr(cfg, "cache_dir", None):
            cfg.cache_dir = str(ensure_dir(run_dir / "search_cache"))
        task_id = task_cfg.get("task_id", "") or run_dir.name
        search_tools = SearchTools(cfg, task_id=task_id)

    # Runtime env map - SAME as OpenAI
    import os as _os
    base_env = dict(_os.environ)
    base_env["PROCESSED_IMAGE_SAVE_PATH"] = str(tool_images_dir.resolve())

    current_img_path = orig_copy.resolve()
    
    # Image index tracking - SAME as OpenAI
    image_list: List[Dict[str, Any]] = []
    download_count = 0
    for i, oc in enumerate(orig_copies):
        if len(orig_copies) == 1:
            image_list.append({"index": 0, "path": str(oc), "label": "original input image"})
        else:
            image_list.append({"index": i, "path": str(oc), "label": f"original input image {i+1}"})

    # Build tools list - SAME as OpenAI
    tools_list = []
    if enable_search:
        tools_list.extend(SEARCH_TOOLS_SCHEMA)

    # Use DeepEyes-specific system prompt (with XML tool_log instructions)
    system_prompt = DEEPEYES_SYSTEM_PROMPT

    # Build initial user message - SAME as OpenAI
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
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    
    # Track conversation history - SAME as OpenAI
    conversation_history: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": history_text},
    ]

    usage = {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    all_warnings: List[str] = []
    tool_call_events: List[Dict[str, Any]] = []
    exec_events: List[Dict[str, Any]] = []
    code_turn_paths: List[Path] = []

    final_answer = ""
    tool_calls_total = 0
    last_turn_had_code = False

    for turn in range(max_rounds):
        # Call model - DeepEyes doesn't support tools parameter
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=12000,
        )
        usage["api_calls"] += 1

        message = resp.choices[0].message
        raw = message.content or ""
        
        # Save raw model output - SAME as OpenAI
        (run_dir / f"raw_model_output_turn_{turn}.txt").write_text(raw, encoding="utf-8")
        
        last_turn_had_code = False
        
        # Parse model output - SAME as OpenAI
        thinking, code, ans, tool_log, parse_warnings = parse_model_output(raw)
        all_warnings.extend([f"[turn {turn}] {w}" for w in parse_warnings])
        
        # PRIORITY 1: Handle tool_log FIRST (before code execution)
        # This ensures search tools are executed and recorded even if model also outputs code
        normalized: List[Tuple[str, Dict[str, Any]]] = []
        for entry in tool_log:
            if not isinstance(entry, dict):
                all_warnings.append(f"[turn {turn}] tool_log entry not an object: {type(entry)}")
                continue
            tname, targs = _normalize_tool_request(entry)
            if not tname:
                all_warnings.append(f"[turn {turn}] tool_log entry missing tool_name")
                continue
            normalized.append((tname, targs))

        if normalized:
            if not enable_search or search_tools is None:
                final_answer = ""
                all_warnings.append("[runtime] enable_search is false but tool_log requested tools.")
                break

            env_map = {
                "LOCAL_INPUT_IMAGE_PATH": str(current_img_path),
                "PROCESSED_IMAGE_SAVE_PATH": str(tool_images_dir.resolve()),
            }

            results_for_model: List[Dict[str, Any]] = []
            for tname, targs in normalized:
                tool_calls_total += 1
                if tool_calls_total > max_tool_calls:
                    all_warnings.append(f"[runtime] Exceeded max_tool_calls={max_tool_calls}")
                    break

                targs = _expand_placeholders(targs, env_map)

                # Check download limit (DISABLED)
                # if tname == "download_image" and download_count >= MAX_DOWNLOAD_IMAGES:
                #     out = {"ok": False, "error": f"Download limit reached. Maximum {MAX_DOWNLOAD_IMAGES} images per task."}
                # else:
                out = _execute_tool(
                    tool_name=tname,
                    tool_args=targs,
                    search_tools=search_tools,
                    current_img_path=current_img_path,
                    orig_copy=orig_copy,
                    tool_images_dir=tool_images_dir,
                    base_env=base_env,
                    python_exe=python_exe,
                    run_dir=run_dir,
                    turn=turn,
                    enable_search=enable_search,
                )
                # if tname == "download_image" and isinstance(out, dict) and out.get("ok"):
                #     download_count += 1

                tool_call_events.append(
                    {
                        "tool_name": tname,
                        "raw_tool_name": tname,
                        "arguments": targs,
                        "output": out,
                        "turn": turn,
                    }
                )
                results_for_model.append({"tool_name": tname, "arguments": targs, "output": out})

            # Feed results back to model
            messages.append({"role": "assistant", "content": raw})
            conversation_history.append({"role": "assistant", "content": raw})
            
            img_count = sum(1 for m in messages if isinstance(m.get("content"), list) 
                           and any(c.get("type") == "image_url" for c in m["content"] if isinstance(c, dict)))
            adaptive_params = get_adaptive_image_params(img_count)
            
            results_text = "<tool_results>" + json.dumps(results_for_model, ensure_ascii=False) + "</tool_results>"
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(
                            current_img_path, 
                            max_pixels=adaptive_params["max_pixels"],
                            quality=adaptive_params["quality"],
                            max_size_mb=adaptive_params["max_size_mb"]
                        )}},
                        {"type": "text", "text": results_text},
                    ],
                }
            )
            
            # Track in conversation history
            for tname, targs in normalized:
                conversation_history.append({
                    "role": "tool_call",
                    "content": json.dumps({"name": tname, "arguments": targs}, ensure_ascii=False),
                })
            for res in results_for_model:
                out = res.get("output", {})
                if isinstance(out, dict) and "context" in out:
                    resp_content = out["context"]
                elif isinstance(out, dict) and "text" in out:
                    resp_content = out["text"]
                elif isinstance(out, dict) and "error" in out:
                    resp_content = f"Error: {out['error']}"
                else:
                    resp_content = json.dumps(out, ensure_ascii=False)
                conversation_history.append({
                    "role": "tool_response",
                    "content": resp_content,
                })
            continue  # Go to next turn after tool execution
        
        # PRIORITY 2: Execute <code> block if present
        if code.strip():
            last_turn_had_code = True
            
            cleaned_code = _clean_deepeyes_code(code, tool_images_dir, turn=turn)
            
            # Write code directly without complex path cleaning
            # Let the model learn to use environment variables correctly
            code_path = run_dir / f"model_code_turn_{turn}.py"
            code_path.write_text(cleaned_code, encoding="utf-8")
            code_turn_paths.append(code_path)

            exec_stdout = run_dir / f"exec_stdout_turn_{turn}.txt"
            exec_stderr = run_dir / f"exec_stderr_turn_{turn}.txt"

            # Record existing images before execution
            imgs_before = set(p.name for p in list_all_output_images(tool_images_dir)) if tool_images_dir.exists() else set()

            env = dict(base_env)
            env["ORIGINAL_IMAGE_PATH"] = str(orig_copy.resolve())
            env["LOCAL_INPUT_IMAGE_PATH"] = str(current_img_path)
            env["PROCESSED_IMAGE_SAVE_PATH"] = str(tool_images_dir.resolve())

            exec_out = exec_python_file(
                python_exe=python_exe,
                script_path=code_path,
                cwd=run_dir,
                env=env,
                stdout_path=exec_stdout,
                stderr_path=exec_stderr,
                timeout_s=180,
            )
            exec_events.append(
                {
                    "tool_name": "run_visual_processing_code",
                    "raw_tool_name": "script_exec",
                    "arguments": {"python_exe": python_exe, "turn": turn},
                    "output": exec_out,
                }
            )

            # Find newly generated images - SAME as OpenAI
            imgs_after = set(p.name for p in list_all_output_images(tool_images_dir)) if tool_images_dir.exists() else set()
            new_img_names = imgs_after - imgs_before
            new_imgs = [p for p in list_all_output_images(tool_images_dir) if p.name in new_img_names]
            new_imgs.sort(key=lambda p: p.stat().st_mtime)
            
            # Add new images to image_list - SAME as OpenAI
            new_img_indices = []
            for img_p in new_imgs:
                match = re.match(r"transformed_image_(\d+)\.png", img_p.name, re.I)
                if match:
                    new_idx = int(match.group(1))
                else:
                    new_idx = len(image_list)
                image_list.append({"index": new_idx, "path": str(img_p), "label": f"generated in turn {turn}"})
                new_img_indices.append((new_idx, img_p))
            
            # Update current image - SAME as OpenAI
            imgs = list_transformed_images(tool_images_dir)
            if not imgs:
                imgs = list_all_output_images(tool_images_dir)
            if imgs:
                current_img_path = imgs[-1].resolve()
            
            # Build code execution result message - SAME as OpenAI
            code_result_parts = []
            if exec_out.get("ok") == "true":
                stdout_content = (run_dir / f"exec_stdout_turn_{turn}.txt").read_text(encoding="utf-8", errors="replace")[:2000]
                if stdout_content.strip():
                    code_result_parts.append(f"[Code output]\n{stdout_content}")
                else:
                    code_result_parts.append("[Code executed successfully, no output]")
            else:
                stderr_content = (run_dir / f"exec_stderr_turn_{turn}.txt").read_text(encoding="utf-8", errors="replace")[:2000]
                code_result_parts.append(f"[Code error]\n{stderr_content}")
            
            # List new images - SAME as OpenAI
            if new_img_indices:
                img_info_list = [f"Image {idx}: {p.name}" for idx, p in new_img_indices]
                code_result_parts.append(f"[New images: {', '.join(img_info_list)}]")
            
            code_result_msg = "\n".join(code_result_parts)
            
            # If model provided answer in same turn as code, accept it
            # (Matches OpenAI behavior - model may know answer after seeing code execution)
            if ans.strip():
                final_answer = ans.strip()
                break
            
            # Feed code execution result back to model - SAME as OpenAI
            messages.append({"role": "assistant", "content": raw})
            
            user_content = []
            # Use adaptive compression - SAME as OpenAI
            img_count = sum(1 for m in messages if isinstance(m.get("content"), list) 
                           and any(c.get("type") == "image_url" for c in m["content"] if isinstance(c, dict)))
            adaptive_params = get_adaptive_image_params(img_count)
            
            for idx, img_p in new_img_indices:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(
                        img_p, 
                        max_pixels=adaptive_params["max_pixels"],
                        quality=adaptive_params["quality"],
                        max_size_mb=adaptive_params["max_size_mb"]
                    )}
                })
                user_content.append({"type": "text", "text": f"[Image {idx}: {img_p.name}]"})
            user_content.append({"type": "text", "text": code_result_msg})
            
            messages.append({"role": "user", "content": user_content})
            
            # Track in conversation history - SAME as OpenAI
            conversation_history.append({"role": "assistant", "content": raw})
            history_parts = [code_result_msg]
            for idx, img_p in new_img_indices:
                history_parts.append(f"[Image {idx} shown: {img_p.name}]")
            conversation_history.append({"role": "user", "content": "\n".join(history_parts)})
            continue
        
        # PRIORITY 3: No tools or code, treat as final answer
        messages.append({"role": "assistant", "content": raw})
        conversation_history.append({"role": "assistant", "content": raw})
        final_answer = (ans or "").strip()
        break

    # If we exhausted all rounds, request final answer - SAME as OpenAI
    if not final_answer and turn == max_rounds - 1:
        final_prompt = "Please provide your final answer now based on all the information gathered. Use <answer>your answer</answer> format."
        messages.append({
            "role": "user",
            "content": final_prompt,
        })
        conversation_history.append({"role": "user", "content": final_prompt})
        
        try:
            final_resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=2000,
            )
            usage["api_calls"] += 1
            
            final_raw = final_resp.choices[0].message.content or ""
            (run_dir / f"raw_model_output_final.txt").write_text(final_raw, encoding="utf-8")
            conversation_history.append({"role": "assistant", "content": final_raw})
            
            _, _, final_ans, _, _ = parse_model_output(final_raw)
            if final_ans.strip():
                final_answer = final_ans.strip()
            else:
                final_answer = final_raw.strip()
        except Exception as e:
            all_warnings.append(f"[runtime] Failed to get final answer: {e}")

    # Save final answer - SAME as OpenAI
    (run_dir / "model_answer.txt").write_text(final_answer, encoding="utf-8")
    
    # Save image index - SAME as OpenAI
    write_json(run_dir / "image_index.json", image_list)
    
    # Save conversation history - SAME format as OpenAI
    tools_json = json.dumps(tools_list, ensure_ascii=False) if tools_list else "[]"
    conversation_record = {
        "task_id": task_cfg.get("task_id", ""),
        "tools": tools_json,
        "images": image_list,
        "messages": conversation_history,
        "final_answer": final_answer,
    }
    with open(run_dir / "conversation.json", "w", encoding="utf-8") as f:
        json.dump(conversation_record, f, ensure_ascii=False, indent=2)

    # Build tool_use_list - SAME as OpenAI
    tool_use_list: List[Dict[str, Any]] = []
    
    # Collect analysis across all code turns
    total_detected_ops = 0
    total_saves = 0
    total_prim_hist: Dict[str, int] = {}
    
    # NEW: collect per-op tool events
    all_tool_events: List[Dict[str, Any]] = []
    
    for code_path in code_turn_paths:
        code_content = code_path.read_text(encoding="utf-8")
        
        # Summary stats still based on infer_ops_and_saves
        ops, saves = infer_ops_and_saves(code_content)
        total_detected_ops += len(ops)
        total_saves += len(saves)
        
        # NEW: UI/tool_use_list should reflect every op
        tool_events = infer_tool_events(code_content)
        all_tool_events.extend(tool_events)
        
        # Histogram now counts every op occurrence
        for ev in tool_events:
            prim = ev.get("tool_name", "unknown")
            total_prim_hist[prim] = total_prim_hist.get(prim, 0) + 1
    
    # Get all images - SAME as OpenAI
    all_images: Dict[str, Path] = {}
    if tool_images_dir.exists():
        for p in tool_images_dir.iterdir():
            if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'):
                all_images[p.name] = p
    
    # NEW: Build tool entries from per-op tool events (not saves)
    tool_entries: List[Dict[str, Any]] = []
    for idx, ev in enumerate(all_tool_events):
        prim = ev.get("tool_name", "unknown")
        args = ev.get("arguments", {}) or {}
        save_name = str(args.get("save") or "")
        
        # Extract just filename if it's a path
        if '/' in save_name or '\\' in save_name:
            save_name = save_name.split('/')[-1].split('\\')[-1]
        
        p = all_images.get(save_name) if save_name else None
        
        out_paths: List[str] = []
        last_path = ""
        if save_name and p and p.exists():
            out_paths = [str(p)]
            last_path = str(p)
        
        tool_entries.append({
            "index": idx,
            "tool_name": prim,
            "raw_tool_name": "python_image_processing",
            "arguments": {
                "operation": prim,
                "save": save_name,
                "line": int(args.get("line") or 0),
            },
            "output": {
                "ok": "true" if out_paths else "false",
                "output_path": last_path,
                "output_paths": out_paths,
            },
            "timestamp": None,
        })
    
    tool_summary = {
        "num_detected_ops": total_detected_ops,
        "primitive_op_hist": total_prim_hist,
        "num_saves": total_saves,
        "num_images_found": len(all_images),
    }

    # Count search tool calls - SAME as OpenAI
    search_tool_hist: Dict[str, int] = {}
    for ev in tool_call_events:
        tname = ev.get("tool_name", "")
        if tname in ("google_search", "google_lens_search", "fetch_webpage"):  # download_image disabled
            search_tool_hist[tname] = search_tool_hist.get(tname, 0) + 1

    # Combine all events - SAME as OpenAI
    combined_entries: List[Dict[str, Any]] = []
    combined_entries.extend(tool_call_events)
    combined_entries.extend(tool_entries)
    combined_entries.extend(exec_events)

    for i, ev in enumerate(combined_entries):
        ev["index"] = i
        ev["timestamp"] = utc_ts()
        tool_use_list.append(ev)

    write_json(run_dir / "tool_use_list.json", tool_use_list)

    # Create replay script - SAME as OpenAI
    if code_turn_paths:
        replay_lines = [
            "import os, runpy",
            "from pathlib import Path",
            "",
            "RUN_DIR = Path(__file__).resolve().parent",
            "ORIG = RUN_DIR / 'orig.png'",
            "OUT = RUN_DIR / 'tool_images'",
            "OUT.mkdir(parents=True, exist_ok=True)",
            "",
            "os.environ['ORIGINAL_IMAGE_PATH'] = str(ORIG)",
            "os.environ['PROCESSED_IMAGE_SAVE_PATH'] = str(OUT)",
            "",
            "cur = ORIG",
            "",
        ]
        for i, pth in enumerate(code_turn_paths):
            replay_lines += [
                f"print('== TURN {i} ==')",
                "os.environ['LOCAL_INPUT_IMAGE_PATH'] = str(cur)",
                f"runpy.run_path(str(RUN_DIR / '{pth.name}'), run_name='__main__')",
                "imgs = sorted(OUT.glob('transformed_image_*.png'))",
                "if not imgs:",
                "    imgs = sorted([p for p in OUT.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])",
                "if imgs:",
                "    cur = imgs[-1]",
                "",
            ]
        replay_lines += ["print('Done. Outputs in:', OUT)"] 
        (run_dir / "replay_general_rollout.py").write_text("\n".join(replay_lines) + "\n", encoding="utf-8")
    else:
        write_replay_script(run_dir)

    # Save run metadata - SAME format as OpenAI
    run_meta = {
        "task_id": task_cfg.get("task_id", ""),
        "task_file": str(task_json.resolve()),
        "mode": "general_rollout",
        "driver": "deepeyes_local",  # Only difference: driver name
        "model": model,
        "temperature": temperature,
        "paths": {
            "run_dir": str(run_dir),
            "orig": str(orig_copy),
            "processed_dir": str(tool_images_dir),
        },
        "usage": usage,
        "effective_tool_calls": len(tool_call_events) + len(exec_events),  # Fixed: include code executions
        "total_images": len(image_list),
        "code_analysis": tool_summary,
        "search_analysis": {
            "search_tool_calls": sum(search_tool_hist.values()),
            "search_tool_hist": search_tool_hist,
        },
        "warnings": all_warnings,
    }
    write_json(run_dir / "run_meta.json", run_meta)
    return run_meta



# ============================================================================
# Main Function (IDENTICAL to OpenAI version, except for client creation)
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_json", type=str, default="")
    ap.add_argument("--task_dir", type=str, default="")
    ap.add_argument("--dataset_root", type=str, default="")
    ap.add_argument("--images_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory. Default: runs/general/deepeyes-rl")
    
    # Model configuration (DeepEyes-specific)
    ap.add_argument("--model_path", type=str, default="ChenShawn/DeepEyes-7B", 
                    help="Path to DeepEyes model (local path or HuggingFace model ID)")
    ap.add_argument("--device", type=str, default="cuda", help="Device to load model on (cuda/cpu)")
    ap.add_argument("--temperature", type=float, default=0.0)

    # Multi-turn tool execution - SAME as OpenAI
    ap.add_argument("--max_rounds", type=int, default=15, help="Max model turns")
    ap.add_argument("--max_tool_calls", type=int, default=15, help="Max total tool calls across turns")
    
    # Rate limiting - SAME as OpenAI
    ap.add_argument("--task_delay", type=float, default=0.5, help="Delay in seconds between tasks")

    # Web search - SAME as OpenAI
    ap.add_argument("--enable_search", action="store_true", help="Enable executable web/Lens search tools")
    ap.add_argument("--search_config", type=str, default="", help="Path to search config JSON")

    # Image size control - SAME as OpenAI
    ap.add_argument("--max_image_pixels", type=int, default=2048*2048, 
                    help="Maximum image pixels (width*height). Default: 4194304 (2048x2048)")
    ap.add_argument("--image_quality", type=int, default=95, 
                    help="JPEG quality for resized images (1-100). Default: 95")

    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable for code execution")
    
    # Skip and shard options - SAME as OpenAI
    ap.add_argument("--skip_existing", action="store_true", help="Skip tasks with existing run_meta.json")
    ap.add_argument("--shard", type=int, default=0, help="Shard index (0-based) for parallel runs")
    ap.add_argument("--num_shards", type=int, default=1, help="Total number of shards for parallel runs")
    ap.add_argument("--max_tasks", type=int, default=0, help="Max tasks to process (0 = unlimited)")
    
    args = ap.parse_args()

    tasks: List[Path] = []
    if args.task_json:
        tasks = [Path(args.task_json)]
    elif args.task_dir:
        tasks = sorted(Path(args.task_dir).glob("*.json"))
    else:
        raise ValueError("Provide --task_json or --task_dir")

    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    images_dir = Path(args.images_dir) if args.images_dir else None
    
    # Set output directory - SAME logic as OpenAI
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("runs/general/deepeyes-rl")

    # Create DeepEyes client (only difference from OpenAI)
    print("=" * 60)
    print("Initializing DeepEyes model for general mode...")
    print("=" * 60)
    client = make_deepeyes_client(args.model_path, args.device)
    print("Model ready!\n")
    
    # Apply sharding - SAME as OpenAI
    if args.num_shards > 1:
        total = len(tasks)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start_idx = args.shard * shard_size
        end_idx = min(start_idx + shard_size, total)
        tasks = tasks[start_idx:end_idx]
        print(f"[Shard {args.shard}/{args.num_shards}] Tasks {start_idx}-{end_idx-1} ({len(tasks)} tasks)")

    results = []
    skipped = 0
    processed = 0
    for idx, t in enumerate(tasks):
        # Check max_tasks limit - SAME as OpenAI
        if args.max_tasks > 0 and processed >= args.max_tasks:
            print(f"[LIMIT] Reached max_tasks={args.max_tasks}, stopping")
            break
        
        # Skip if already completed - SAME as OpenAI
        if args.skip_existing:
            run_dir = out_dir / t.stem
            if (run_dir / "run_meta.json").exists():
                skipped += 1
                print(f"[SKIP] {t.name} (already completed)")
                continue
        
        try:
            ds_root = resolve_dataset_root(t, dataset_root)
            
            res = run_one_rollout(
                client=client,
                task_json=t,
                dataset_root=ds_root,
                images_dir=images_dir,
                out_dir=out_dir,
                model="deepeyes-rl",
                temperature=args.temperature,
                python_exe=args.python,
                enable_search=args.enable_search,
                search_cfg_path=Path(args.search_config) if args.search_config else None,
                max_rounds=args.max_rounds,
                max_tool_calls=args.max_tool_calls,
                max_image_pixels=args.max_image_pixels,
                image_quality=args.image_quality,
                use_native_tools=False,  # DeepEyes doesn't support native tools
            )
            results.append(res)
            processed += 1
            print(f"[OK] {t.name} -> {Path(res['paths']['run_dir']).name} answer={Path(res['paths']['run_dir'])/'model_answer.txt'}")
            
            # Add delay between tasks - SAME as OpenAI
            if idx < len(tasks) - 1 and args.task_delay > 0:
                time.sleep(args.task_delay)
                
        except Exception as e:
            print(f"[ERR] {t}: {e}")
            traceback.print_exc()

    ensure_dir(out_dir)
    
    # Print summary - SAME as OpenAI
    print(f"\n=== Summary ===")
    print(f"Completed: {len(results)}")
    print(f"Skipped: {skipped}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
