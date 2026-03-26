#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common_utils import read_json, write_json, ensure_dir, image_to_data_url, make_openai_client, list_transformed_pngs
from verifiers import dispatch_verifier

def run_visual_judge(client: Any, judge_model: str, question: str, image_path: Path) -> str:
    """
    Run visual judge on an image. 
    Returns the model's answer to the question (can be Yes/No or short answer).
    Returns 'Error: ...' if something goes wrong.
    """
    try:
        # Check if image exists and is valid
        if not image_path.exists():
            return f"Error: Image not found: {image_path}"
        
        # Try to open the image first to validate it
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img.verify()  # Verify it's a valid image
        except Exception as img_err:
            return f"Error: Invalid image: {str(img_err)[:50]}"
        
        # Updated prompt to support both Yes/No and short answer questions
        sys_prompt = "You are a helpful assistant that answers questions about images. Provide concise, accurate answers based on what you see in the image."
        user = [
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            {"type": "text", "text": question},
        ]
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=100,  # Increased from 10 to support short answers
        )
        ans = (resp.choices[0].message.content or "").strip()
        
        # Return the answer as-is (no longer forcing Yes/No format)
        return ans
    except Exception as e:
        return f"Error: {str(e)[:100]}"


def run_search_judge(client: Any, judge_model: str, checkpoint_desc: str, 
                     model_queries: List[str], search_results: List[str]) -> Dict[str, Any]:
    """
    Use LLM to judge if the model's search queries align with the checkpoint requirements.
    
    Args:
        client: OpenAI client
        judge_model: Judge model name (e.g., "gpt-4o-mini")
        checkpoint_desc: Checkpoint description (contains expected keywords and findings)
        model_queries: List of actual search queries used by the model
        search_results: List of search result contexts
    
    Returns:
        {
            "passed": bool,
            "judge_response": str,
            "reasoning": str,
            "queries_evaluated": List[str],
            "expected_answer_found": bool (optional)
        }
    """
    # Extract expected answer from description if present
    # Support multiple formats: "expected to find:", "Expected:", etc.
    import re
    expected_answer = None
    
    # Try multiple patterns in order of specificity
    # Note: Using re.IGNORECASE to handle both lowercase and uppercase variants
    patterns = [
        r'\(expected to find:\s*([^)]+)\)',      # (expected to find: ...) or (Expected to find: ...)
        r'\(expected answer:\s*([^)]+)\)',       # (expected answer: ...) or (Expected answer: ...)
        r'\(expected:\s*([^)]+)\)',              # (expected: ...) or (Expected: ...)
        r'expected to find:\s*([^\n\)]+)',       # expected to find: ... (without parentheses)
        r'expected answer:\s*([^\n\)]+)',        # expected answer: ... (without parentheses)
        r'expected:\s*([^\n\)]+)',               # expected: ... or Expected: ... (without parentheses)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, checkpoint_desc, re.IGNORECASE)
        if match:
            expected_answer = match.group(1).strip()
            # Remove trailing punctuation
            expected_answer = expected_answer.rstrip('.,;:')
            break
    
    # Construct judge prompt
    queries_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(model_queries))
    results_text = "\n".join(f"Query {i+1} results: {r[:500]}..." for i, r in enumerate(search_results[:3]))
    
    # Add expected answer emphasis if present
    expected_answer_section = ""
    if expected_answer:
        expected_answer_section = f"""
**CRITICAL: Expected Answer to Find:**
{expected_answer}

You MUST verify that this specific information appears in the search results.
"""
    
    prompt = f"""You are evaluating whether a model's search successfully found the required information.

**Expected Search Strategy:**
{checkpoint_desc}
{expected_answer_section}
**Model's Actual Search Queries:**
{queries_text}

**Search Results Summary:**
{results_text}

**Evaluation Criteria (Two-Stage Check):**

STAGE 1 - Query Relevance (Lenient Requirement):
- Be LENIENT when evaluating query quality - broad queries are acceptable if they're reasonable
- Accept queries that mention the main entity, even if not highly specific
- Accept: entity name alone (e.g., "Chevrolet", "Chevrolet Wikipedia") - these can return useful information
- Accept: entity + general context (e.g., "Chevrolet China", "Chevrolet history")
- Accept: entity + specific information (e.g., "Chevrolet China entry date")
- Accept: synonyms, paraphrases, different word orders, translations
- Reject ONLY IF: wrong entity (e.g., "BMW" when looking for Chevrolet) OR completely unrelated topic

**Key principle for Stage 1:** If a reasonable person would consider the query relevant to finding the information, ACCEPT it.

STAGE 2 - Expected Answer (Primary Criterion):
- This is the MAIN criterion for passing
- Do the search results actually contain the expected answer/information?
- If an expected answer is specified, it MUST appear in the search results
- Even broad queries like "Chevrolet" or "Chevrolet Wikipedia" should PASS if results contain the answer

**Decision Logic:**
1. If queries are completely unrelated to the topic → FAIL
2. If queries are reasonable (even if broad) BUT results don't contain expected answer → FAIL
3. If queries are reasonable AND results contain expected answer → PASS

**Examples:**

Expected: "Chevrolet entered China date"
✅ PASS: "Chevrolet China entry" + results contain date (specific query, good results)
✅ PASS: "雪佛兰进入中国市场" + results contain date (translation, good results)
✅ PASS: "when did Chevrolet enter China" + results contain date (natural language, good results)
✅ PASS: "Chevrolet Wikipedia" + results contain date (broad but reasonable, found answer)
✅ PASS: "Chevrolet" + results contain date (very broad but reasonable, found answer)
✅ PASS: "Chevrolet China" + results contain date (reasonable scope, found answer)
❌ FAIL: "BMW entered China" + results about BMW (wrong entity)
❌ FAIL: "Chevrolet China entry" + results don't contain date (reasonable query but no answer found)
❌ FAIL: "car manufacturers" + results contain date (too generic, doesn't target entity)

**Response Format:**
VERDICT: [PASS/FAIL]
REASONING: [First check if query targets correct entity/topic, then check if expected answer was found in results]
"""
    
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300
        )
        
        judge_text = response.choices[0].message.content.strip()
        
        # Parse response
        passed = "PASS" in judge_text.upper().split("\n")[0]
        reasoning = ""
        for line in judge_text.split("\n"):
            if line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
                break
        
        result = {
            "passed": passed,
            "judge_response": judge_text,
            "reasoning": reasoning,
            "queries_evaluated": model_queries
        }
        
        # Add expected answer info if present
        if expected_answer:
            result["expected_answer"] = expected_answer
            # Try to determine if expected answer was found based on reasoning
            result["expected_answer_found"] = passed  # If passed, assume answer was found
        
        return result
    
    except Exception as e:
        return {
            "passed": False,
            "judge_response": f"Judge error: {str(e)}",
            "reasoning": "Error during evaluation",
            "queries_evaluated": model_queries
        }


def check_answer(cp: Dict[str, Any], model_answer: str) -> bool:
    cfg = cp["answer_check"]
    match_type = cfg.get("match_type", "exact")
    ans = (model_answer or "").strip()
    ans_lower = ans.lower()
    
    # Handle different match types
    if match_type in ("exact", "exact_match"):
        target = cfg.get("target", "")
        return ans == target
    
    if match_type == "contains":
        target = cfg.get("target", "")
        return target in ans
    
    if match_type == "string":
        # Flexible string match - check if key parts are present
        target = cfg.get("target", "")
        return target.lower() in ans_lower or ans_lower in target.lower()
    
    if match_type == "numeric":
        target = cfg.get("target", "0")
        try:
            v = float(ans)
            t = float(target)
            tol = float(cfg.get("tolerance", 0.0))
            return abs(v - t) <= tol
        except Exception:
            return False
    
    if match_type == "keyword_inclusion":
        # Check if any of the keywords are present in the answer
        keywords = cfg.get("keywords", [])
        min_matches = int(cfg.get("min_matches", 1))
        matches = sum(1 for kw in keywords if kw.lower() in ans_lower)
        return matches >= min_matches
    
    if match_type == "extraction_match":
        # Check multiple targets with tolerances
        targets = cfg.get("targets", [])
        if not targets:
            return False
        all_match = True
        for t in targets:
            key = t.get("key", "")
            value = t.get("value", "")
            tolerance = float(t.get("tolerance", 0.0))
            # Try to find the value in the answer
            try:
                # Look for numeric values in answer
                import re
                numbers = re.findall(r'[\d.]+', ans)
                found = False
                target_val = float(value)
                for num_str in numbers:
                    try:
                        num = float(num_str)
                        if abs(num - target_val) <= tolerance:
                            found = True
                            break
                    except:
                        continue
                if not found:
                    all_match = False
            except:
                all_match = False
        return all_match
    
    # Default: try contains check if target exists
    target = cfg.get("target", "")
    if target:
        return target.lower() in ans_lower
    return False

def find_any_hit_index_by_tools(cp_tools: List[str], tool_use_list: List[Dict[str, Any]]) -> Optional[int]:
    """Find the FIRST tool call that matches any of the checkpoint tools or operations."""
    tools_l = {t.lower() for t in (cp_tools or [])}

    def ev_tool_names(ev: Dict[str, Any]) -> List[str]:
        names: List[str] = []
        for k in ("tool_name", "raw_tool_name"):
            v = (ev.get(k) or "").strip()
            if v:
                names.append(v)
        args = ev.get("arguments") or {}
        for k in ("tool_name", "raw_tool_name", "op"):
            v = (args.get(k) or "").strip()
            if v:
                names.append(v)
        return [n.lower() for n in names]

    for ev in tool_use_list:
        if (ev.get('output') or {}).get('declared_only'):
            continue
        if any(n in tools_l for n in ev_tool_names(ev)):
            return int(ev.get("index", 0))
    return None


def find_hit_by_op(op: str, tool_use_list: List[Dict[str, Any]]) -> Optional[int]:
    """Find the FIRST tool call that matches a specific operation (crop, rotate, etc.)."""
    op = op.lower().strip()
    
    for ev in tool_use_list:
        if (ev.get('output') or {}).get('declared_only'):
            continue
        
        # Check tool_name directly (for atomic mode)
        tool_name = (ev.get("tool_name") or "").lower()
        if tool_name == op:
            return int(ev.get("index", 0))
        
        # Check op in arguments (for general mode)
        args = ev.get("arguments") or {}
        arg_op = (args.get("op") or "").lower()
        if arg_op == op:
            return int(ev.get("index", 0))
    
    return None


def find_last_hit_index_by_tools(cp_tools: List[str], tool_use_list: List[Dict[str, Any]]) -> Optional[int]:
    """Find the LAST tool call that matches any of the checkpoint tools."""
    tools_l = {t.lower() for t in (cp_tools or [])}

    def ev_tool_names(ev: Dict[str, Any]) -> List[str]:
        names: List[str] = []
        for k in ("tool_name", "raw_tool_name"):
            v = (ev.get(k) or "").strip()
            if v:
                names.append(v)
        args = ev.get("arguments") or {}
        for k in ("tool_name", "raw_tool_name", "op"):
            v = (args.get(k) or "").strip()
            if v:
                names.append(v)
        return [n.lower() for n in names]

    last_hit = None
    for ev in tool_use_list:
        if (ev.get('output') or {}).get('declared_only'):
            continue
        if any(n in tools_l for n in ev_tool_names(ev)):
            last_hit = int(ev.get("index", 0))
    return last_hit

def collect_images_for_visual_check(run_dir: Path, tool_use_list: List[Dict[str, Any]], start_index: int, cp_tools: Optional[List[str]] = None) -> List[Path]:
    """
    Collect ALL transformed/processed images for visual check.
    
    Key insight: Model may try multiple times (trial and error), so we should check
    ALL images produced by the relevant tools, and pass if ANY of them is correct.
    
    Returns a list of image paths (may be empty).
    """
    from PIL import Image
    
    def is_valid_image(p: Path) -> bool:
        """Check if image file is valid and can be opened."""
        try:
            with Image.open(p) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    tools_l = {t.lower() for t in (cp_tools or [])} if cp_tools else set()
    
    def ev_tool_names(ev: Dict[str, Any]) -> List[str]:
        names: List[str] = []
        for k in ("tool_name", "raw_tool_name"):
            v = (ev.get(k) or "").strip()
            if v:
                names.append(v)
        args = ev.get("arguments") or {}
        for k in ("tool_name", "raw_tool_name", "op"):
            v = (args.get(k) or "").strip()
            if v:
                names.append(v)
        return [n.lower() for n in names]
    
    # Collect ALL matching images
    images: List[Path] = []
    seen_paths: set = set()
    
    for ev in tool_use_list:
        if (ev.get('output') or {}).get('declared_only'):
            continue
        ev_index = int(ev.get("index", 0))
        if ev_index < int(start_index):
            continue
        
        # If cp_tools specified, only consider matching tools
        if tools_l:
            if not any(n in tools_l for n in ev_tool_names(ev)):
                continue
        
        out = ev.get("output") or {}
        for key in ("output_path", "saved_path", "image_path"):
            cand = out.get(key) or ""
            if cand:
                p = Path(cand)
                if p.exists() and str(p) not in seen_paths and is_valid_image(p):
                    images.append(p)
                    seen_paths.add(str(p))
                    break
    
    # Also check tool_images directory
    tool_images_dir = run_dir / "tool_images"
    if tool_images_dir.exists():
        for img in list_transformed_pngs(tool_images_dir):
            if str(img) not in seen_paths and is_valid_image(img):
                images.append(img)
                seen_paths.add(str(img))
    
    return images

def check_final_answer_only(task_cfg: Dict[str, Any], model_answer: str) -> Dict[str, Any]:
    """
    Check if the final answer matches the golden answer.
    This is a simple accuracy metric that ignores the process.
    
    Enhanced logic for numeric answers:
    - For single/double digit numbers with 'contains' match, uses word boundary matching
    - This prevents false positives like "1" matching "2017"
    
    Returns:
        {"correct": bool, "model_answer": str, "golden_answer": str, "match_type": str}
    """
    import re
    
    golden = task_cfg.get("golden_answer") or {}
    golden_value = str(golden.get("value", "")).strip()
    match_type = golden.get("match_type", "contains")  # default to contains for flexibility
    tolerance = float(golden.get("tolerance", 0.0))
    
    ans = (model_answer or "").strip()
    correct = False
    
    if match_type == "exact":
        correct = ans == golden_value
    elif match_type == "contains":
        # Enhanced logic for numeric golden answers
        # Check if golden_value is purely numeric (including decimals and negative numbers)
        is_numeric = golden_value.replace('.', '').replace('-', '').replace('+', '').isdigit()
        
        if is_numeric and len(golden_value) <= 2:
            # For single/double digit numbers, use word boundary matching to avoid false positives
            # Example: "1" should not match "2017", but should match "answer is 1"
            
            # Strategy 1: Try to find in final answer section first
            final_answer_patterns = [
                r'therefore[,\s]+(?:the\s+)?answer\s+is\s+([^\.\n]+)',
                r'(?:the\s+)?answer\s+is\s+([^\.\n]+)',
                r'answer:\s*([^\.\n]+)',
                r'final\s+answer:\s*([^\.\n]+)',
            ]
            
            found_in_final = False
            for pattern in final_answer_patterns:
                match = re.search(pattern, ans.lower())
                if match:
                    final_section = match.group(1)
                    # Use word boundary in final section
                    if re.search(r'\b' + re.escape(golden_value) + r'\b', final_section):
                        found_in_final = True
                        break
            
            if found_in_final:
                correct = True
            else:
                # Strategy 2: If no final answer pattern found, use word boundary on full text
                # This is more lenient but still avoids matching partial numbers
                correct = bool(re.search(r'\b' + re.escape(golden_value) + r'\b', ans))
        else:
            # For non-numeric or multi-digit numbers, use standard case-insensitive contains
            correct = golden_value.lower() in ans.lower()
    elif match_type == "numeric":
        try:
            v = float(ans)
            t = float(golden_value)
            correct = abs(v - t) <= tolerance
        except Exception:
            correct = False
    else:
        # Default: contains check
        correct = golden_value.lower() in ans.lower()
    
    return {
        "correct": correct,
        "model_answer": ans,
        "golden_answer": golden_value,
        "match_type": match_type
    }

def eval_process(task_cfg: Dict[str, Any], run_dir: Path, run_meta: Dict[str, Any], model_answer: str, tool_use_list: List[Dict[str, Any]], vjudge, client: Any = None, judge_model: str = "gpt-4o-mini") -> Dict[str, Any]:
    pe = task_cfg.get("process_evaluation") or {}
    checkpoints = pe.get("checkpoints") or []
    ordering = pe.get("ordering_constraints") or []
    eff = pe.get("efficiency") or {}

    cp_results: Dict[str, Any] = {}
    cp_hit: Dict[str, Optional[int]] = {}
    
    # Track which tool_use_list entries have been matched to checkpoints
    # This ensures one-to-one matching: each tool call can only satisfy one checkpoint
    used_tool_indices: set = set()

    for cp in checkpoints:
        cid = cp["id"]
        axis = cp["axis"]
        passed = False
        hit_index: Optional[int] = None
        extra: Dict[str, Any] = {}

        if axis == "S":
            # ===== NEW LOGIC: S-axis only handles search tools =====
            cp_tools = cp.get("tools") or []
            tools_l = {t.lower() for t in cp_tools}
            
            # Check if this is a search-related checkpoint
            is_google_search = "google_search" in tools_l
            is_google_lens = "google_lens_search" in tools_l
            
            if is_google_search or is_google_lens:
                # Handle search tools
                if is_google_search:
                    # google_search: needs LLM judge to evaluate search quality
                    search_queries = []
                    search_results = []
                    search_indices = []
                    
                    for ev in tool_use_list:
                        if (ev.get('output') or {}).get('declared_only'):
                            continue
                        
                        tool_name = (ev.get("tool_name") or "").lower()
                        if tool_name == "google_search":
                            ev_index = int(ev.get("index", 0))
                            
                            # Skip already used tool calls (one-to-one matching)
                            if ev_index in used_tool_indices:
                                continue
                            
                            args = ev.get("arguments") or {}
                            query = args.get("query", "")
                            if query:
                                search_queries.append(query)
                                output = ev.get("output") or {}
                                context = output.get("context", "")
                                search_results.append(context)
                                search_indices.append(ev_index)
                    
                    if search_queries and client:
                        search_check_result = run_search_judge(
                            client=client,
                            judge_model=judge_model,
                            checkpoint_desc=cp.get("description", ""),
                            model_queries=search_queries,
                            search_results=search_results
                        )
                        
                        if search_check_result["passed"]:
                            passed = True
                            hit_index = search_indices[0] if search_indices else None
                            if hit_index is not None:
                                used_tool_indices.add(hit_index)
                        
                        extra["search_check"] = search_check_result
                
                elif is_google_lens:
                    # google_lens_search: one-to-one matching
                    for ev in tool_use_list:
                        ev_index = int(ev.get("index", 0))
                        
                        if ev_index in used_tool_indices:
                            continue
                        
                        if (ev.get('output') or {}).get('declared_only'):
                            continue
                        
                        tool_name = (ev.get("tool_name") or "").lower()
                        if tool_name == "google_lens_search":
                            passed = True
                            hit_index = ev_index
                            used_tool_indices.add(ev_index)
                            extra["tool_hit_index"] = ev_index
                            extra["matched_by"] = "google_lens_search"
                            break
            else:
                # Not a search tool - S-axis should not handle this
                passed = False
                extra["error"] = "S-axis checkpoint must be search-related (google_search or google_lens_search)"
            
            cp_hit[cid] = hit_index
            cp_results[cid] = {"axis": axis, "passed": bool(passed), "hit_index": hit_index, **extra}
            continue

        elif axis == "R":
            if "answer_check" in cp:
                passed = check_answer(cp, model_answer)
                hit_index = len(tool_use_list)
                cp_hit[cid] = hit_index
                cp_results[cid] = {"axis": axis, "passed": bool(passed), "hit_index": hit_index, "answer_check": cp["answer_check"]}
                continue
            
            cp_hit[cid] = None
            cp_results[cid] = {"axis": axis, "passed": False, "error": "Unknown R checkpoint type"}
            continue

        elif axis == "V":
            # Special case: final answer check (should be minimal output)
            if "answer_check" in cp:
                cp_hit[cid] = None
                cp_results[cid] = {"axis": axis}
                continue
            
            # ===== NEW LOGIC: V-axis handles image processing tools + visual checks =====
            tool_check_passed = False
            visual_check_passed = False
            has_tool_check = False
            has_visual_check = False
            hit_index = None  # Initialize hit_index
            
            # Case 1: Tool check (code_check) - for image processing operations
            if "code_check" in cp:
                has_tool_check = True
                cc = cp["code_check"] or {}
                verifier = cc.get("verifier", "") or ""
                
                # Extract operation type (crop, rotate, resize, enhance, flip)
                verifier_op = None
                verifier_lower = verifier.lower()
                for prim in ("crop", "rotate", "flip", "resize", "enhance"):
                    if prim in verifier_lower:
                        verifier_op = prim
                        break
                
                # First try to run verifier
                vr = dispatch_verifier(verifier, run_dir, run_meta, task_cfg, cp, tool_use_list)
                if vr.passed:
                    tool_check_passed = True
                    hit_index = vr.hit_index
                    if hit_index is not None:
                        used_tool_indices.add(hit_index)
                    extra["code_check"] = {
                        "verifier": verifier, 
                        "verifier_op": verifier_op, 
                        "passed": True, 
                        "hit_index": vr.hit_index, 
                        "detail": vr.detail or {}
                    }
                
                # If verifier didn't pass, try to match tool name directly (one-to-one)
                if not tool_check_passed and verifier_op:
                    for ev in tool_use_list:
                        ev_index = int(ev.get("index", 0))
                        
                        # Skip already used tool calls
                        if ev_index in used_tool_indices:
                            continue
                        
                        if (ev.get('output') or {}).get('declared_only'):
                            continue
                        
                        # Check tool_name directly (for atomic mode)
                        tool_name = (ev.get("tool_name") or "").lower()
                        if tool_name == verifier_op:
                            tool_check_passed = True
                            hit_index = ev_index
                            used_tool_indices.add(ev_index)
                            extra["op_hit_index"] = ev_index
                            extra["matched_by"] = "verifier_op"
                            break
                        
                        # Check op in arguments (for general mode)
                        args = ev.get("arguments") or {}
                        arg_op = (args.get("op") or "").lower()
                        if arg_op == verifier_op:
                            tool_check_passed = True
                            hit_index = ev_index
                            used_tool_indices.add(ev_index)
                            extra["op_hit_index"] = ev_index
                            extra["matched_by"] = "verifier_op"
                            break
                
                # If no verifier_op, try using tools list as fallback
                if not tool_check_passed:
                    cp_tools = cp.get("tools") or []
                    if cp_tools:
                        tools_l = {t.lower() for t in cp_tools}
                        
                        def ev_tool_names(ev: Dict[str, Any]) -> List[str]:
                            names: List[str] = []
                            for k in ("tool_name", "raw_tool_name"):
                                v = (ev.get(k) or "").strip()
                                if v:
                                    names.append(v)
                            args = ev.get("arguments") or {}
                            for k in ("tool_name", "raw_tool_name", "op"):
                                v = (args.get(k) or "").strip()
                                if v:
                                    names.append(v)
                            return [n.lower() for n in names]
                        
                        for ev in tool_use_list:
                            ev_index = int(ev.get("index", 0))
                            
                            if ev_index in used_tool_indices:
                                continue
                            
                            if (ev.get('output') or {}).get('declared_only'):
                                continue
                            
                            if any(n in tools_l for n in ev_tool_names(ev)):
                                tool_check_passed = True
                                hit_index = ev_index
                                used_tool_indices.add(ev_index)
                                extra["tool_hit_index"] = ev_index
                                extra["matched_by"] = "tools_list"
                                break
            
            # Case 2: Visual check (visual_check)
            if "visual_check" in cp:
                has_visual_check = True
                vc = cp["visual_check"]
                
                # Collect all generated images using the comprehensive collection function
                # This will check both tool_use_list outputs and tool_images directory
                tool_images_dir = run_dir / "tool_images"
                all_images: List[Path] = []
                
                # Method 1: Collect from tool_images directory (all PNG files, not just transformed_image_*.png)
                if tool_images_dir.exists():
                    from PIL import Image
                    # Get all PNG/JPG files, not just transformed_image_*.png pattern
                    for img_file in tool_images_dir.iterdir():
                        if img_file.is_file() and img_file.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                            try:
                                with Image.open(img_file) as test_img:
                                    test_img.verify()
                                all_images.append(img_file)
                            except Exception:
                                pass
                
                # Method 2: If no images found in tool_images, check tool_use_list outputs
                if not all_images:
                    for ev in tool_use_list:
                        if (ev.get('output') or {}).get('declared_only'):
                            continue
                        out = ev.get("output") or {}
                        for key in ("output_path", "saved_path", "image_path"):
                            cand = out.get(key) or ""
                            if cand:
                                p = Path(cand)
                                if p.exists():
                                    try:
                                        from PIL import Image
                                        with Image.open(p) as test_img:
                                            test_img.verify()
                                        all_images.append(p)
                                        break
                                    except Exception:
                                        pass
                
                # Method 3: If still no images found, fall back to original image
                if not all_images:
                    orig_img = run_dir / "orig.png"
                    if not orig_img.exists():
                        orig_candidates = list(run_dir.glob("orig*.png"))
                        if orig_candidates:
                            orig_img = orig_candidates[0]
                    
                    if orig_img.exists():
                        all_images.append(orig_img)
                
                if all_images:
                    # Check all images - pass if ANY of them satisfies the visual check
                    judge_results = []
                    for img_path in all_images:
                        judge_ans = vjudge(vc["question"], img_path)
                        expected = vc["expected_answer"].strip()
                        judge_ans_clean = judge_ans.strip()
                        
                        # Check if expected answer is in judge answer (case-insensitive)
                        img_passed = (expected.lower() in judge_ans_clean.lower()) or (judge_ans_clean.lower() == expected.lower())
                        
                        judge_results.append({
                            "image": str(img_path.name),
                            "judge_answer": judge_ans,
                            "expected": expected,
                            "passed": img_passed
                        })
                        
                        # If any image passes, mark visual check as passed and stop
                        if img_passed:
                            visual_check_passed = True
                            break
                    
                    extra["visual_check"] = {
                        "question": vc["question"],
                        "expected_answer": vc["expected_answer"],
                        "judge_results": judge_results,
                        "passed": visual_check_passed,
                        "images_checked": len(judge_results)
                    }
                else:
                    extra["visual_check"] = {
                        "error": "No images found for visual check",
                        "passed": False
                    }
            
            # Comprehensive judgment: if both tool check and visual check exist, both must pass
            if has_tool_check and has_visual_check:
                passed = tool_check_passed and visual_check_passed
            elif has_tool_check:
                passed = tool_check_passed
            elif has_visual_check:
                passed = visual_check_passed
            else:
                passed = False
                extra["error"] = "V-axis checkpoint must have code_check or visual_check"
            
            cp_hit[cid] = hit_index
            cp_results[cid] = {"axis": axis, "passed": bool(passed), "hit_index": hit_index, **extra}
            continue

        elif axis == "R":
            if "answer_check" in cp:
                passed = check_answer(cp, model_answer)
                hit_index = len(tool_use_list)
                cp_hit[cid] = hit_index
                cp_results[cid] = {"axis": axis, "passed": bool(passed), "hit_index": hit_index, "answer_check": cp["answer_check"]}
                continue
            
            cp_hit[cid] = None
            cp_results[cid] = {"axis": axis, "passed": False, "error": "Unknown R checkpoint type"}
            continue

        else:
            cp_hit[cid] = None
            cp_results[cid] = {"axis": axis, "passed": False, "error": f"Unknown axis {axis}"}

    r_total, r_pass = len(ordering), 0
    r_details = []
    for oc in ordering:
        seq = oc["sequence"]
        max_gap = int(oc.get("max_gap", 0))
        ok = True
        indices: List[int] = []

        for cid in seq:
            if not cp_results.get(cid, {}).get("passed", False):
                ok = False
                break
            idx = cp_hit.get(cid)
            if idx is None:
                ok = False
                break
            indices.append(int(idx))

        if ok:
            for a, b in zip(indices, indices[1:]):
                # Allow equal indices (same tool call can satisfy multiple checkpoints)
                # Only fail if strictly decreasing (b < a means wrong order)
                if b < a:
                    ok = False
                    break
                if (b - a - 1) > max_gap:
                    ok = False
                    break

        if ok:
            r_pass += 1
        r_details.append({"id": oc.get("id", ""), "passed": bool(ok), "indices": indices, "max_gap": max_gap})

    s_ids = [cp["id"] for cp in checkpoints if cp["axis"] == "S"]
    # Exclude final answer checkpoint (identified by "answer_check" key) from V-axis calculation
    v_ids = [cp["id"] for cp in checkpoints if cp["axis"] == "V" and "answer_check" not in cp]
    s_pass = sum(1 for cid in s_ids if cp_results[cid]["passed"])
    v_pass = sum(1 for cid in v_ids if cp_results[cid]["passed"])

    # Set to None if no checkpoints exist for that axis
    S = s_pass / len(s_ids) if s_ids else None
    V = v_pass / len(v_ids) if v_ids else None
    R = r_pass / r_total if r_total else 1.0

    ref_calls = int(eff.get("reference_tool_calls", 0))
    penalty_per = float(eff.get("penalty_per_extra_call", 0.0))
    max_calls = int(eff.get("max_tool_calls", 0) or 0)

    effective_calls = int(run_meta.get("effective_tool_calls", len(tool_use_list)))
    excess = max(0, effective_calls - ref_calls)
    penalty = min(1.0, excess * penalty_per)
    if max_calls > 0 and effective_calls > max_calls:
        penalty = 1.0

    # Calculate base only from axes that have checkpoints
    valid_scores = [score for score in [S, V] if score is not None]
    base = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    PQS = base * (1.0 - penalty)
    
    return {
        "checkpoint_results": cp_results, 
        # "ordering_results": r_details,  # Commented out: ordering results not needed in output
        "scores": {
            "S": S, 
            "V": V, 
            "R": R,  # Keep for internal calculation
            "base": base, 
            "penalty": penalty, 
            "PQS": PQS  # Keep for internal calculation
        }, 
        "efficiency": {
            "effective_tool_calls": effective_calls,
            "reference_tool_calls": ref_calls,
            "excess_calls": excess,
        }
    }

def score_one_run(client: Any, judge_model: str, run_dir: Path) -> Dict[str, Any]:
    # Check if run_meta.json exists - if not, this is an incomplete run due to API failure
    if not (run_dir / "run_meta.json").exists():
        # Create a minimal failed result for incomplete runs
        # These runs failed due to API issues and should be counted as complete failures
        
        # Try to infer task info from directory structure or use placeholder
        task_file = "unknown"
        model_name = "unknown"
        
        # Create failed evaluation result with all checkpoints marked as failed
        failed_eval = {
            "checkpoint_results": {},
            "ordering_results": [],
            "scores": {
                "S": 0.0,
                "V": 0.0,
                "R": 0.0,
                "base": 0.0,
                "penalty": 0.0,
                "PQS": 0.0
            },
            "efficiency": {
                "effective_tool_calls": 0,
                "reference_tool_calls": 0,
                "excess_calls": 0
            },
            "final_answer_accuracy": {
                "correct": False,
                "model_answer": "",
                "golden_answer": "",
                "match_type": "unknown"
            },
            "incomplete_run": True,
            "failure_reason": "Missing run_meta.json - incomplete execution"
        }
        
        scored = {
            "run_dir": str(run_dir),
            "task_file": task_file,
            "model_name": model_name,
            "model_answer": "",
            "golden_answer": "",
            "eval": failed_eval,
            "incomplete_run": True
        }
        
        write_json(run_dir / "result_scored.json", scored)
        return scored
    
    run_meta = read_json(run_dir / "run_meta.json")
    task_cfg = read_json(Path(run_meta["task_file"]))
    tool_use_list = read_json(run_dir / "tool_use_list.json") if (run_dir / "tool_use_list.json").exists() else []
    model_answer = (run_dir / "model_answer.txt").read_text(encoding="utf-8").strip() if (run_dir / "model_answer.txt").exists() else ""

    def vjudge(question: str, img: Path) -> str:
        return run_visual_judge(client, judge_model, question, img)

    ev = eval_process(task_cfg, run_dir, run_meta, model_answer, tool_use_list, vjudge, client=client, judge_model=judge_model)
    
    # Add final answer accuracy (simple metric that only checks if answer is correct)
    final_answer_check = check_final_answer_only(task_cfg, model_answer)
    ev["final_answer_accuracy"] = final_answer_check
    
    # Remove PQS and R from scores before saving to JSON
    scores_for_json = {k: v for k, v in ev["scores"].items() if k not in ["PQS", "R"]}
    ev_for_json = {**ev, "scores": scores_for_json}
    
    scored = {"run_dir": str(run_dir), **run_meta, "model_answer": model_answer, "golden_answer": (task_cfg.get("golden_answer") or {}).get("value",""), "eval": ev_for_json}
    write_json(run_dir / "result_scored.json", scored)
    
    # Return original ev with PQS and R for internal calculations
    return {"run_dir": str(run_dir), **run_meta, "model_answer": model_answer, "golden_answer": (task_cfg.get("golden_answer") or {}).get("value",""), "eval": ev}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, required=True, help="Directory containing run results (e.g., runs/general/gpt-4o)")
    ap.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--api_key", type=str, default="")
    ap.add_argument("--base_url", type=str, default="")
    ap.add_argument("--api_config", type=str, default="")
    ap.add_argument("--out_json", type=str, default="", help="Output JSON path. Default: runs/scores/{mode}_{model}_scored.json")
    
    # Sharding and skip options
    ap.add_argument("--skip_existing", action="store_true", help="Skip runs that already have result_scored.json")
    ap.add_argument("--shard", type=int, default=0, help="Shard index (0-based) for parallel evaluation")
    ap.add_argument("--num_shards", type=int, default=1, help="Total number of shards for parallel evaluation")
    
    args = ap.parse_args()

    client = make_openai_client(api_key=args.api_key or None, base_url=args.base_url or None, api_config=Path(args.api_config) if args.api_config else None)

    runs_dir = Path(args.runs_dir)
    # Include both complete runs (with run_meta.json) and incomplete runs (with orig.png but no run_meta.json)
    # Incomplete runs are typically caused by API failures and should be counted as failures
    run_folders = sorted([
        p for p in runs_dir.iterdir() 
        if p.is_dir() and (
            (p / "run_meta.json").exists() or  # Complete run
            (p / "orig.png").exists()  # Incomplete run with at least orig.png
        )
    ])
    
    # Apply sharding if specified (contiguous blocks)
    if args.num_shards > 1:
        total = len(run_folders)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start_idx = args.shard * shard_size
        end_idx = min(start_idx + shard_size, total)
        run_folders = run_folders[start_idx:end_idx]
        print(f"[Shard {args.shard}/{args.num_shards}] Evaluating {len(run_folders)} runs (index {start_idx}-{end_idx-1})")

    results = []
    skipped = 0
    for rd in run_folders:
        # Skip if already evaluated
        if args.skip_existing and (rd / "result_scored.json").exists():
            skipped += 1
            # Load existing result for aggregation
            try:
                existing = read_json(rd / "result_scored.json")
                results.append(existing)
                # Use .get() to safely access PQS (may not exist in incomplete runs)
                pqs = existing['eval']['scores'].get('PQS', 0.0)
                print(f"[SKIP] {rd.name} (already evaluated, PQS={pqs:.3f})")
            except Exception:
                print(f"[SKIP] {rd.name} (already evaluated)")
            continue
        
        try:
            res = score_one_run(client, args.judge_model, rd)
            results.append(res)
            
            # Check if this is an incomplete run
            if res.get('incomplete_run', False):
                print(f"[INCOMPLETE] {rd.name} - model failure, marked as failed (S=0.00 V=0.00)")
            else:
                s_val = res['eval']['scores']['S']
                v_val = res['eval']['scores']['V']
                s_str = f"{s_val:.2f}" if s_val is not None else "N/A"
                v_str = f"{v_val:.2f}" if v_val is not None else "N/A"
                print(f"[OK] {rd.name} S={s_str} V={v_str}")
        except Exception as e:
            print(f"[ERR] {rd}: {e}")
            traceback.print_exc()
    
    # Print shard summary
    if args.num_shards > 1:
        print(f"\n[Shard {args.shard}] Completed: {len(results) - skipped}, Skipped: {skipped}")

    # Calculate overall model scores (aggregated across all cases)
    overall_scores = {}
    overthink_analysis = {}
    level_scores = {}  # Store scores by level
    
    if results:
        # Group results by level
        results_by_level = {'L1': [], 'L2': [], 'L3': []}
        for r in results:
            task_file = r.get('task_file', '') or r.get('task_json', '')
            if task_file and Path(task_file).exists():
                try:
                    task_cfg = read_json(Path(task_file))
                    # Extract level from meta.level field
                    meta = task_cfg.get('meta', {})
                    level_num = meta.get('level', 0)
                    
                    # Map level number to level name
                    if level_num == 1:
                        results_by_level['L1'].append(r)
                    elif level_num == 2:
                        results_by_level['L2'].append(r)
                    elif level_num == 3:
                        results_by_level['L3'].append(r)
                except:
                    pass
        
        # Helper function to calculate scores for a subset of results
        def calculate_scores_for_subset(subset_results, subset_name=""):
            if not subset_results:
                return None
            
            total = len(subset_results)
            # Use .get() with default 0.0 to handle missing PQS field
            sum_pqs = sum(r['eval']['scores'].get('PQS', 0.0) for r in subset_results)
            
            # Only sum non-None S and V scores
            s_scores = [r['eval']['scores']['S'] for r in subset_results if r['eval']['scores']['S'] is not None]
            v_scores = [r['eval']['scores']['V'] for r in subset_results if r['eval']['scores']['V'] is not None]
            sum_s = sum(s_scores)
            sum_v = sum(v_scores)
            
            # Use .get() with default 0.0 to handle missing R field
            sum_r = sum(r['eval']['scores'].get('R', 0.0) for r in subset_results)
            sum_penalty = sum(r['eval']['scores'].get('penalty', 0.0) for r in subset_results)
            
            # Final answer accuracy
            final_correct = sum(1 for r in subset_results if r['eval'].get('final_answer_accuracy', {}).get('correct', False))
            
            # Efficiency metrics
            total_effective_calls = sum(r['eval']['efficiency']['effective_tool_calls'] for r in subset_results)
            
            # Checkpoint pass rate metrics
            cases_pass_50_checkpoints = 0
            cases_pass_80_checkpoints = 0
            
            for r in subset_results:
                checkpoint_results = r['eval'].get('checkpoint_results', {})
                if checkpoint_results:
                    # Count total checkpoints and passed checkpoints (excluding final answer checkpoint)
                    total_checkpoints = 0
                    passed_checkpoints = 0
                    
                    for cp_id, cp_data in checkpoint_results.items():
                        # Skip final answer checkpoint (axis == "V" with no other checks)
                        if cp_data.get('axis') == 'V' and len(cp_data) == 1:
                            continue
                        
                        total_checkpoints += 1
                        if cp_data.get('passed', False):
                            passed_checkpoints += 1
                    
                    if total_checkpoints > 0:
                        pass_rate = passed_checkpoints / total_checkpoints
                        if pass_rate >= 0.5:
                            cases_pass_50_checkpoints += 1
                        if pass_rate >= 0.8:
                            cases_pass_80_checkpoints += 1
            
            # Tool usage statistics
            tool_counter: Dict[str, int] = {}
            for r in subset_results:
                tool_list_path = Path(r['run_dir']) / "tool_use_list.json"
                if tool_list_path.exists():
                    tool_list = read_json(tool_list_path)
                    for ev in tool_list:
                        if (ev.get('output') or {}).get('declared_only'):
                            continue
                        
                        ev_args = ev.get("arguments") or {}
                        op = ev_args.get("op") or ""
                        tool_name = ev.get("tool_name") or ev.get("raw_tool_name") or ""
                        
                        if op:
                            key = op.lower()
                        elif tool_name:
                            key = tool_name.lower()
                        else:
                            continue
                        
                        tool_counter[key] = tool_counter.get(key, 0) + 1
            
            sorted_tools = sorted(tool_counter.items(), key=lambda x: -x[1])
            
            return {
                "total_cases": total,
                # "avg_PQS": round(sum_pqs / total, 4),  # Commented out: PQS not needed in JSON output
                "avg_S": round(sum_s / len(s_scores), 4) if s_scores else 0.0,
                "avg_V": round(sum_v / len(v_scores), 4) if v_scores else 0.0,
                # "avg_R": round(sum_r / total, 4),  # Commented out: R-axis not needed in JSON output
                "total_s_cases": len(s_scores),
                "total_v_cases": len(v_scores),
                "final_answer_accuracy": round(final_correct / total, 4),
                "final_answer_correct_count": final_correct,
                "avg_penalty": round(sum_penalty / total, 4),
                "total_tool_calls": total_effective_calls,
                "avg_tool_calls_per_case": round(total_effective_calls / total, 2),
                "unique_tool_types": len(tool_counter),
                "tool_usage": dict(sorted_tools),
                # "pass_rate_PQS_50": round(sum(1 for r in subset_results if r['eval']['scores']['PQS'] >= 0.5) / total, 4),  # Commented out: PQS not needed
                # "pass_rate_PQS_80": round(sum(1 for r in subset_results if r['eval']['scores']['PQS'] >= 0.8) / total, 4),  # Commented out: PQS not needed
                "perfect_cases": sum(1 for r in subset_results if r['eval']['scores'].get('PQS', 0.0) == 1.0),
                # New checkpoint pass rate metrics
                "checkpoint_pass_rate_50": round(cases_pass_50_checkpoints / total, 4),
                "checkpoint_pass_rate_80": round(cases_pass_80_checkpoints / total, 4),
                "cases_pass_50_checkpoints": cases_pass_50_checkpoints,
                "cases_pass_80_checkpoints": cases_pass_80_checkpoints,
            }
        
        # Calculate scores for each level
        for level_name, level_results in results_by_level.items():
            if level_results:
                level_scores[level_name] = calculate_scores_for_subset(level_results, level_name)
        
        # Calculate overall scores (existing logic)
        total = len(results)
        # Use .get() with default 0.0 to handle missing PQS field
        sum_pqs = sum(r['eval']['scores'].get('PQS', 0.0) for r in results)
        
        # Only sum non-None S and V scores
        s_scores = [r['eval']['scores']['S'] for r in results if r['eval']['scores']['S'] is not None]
        v_scores = [r['eval']['scores']['V'] for r in results if r['eval']['scores']['V'] is not None]
        sum_s = sum(s_scores)
        sum_v = sum(v_scores)
        
        # Use .get() with default 0.0 to handle missing R field
        sum_r = sum(r['eval']['scores'].get('R', 0.0) for r in results)
        sum_penalty = sum(r['eval']['scores'].get('penalty', 0.0) for r in results)
        
        # Final answer accuracy (simple metric)
        final_correct = sum(1 for r in results if r['eval'].get('final_answer_accuracy', {}).get('correct', False))
        
        # Efficiency metrics
        total_effective_calls = sum(r['eval']['efficiency']['effective_tool_calls'] for r in results)
        
        # Tool usage statistics - count by sub-operation (crop, rotate, etc.)
        tool_counter: Dict[str, int] = {}
        for r in results:
            tool_list_path = Path(r['run_dir']) / "tool_use_list.json"
            if tool_list_path.exists():
                tool_list = read_json(tool_list_path)
                for ev in tool_list:
                    if (ev.get('output') or {}).get('declared_only'):
                        continue
                    
                    # Get the actual operation (sub-type), not the wrapper tool name
                    ev_args = ev.get("arguments") or {}
                    
                    # Priority: op > tool_name in args > raw tool_name
                    op = ev_args.get("op") or ""
                    tool_name = ev.get("tool_name") or ev.get("raw_tool_name") or ""
                    
                    # For atomic mode tools, use the tool_name directly (crop, rotate, etc.)
                    # For general mode, use op if available, otherwise tool_name
                    if op:
                        # Use the sub-operation (crop, rotate, resize, enhance, etc.)
                        key = op.lower()
                    elif tool_name:
                        # Use tool name directly (google_search, google_lens_search, calculator, etc.)
                        key = tool_name.lower()
                    else:
                        continue
                    
                    tool_counter[key] = tool_counter.get(key, 0) + 1
        
        # Sort tools by frequency
        sorted_tools = sorted(tool_counter.items(), key=lambda x: -x[1])
        
        # === Overthink Analysis ===
        # Compare model tool usage vs human reference
        human_total_ref_calls = 0
        human_total_s_checkpoints = 0
        human_tool_counter: Dict[str, int] = {}
        overthink_cases = 0
        underthink_cases = 0
        optimal_cases = 0
        exceeded_max_cases = 0
        overthink_ratios = []
        
        for r in results:
            eff = r['eval'].get('efficiency', {})
            ref_calls = eff.get('reference_tool_calls', 0)
            model_calls = eff.get('effective_tool_calls', 0)
            
            human_total_ref_calls += ref_calls
            
            # Count S checkpoints from task config
            task_file = r.get('task_file', '')
            if task_file and Path(task_file).exists():
                try:
                    task_cfg = read_json(Path(task_file))
                    pe = task_cfg.get('process_evaluation', {})
                    checkpoints = pe.get('checkpoints', [])
                    s_cps = [cp for cp in checkpoints if cp.get('axis') == 'S']
                    human_total_s_checkpoints += len(s_cps)
                    
                    # Extract human tool usage from S checkpoints (search tools only)
                    # S-axis: google_search, google_lens_search
                    for cp in s_cps:
                        tools = cp.get('tools', [])
                        for t in tools:
                            t_lower = t.lower()
                            # Directly use tool name from tools list
                            # S-axis only contains: google_search, google_lens_search
                            if t_lower in ('google_search', 'google_lens_search'):
                                human_tool_counter[t_lower] = human_tool_counter.get(t_lower, 0) + 1
                            else:
                                # Keep original name for any other tools
                                human_tool_counter[t_lower] = human_tool_counter.get(t_lower, 0) + 1
                    
                    # Extract human tool usage from V checkpoints (visual tools)
                    # V-axis: crop, flip, rotate, enhance, resize
                    v_cps = [cp for cp in checkpoints if cp.get('axis') == 'V']
                    for cp in v_cps:
                        tools = cp.get('tools', [])
                        cc = cp.get('code_check', {})
                        verifier = (cc.get('verifier', '') or '').lower()
                        
                        for t in tools:
                            t_lower = t.lower()
                            # For V-axis, extract specific operation from verifier
                            # e.g., verify_crop_ast -> crop, verify_rotate_image_ast -> rotate
                            matched_op = None
                            for op in ('crop', 'flip', 'rotate', 'enhance', 'resize'):
                                if op in verifier:
                                    matched_op = op
                                    break
                            
                            if matched_op:
                                human_tool_counter[matched_op] = human_tool_counter.get(matched_op, 0) + 1
                            else:
                                # If no specific op found, use tool name directly
                                human_tool_counter[t_lower] = human_tool_counter.get(t_lower, 0) + 1
                    
                    # Check max_tool_calls
                    max_calls = pe.get('efficiency', {}).get('max_tool_calls', 0)
                    if max_calls > 0 and model_calls > max_calls:
                        exceeded_max_cases += 1
                except:
                    pass
            
            # Overthink classification
            if ref_calls > 0:
                if model_calls > ref_calls:
                    overthink_cases += 1
                elif model_calls < ref_calls:
                    underthink_cases += 1
                else:
                    optimal_cases += 1
                overthink_ratios.append(model_calls / ref_calls)
        
        avg_overthink_ratio = sum(overthink_ratios) / len(overthink_ratios) if overthink_ratios else 0
        sorted_human_tools = sorted(human_tool_counter.items(), key=lambda x: -x[1])
        
        overthink_analysis = {
            # Human reference
            "human_total_reference_calls": human_total_ref_calls,
            "human_avg_reference_calls": round(human_total_ref_calls / total, 2),
            "human_total_s_checkpoints": human_total_s_checkpoints,
            "human_avg_s_checkpoints": round(human_total_s_checkpoints / total, 2),
            "human_tool_usage": dict(sorted_human_tools),
            
            # Model actual
            "model_total_calls": total_effective_calls,
            "model_avg_calls": round(total_effective_calls / total, 2),
            
            # Comparison
            "total_excess_calls": total_effective_calls - human_total_ref_calls,
            "avg_excess_calls": round((total_effective_calls - human_total_ref_calls) / total, 2),
            "avg_overthink_ratio": round(avg_overthink_ratio, 2),
            
            # Distribution
            "cases_overthink": overthink_cases,
            "cases_underthink": underthink_cases,
            "cases_optimal": optimal_cases,
            "cases_exceeded_max": exceeded_max_cases,
            "pct_overthink": round(overthink_cases / total * 100, 1),
            "pct_underthink": round(underthink_cases / total * 100, 1),
            "pct_optimal": round(optimal_cases / total * 100, 1),
        }
        
        # === Search Evaluation Statistics (新增) ===
        search_checkpoints_total = 0
        search_checkpoints_passed = 0
        
        for r in results:
            # Count total google_search checkpoints from original task JSON
            task_file = r.get('task_json')
            if task_file:
                try:
                    task_cfg = read_json(Path(task_file))
                    pe = task_cfg.get('process_evaluation', {})
                    checkpoints = pe.get('checkpoints', [])
                    
                    # Count checkpoints that require google_search tool
                    for cp in checkpoints:
                        cp_tools = cp.get("tools") or []
                        if any("google_search" == t.lower() for t in cp_tools):
                            search_checkpoints_total += 1
                            
                            # Check if this checkpoint passed in the evaluation
                            cid = cp.get("id")
                            cp_results = r['eval'].get('checkpoint_results', {})
                            cp_data = cp_results.get(cid, {})
                            if cp_data.get('search_check', {}).get('passed', False):
                                search_checkpoints_passed += 1
                except:
                    pass
        
        # Calculate checkpoint pass rates for overall
        cases_pass_50_checkpoints = 0
        cases_pass_80_checkpoints = 0
        
        for r in results:
            checkpoint_results = r['eval'].get('checkpoint_results', {})
            if checkpoint_results:
                # Count total checkpoints and passed checkpoints (excluding final answer checkpoint)
                total_checkpoints = 0
                passed_checkpoints = 0
                
                for cp_id, cp_data in checkpoint_results.items():
                    # Skip final answer checkpoint (axis == "V" with no other checks)
                    if cp_data.get('axis') == 'V' and len(cp_data) == 1:
                        continue
                    
                    total_checkpoints += 1
                    if cp_data.get('passed', False):
                        passed_checkpoints += 1
                
                if total_checkpoints > 0:
                    pass_rate = passed_checkpoints / total_checkpoints
                    if pass_rate >= 0.5:
                        cases_pass_50_checkpoints += 1
                    if pass_rate >= 0.8:
                        cases_pass_80_checkpoints += 1
        
        overall_scores = {
            "total_cases": total,
            
            # === Core Metrics ===
            # "avg_PQS": round(sum_pqs / total, 4),  # Commented out: PQS not needed in JSON output
            "avg_S": round(sum_s / len(s_scores), 4) if s_scores else 0.0,
            "avg_V": round(sum_v / len(v_scores), 4) if v_scores else 0.0,
            # "avg_R": round(sum_r / total, 4),  # Commented out: R-axis not needed in JSON output
            "total_s_cases": len(s_scores),  # Number of cases with S-axis checkpoints
            "total_v_cases": len(v_scores),  # Number of cases with V-axis checkpoints
            
            # === Final Answer (ignores process) ===
            "final_answer_accuracy": round(final_correct / total, 4),
            "final_answer_correct_count": final_correct,
            
            # === Efficiency ===
            "avg_penalty": round(sum_penalty / total, 4),
            "total_tool_calls": total_effective_calls,
            "avg_tool_calls_per_case": round(total_effective_calls / total, 2),
            
            # === Tool Usage Statistics ===
            "unique_tool_types": len(tool_counter),
            "tool_usage": dict(sorted_tools),  # tool_name -> count
            
            # === Pass Rates ===
            # "pass_rate_PQS_50": round(sum(1 for r in results if r['eval']['scores']['PQS'] >= 0.5) / total, 4),  # Commented out: PQS not needed
            # "pass_rate_PQS_80": round(sum(1 for r in results if r['eval']['scores']['PQS'] >= 0.8) / total, 4),  # Commented out: PQS not needed
            "perfect_cases": sum(1 for r in results if r['eval']['scores'].get('PQS', 0.0) == 1.0),
            
            # === Checkpoint Pass Rates ===
            "checkpoint_pass_rate_50": round(cases_pass_50_checkpoints / total, 4),
            "checkpoint_pass_rate_80": round(cases_pass_80_checkpoints / total, 4),
            "cases_pass_50_checkpoints": cases_pass_50_checkpoints,
            "cases_pass_80_checkpoints": cases_pass_80_checkpoints,
            
            # === Search Evaluation ===
            "search_checkpoints_total": search_checkpoints_total,
            "search_checkpoints_passed": search_checkpoints_passed,
            "search_checkpoint_pass_rate": round(search_checkpoints_passed / search_checkpoints_total, 4) if search_checkpoints_total > 0 else 0.0,
        }
        
        print("\n" + "="*70)
        print("OVERALL MODEL SCORES")
        print("="*70)
        print(f"Total Cases: {total}")
        print()
        print("--- Core Metrics ---")
        print(f"  Final Answer Accuracy: {overall_scores['final_answer_accuracy']*100:.1f}% ({final_correct}/{total})")
        # print(f"  Average PQS: {overall_scores['avg_PQS']:.4f}")  # Commented out: PQS not needed
        print(f"  Average S (Skill): {overall_scores['avg_S']:.4f} (based on {overall_scores['total_s_cases']} cases with S-axis)")
        print(f"  Average V (Visual): {overall_scores['avg_V']:.4f} (based on {overall_scores['total_v_cases']} cases with V-axis)")
        # print(f"  Average R (Order): {overall_scores['avg_R']:.4f}")  # Commented out: R-axis not needed
        print()
        print("--- Efficiency ---")
        print(f"  Total Tool Calls: {total_effective_calls}")
        print(f"  Avg Calls/Case: {overall_scores['avg_tool_calls_per_case']:.2f}")
        print(f"  Avg Penalty: {overall_scores['avg_penalty']:.4f}")
        print()
        print("--- Pass Rates ---")
        # print(f"  PQS >= 50%: {overall_scores['pass_rate_PQS_50']*100:.1f}%")  # Commented out: PQS not needed
        # print(f"  PQS >= 80%: {overall_scores['pass_rate_PQS_80']*100:.1f}%")  # Commented out: PQS not needed
        print(f"  Cases Pass >= 50% Checkpoints: {overall_scores['checkpoint_pass_rate_50']*100:.1f}% ({overall_scores['cases_pass_50_checkpoints']}/{total})")
        print(f"  Cases Pass >= 80% Checkpoints: {overall_scores['checkpoint_pass_rate_80']*100:.1f}% ({overall_scores['cases_pass_80_checkpoints']}/{total})")
        print(f"  Perfect: {overall_scores['perfect_cases']}")
        
        # === Print Level-wise Scores ===
        if level_scores:
            print("\n" + "="*70)
            print("LEVEL-WISE BREAKDOWN")
            print("="*70)
            
            for level_name in ['L1', 'L2', 'L3']:
                if level_name in level_scores and level_scores[level_name]:
                    ls = level_scores[level_name]
                    print(f"\n--- {level_name} ---")
                    print(f"  Total Cases: {ls['total_cases']}")
                    print(f"  Final Answer Accuracy: {ls['final_answer_accuracy']*100:.1f}% ({ls['final_answer_correct_count']}/{ls['total_cases']})")
                    # print(f"  Average PQS: {ls['avg_PQS']:.4f}")  # Commented out: PQS not needed
                    print(f"  Average S: {ls['avg_S']:.4f} (based on {ls['total_s_cases']} cases)")
                    print(f"  Average V: {ls['avg_V']:.4f} (based on {ls['total_v_cases']} cases)")
                    # print(f"  Average R: {ls['avg_R']:.4f}")  # Commented out: R-axis not needed
                    print(f"  Avg Tool Calls/Case: {ls['avg_tool_calls_per_case']:.2f}")
                    # print(f"  PQS >= 50%: {ls['pass_rate_PQS_50']*100:.1f}%")  # Commented out: PQS not needed
                    # print(f"  PQS >= 80%: {ls['pass_rate_PQS_80']*100:.1f}%")  # Commented out: PQS not needed
                    print(f"  Cases Pass >= 50% Checkpoints: {ls['checkpoint_pass_rate_50']*100:.1f}% ({ls['cases_pass_50_checkpoints']}/{ls['total_cases']})")
                    print(f"  Cases Pass >= 80% Checkpoints: {ls['checkpoint_pass_rate_80']*100:.1f}% ({ls['cases_pass_80_checkpoints']}/{ls['total_cases']})")
                    print(f"  Perfect: {ls['perfect_cases']}")
        
        # === Model Tool Usage (Overall) ===
        print("\n" + "="*70)
        print("MODEL TOOL USAGE (Overall)")
        print("="*70)
        print(f"Unique Tool Types: {len(tool_counter)}")
        for tool, count in sorted_tools[:10]:  # Top 10 tools
            print(f"  {tool}: {count}")
        if len(sorted_tools) > 10:
            print(f"  ... and {len(sorted_tools) - 10} more")
        
        
        # === Search Evaluation Output ===
        if overall_scores.get('search_checkpoints_total', 0) > 0:
            print()
            print("--- Search Evaluation ---")
            print(f"  Search Checkpoints: {overall_scores['search_checkpoints_passed']}/{overall_scores['search_checkpoints_total']} passed ({overall_scores['search_checkpoint_pass_rate']*100:.1f}%)")

        # Overthink Analysis
        print()
        print("="*70)
        print("OVERTHINK ANALYSIS (Model vs Human Reference)")
        print("="*70)
        print()
        print("--- Human Reference (Ground Truth) ---")
        print(f"  Total Reference Calls: {overthink_analysis['human_total_reference_calls']}")
        print(f"  Avg Reference Calls/Case: {overthink_analysis['human_avg_reference_calls']:.2f}")
        print(f"  Avg S-Checkpoints/Case: {overthink_analysis['human_avg_s_checkpoints']:.2f}")
        print("  Human Expected Tool Usage:")
        for tool, count in sorted_human_tools[:8]:
            print(f"    {tool}: {count}")
        print()
        print("--- Model Actual ---")
        print(f"  Model Total Calls: {overthink_analysis['model_total_calls']}")
        print(f"  Model Avg Calls/Case: {overthink_analysis['model_avg_calls']:.2f}")
        print()
        print("--- Comparison (Model - Human) ---")
        excess = overthink_analysis['total_excess_calls']
        avg_excess = overthink_analysis['avg_excess_calls']
        print(f"  Total Excess Calls: {excess:+d} (model used {abs(excess)} {'more' if excess > 0 else 'fewer'} calls than human)")
        print(f"  Avg Excess/Case: {avg_excess:+.2f}")
        print(f"  Avg Overthink Ratio: {overthink_analysis['avg_overthink_ratio']:.2f}x")
        print()
        print("--- Distribution ---")
        print(f"  Overthink (model > ref): {overthink_analysis['cases_overthink']} ({overthink_analysis['pct_overthink']:.1f}%)")
        print(f"  Underthink (model < ref): {overthink_analysis['cases_underthink']} ({overthink_analysis['pct_underthink']:.1f}%)")
        print(f"  Optimal (model = ref): {overthink_analysis['cases_optimal']} ({overthink_analysis['pct_optimal']:.1f}%)")
        print(f"  Exceeded Max Allowed: {overthink_analysis['cases_exceeded_max']}")
        print("="*70 + "\n")

    # Auto-generate output path based on runs_dir structure
    if args.out_json:
        out_path = Path(args.out_json)
    else:
        # Extract mode and model from runs_dir path (e.g., runs/general/gpt-4o -> general_gpt-4o)
        parts = runs_dir.parts
        if len(parts) >= 2:
            mode = parts[-2] if parts[-2] in ("general", "atomic") else "unknown"
            model_name = parts[-1]
            # Add shard suffix if running in shard mode
            if args.num_shards > 1:
                out_path = Path(f"runs/scores/{mode}_{model_name}_shard{args.shard}_scored.json")
            else:
                out_path = Path(f"runs/scores/{mode}_{model_name}_scored.json")
        else:
            out_path = runs_dir / "summary_scored.json"
    
    ensure_dir(out_path.parent)
    
    # For shard mode: only save summary stats, not individual results (those are in result_scored.json)
    # For full mode: save everything
    output_data = {
        "model": runs_dir.name,
        "mode": parts[-2] if len(parts) >= 2 and parts[-2] in ("general", "atomic") else "unknown",
        "shard": args.shard if args.num_shards > 1 else None,
        "num_shards": args.num_shards if args.num_shards > 1 else None,
        "overall_scores": overall_scores,
        "level_scores": level_scores,  # Add level-wise scores
        "overthink_analysis": overthink_analysis,
        "count": len(results),
        # Only include run_ids for reference, not full results
        "run_ids": [Path(r['run_dir']).name for r in results] if args.num_shards > 1 else None,
    }
    
    # Only include full results in non-shard mode
    if args.num_shards <= 1:
        output_data["results"] = results
    
    write_json(out_path, output_data)
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
