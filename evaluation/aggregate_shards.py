#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate evaluation results directly from run directories.
Reads result_scored.json from each run folder, ignoring shard files.

Usage:
    python aggregate_shards.py --runs_dir runs/atomic/gpt-5-mini
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common_utils import read_json, write_json, ensure_dir


def aggregate_from_runs(runs_dir: Path) -> Dict[str, Any]:
    """
    Aggregate results by reading result_scored.json from each run directory.
    This is shard-agnostic and works regardless of how many shards were used.
    """
    # Find all run directories with result_scored.json
    all_results: List[Dict[str, Any]] = []
    missing_runs: List[str] = []
    error_runs: List[str] = []
    
    run_folders = sorted([p for p in runs_dir.iterdir() if p.is_dir()])
    
    for rd in run_folders:
        result_file = rd / "result_scored.json"
        if not result_file.exists():
            # Check if this is a valid run (has run_meta.json)
            if (rd / "run_meta.json").exists():
                missing_runs.append(rd.name)
            continue
        
        try:
            result = read_json(result_file)
            all_results.append(result)
        except Exception as e:
            error_runs.append(f"{rd.name}: {str(e)[:50]}")
    
    return {
        "results": all_results,
        "missing_runs": missing_runs,
        "error_runs": error_runs,
    }


def compute_aggregated_scores(results: List[Dict[str, Any]], runs_dir: Path) -> Dict[str, Any]:
    """
    Compute aggregated scores from individual results.
    Handles sum vs average correctly based on metric type.
    """
    if not results:
        return {}
    
    total = len(results)
    
    # === Core Metrics (need averaging) ===
    sum_pqs = sum(r['eval']['scores']['PQS'] for r in results)
    sum_s = sum(r['eval']['scores']['S'] for r in results)
    sum_v = sum(r['eval']['scores']['V'] for r in results)
    sum_r = sum(r['eval']['scores']['R'] for r in results)
    sum_penalty = sum(r['eval']['scores']['penalty'] for r in results)
    sum_base = sum(r['eval']['scores']['base'] for r in results)
    
    # === Final Answer Accuracy (count correct, then compute rate) ===
    final_correct = sum(1 for r in results if r['eval'].get('final_answer_accuracy', {}).get('correct', False))
    
    # === Efficiency (sum totals, then compute averages) ===
    total_effective_calls = sum(r['eval']['efficiency']['effective_tool_calls'] for r in results)
    total_ref_calls = sum(r['eval']['efficiency']['reference_tool_calls'] for r in results)
    total_excess = sum(r['eval']['efficiency']['excess_calls'] for r in results)
    
    # === Pass Rates (count, then compute rate) ===
    pqs_50_count = sum(1 for r in results if r['eval']['scores']['PQS'] >= 0.5)
    pqs_80_count = sum(1 for r in results if r['eval']['scores']['PQS'] >= 0.8)
    perfect_count = sum(1 for r in results if r['eval']['scores']['PQS'] == 1.0)
    
    # === Tool Usage (sum counts across all runs) ===
    tool_counter: Dict[str, int] = {}
    for r in results:
        # Try to get from run_meta's image_tool_analysis
        img_hist = r.get('image_tool_analysis', {}).get('image_tool_hist', {})
        for tool, count in img_hist.items():
            tool_counter[tool] = tool_counter.get(tool, 0) + count
        
        # Also check search_analysis
        search_hist = r.get('search_analysis', {}).get('search_tool_hist', {})
        for tool, count in search_hist.items():
            tool_counter[tool] = tool_counter.get(tool, 0) + count
        
        # Fallback: read tool_use_list.json if available
        if not img_hist and not search_hist:
            tool_list_path = Path(r['run_dir']) / "tool_use_list.json"
            if tool_list_path.exists():
                try:
                    tool_list = read_json(tool_list_path)
                    for ev in tool_list:
                        if (ev.get('output') or {}).get('declared_only'):
                            continue
                        ev_args = ev.get("arguments") or {}
                        op = ev_args.get("op") or ""
                        tool_name = ev.get("tool_name") or ev.get("raw_tool_name") or ""
                        key = (op.lower() if op else tool_name.lower()) if (op or tool_name) else None
                        if key:
                            tool_counter[key] = tool_counter.get(key, 0) + 1
                except:
                    pass
    
    sorted_tools = sorted(tool_counter.items(), key=lambda x: -x[1])
    
    # === Overthink Analysis ===
    overthink_cases = 0
    underthink_cases = 0
    optimal_cases = 0
    overthink_ratios = []
    
    for r in results:
        eff = r['eval'].get('efficiency', {})
        ref_calls = eff.get('reference_tool_calls', 0)
        model_calls = eff.get('effective_tool_calls', 0)
        
        if ref_calls > 0:
            ratio = model_calls / ref_calls
            overthink_ratios.append(ratio)
            if model_calls > ref_calls:
                overthink_cases += 1
            elif model_calls < ref_calls:
                underthink_cases += 1
            else:
                optimal_cases += 1
    
    avg_overthink_ratio = sum(overthink_ratios) / len(overthink_ratios) if overthink_ratios else 0
    
    # === Human Reference Tool Usage (from task configs) ===
    human_tool_counter: Dict[str, int] = {}
    human_s_checkpoints = 0
    
    for r in results:
        task_file = r.get('task_file', '')
        if task_file and Path(task_file).exists():
            try:
                task_cfg = read_json(Path(task_file))
                pe = task_cfg.get('process_evaluation', {})
                checkpoints = pe.get('checkpoints', [])
                s_cps = [cp for cp in checkpoints if cp.get('axis') == 'S']
                human_s_checkpoints += len(s_cps)
                
                for cp in s_cps:
                    cc = cp.get('code_check', {})
                    verifier = (cc.get('verifier', '') or '').lower()
                    
                    # Extract op from verifier name
                    for op in ('crop', 'rotate', 'resize', 'enhance', 'grayscale', 'blur', 
                               'sharpen', 'denoise', 'edge_detect', 'invert', 'equalize', 
                               'threshold', 'autocontrast'):
                        if op in verifier:
                            human_tool_counter[op] = human_tool_counter.get(op, 0) + 1
                            break
                    else:
                        # Check tools list
                        tools = cp.get('tools', [])
                        for t in tools:
                            t_lower = t.lower()
                            if 'search' in t_lower or 'web' in t_lower or 'google' in t_lower:
                                if 'lens' in verifier:
                                    human_tool_counter['google_lens_search'] = human_tool_counter.get('google_lens_search', 0) + 1
                                else:
                                    human_tool_counter['google_search'] = human_tool_counter.get('google_search', 0) + 1
                            elif 'calculator' in t_lower:
                                human_tool_counter['calculator'] = human_tool_counter.get('calculator', 0) + 1
            except:
                pass
    
    sorted_human_tools = sorted(human_tool_counter.items(), key=lambda x: -x[1])
    
    return {
        "total_cases": total,
        
        # Core Metrics (averaged)
        "avg_PQS": round(sum_pqs / total, 4),
        "avg_S": round(sum_s / total, 4),
        "avg_V": round(sum_v / total, 4),
        "avg_R": round(sum_r / total, 4),
        "avg_base": round(sum_base / total, 4),
        "avg_penalty": round(sum_penalty / total, 4),
        
        # Final Answer (count + rate)
        "final_answer_correct": final_correct,
        "final_answer_accuracy": round(final_correct / total, 4),
        
        # Efficiency (totals + averages)
        "total_tool_calls": total_effective_calls,
        "total_reference_calls": total_ref_calls,
        "total_excess_calls": total_excess,
        "avg_tool_calls": round(total_effective_calls / total, 2),
        "avg_reference_calls": round(total_ref_calls / total, 2),
        "avg_excess_calls": round(total_excess / total, 2),
        
        # Pass Rates (count + rate)
        "pqs_50_count": pqs_50_count,
        "pqs_80_count": pqs_80_count,
        "perfect_count": perfect_count,
        "pass_rate_PQS_50": round(pqs_50_count / total, 4),
        "pass_rate_PQS_80": round(pqs_80_count / total, 4),
        
        # Tool Usage (summed counts)
        "model_tool_usage": dict(sorted_tools),
        "human_tool_usage": dict(sorted_human_tools),
        "human_s_checkpoints": human_s_checkpoints,
        "human_avg_s_checkpoints": round(human_s_checkpoints / total, 2),
        
        # Overthink Analysis
        "overthink": {
            "cases_overthink": overthink_cases,
            "cases_underthink": underthink_cases,
            "cases_optimal": optimal_cases,
            "pct_overthink": round(overthink_cases / total * 100, 1),
            "pct_underthink": round(underthink_cases / total * 100, 1),
            "pct_optimal": round(optimal_cases / total * 100, 1),
            "avg_ratio": round(avg_overthink_ratio, 2),
        },
    }


def print_summary(scores: Dict[str, Any], missing: List[str], errors: List[str]):
    """Print a formatted summary of aggregated results."""
    print("\n" + "="*70)
    print("AGGREGATED EVALUATION RESULTS")
    print("="*70)
    
    if missing:
        print(f"\n⚠️  Missing evaluations: {len(missing)} runs")
        if len(missing) <= 10:
            print(f"   {', '.join(missing)}")
        else:
            print(f"   {', '.join(missing[:10])}... and {len(missing)-10} more")
    
    if errors:
        print(f"\n⚠️  Error loading: {len(errors)} runs")
    
    total = scores['total_cases']
    print(f"\n📊 Total Evaluated: {total} cases")
    
    print("\n--- Core Metrics ---")
    print(f"  Final Answer Accuracy: {scores['final_answer_accuracy']*100:.1f}% ({scores['final_answer_correct']}/{total})")
    print(f"  Average PQS: {scores['avg_PQS']:.4f}")
    print(f"  Average S (Skill): {scores['avg_S']:.4f}")
    print(f"  Average V (Visual): {scores['avg_V']:.4f}")
    print(f"  Average R (Order): {scores['avg_R']:.4f}")
    
    print("\n--- Efficiency ---")
    print(f"  Model Total Calls: {scores['total_tool_calls']}")
    print(f"  Human Ref Total: {scores['total_reference_calls']}")
    excess = scores['total_excess_calls']
    print(f"  Excess: {excess:+d} ({abs(excess)} {'more' if excess > 0 else 'fewer'})")
    print(f"  Avg Calls/Case: {scores['avg_tool_calls']:.2f} (ref: {scores['avg_reference_calls']:.2f})")
    print(f"  Avg Penalty: {scores['avg_penalty']:.4f}")
    
    print("\n--- Pass Rates ---")
    print(f"  PQS >= 50%: {scores['pass_rate_PQS_50']*100:.1f}% ({scores['pqs_50_count']}/{total})")
    print(f"  PQS >= 80%: {scores['pass_rate_PQS_80']*100:.1f}% ({scores['pqs_80_count']}/{total})")
    print(f"  Perfect (PQS=1.0): {scores['perfect_count']}")
    
    print("\n--- Overthink Analysis ---")
    ot = scores['overthink']
    print(f"  Overthink (model > ref): {ot['cases_overthink']} ({ot['pct_overthink']:.1f}%)")
    print(f"  Underthink (model < ref): {ot['cases_underthink']} ({ot['pct_underthink']:.1f}%)")
    print(f"  Optimal (model = ref): {ot['cases_optimal']} ({ot['pct_optimal']:.1f}%)")
    print(f"  Avg Overthink Ratio: {ot['avg_ratio']:.2f}x")
    
    print("\n--- Model Tool Usage (Top 10) ---")
    model_tools = list(scores['model_tool_usage'].items())[:10]
    for tool, count in model_tools:
        print(f"  {tool}: {count}")
    
    print("\n--- Human Expected Tools ---")
    human_tools = list(scores['human_tool_usage'].items())[:8]
    for tool, count in human_tools:
        print(f"  {tool}: {count}")
    
    print("="*70 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, required=True, 
                    help="Directory containing run results (e.g., runs/atomic/gpt-5-mini)")
    ap.add_argument("--out_json", type=str, default="",
                    help="Output JSON path. Default: runs/scores/{mode}_{model}_aggregated.json")
    args = ap.parse_args()
    
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Error: {runs_dir} does not exist")
        return
    
    # Extract mode and model from path
    parts = runs_dir.parts
    mode = parts[-2] if len(parts) >= 2 and parts[-2] in ("general", "atomic") else "unknown"
    model_name = parts[-1]
    
    print(f"Aggregating results from: {runs_dir}")
    print(f"Mode: {mode}, Model: {model_name}")
    
    # Aggregate from individual run directories
    data = aggregate_from_runs(runs_dir)
    results = data["results"]
    missing = data["missing_runs"]
    errors = data["error_runs"]
    
    print(f"Found {len(results)} evaluated runs")
    if missing:
        print(f"Missing evaluations: {len(missing)} runs")
    
    if not results:
        print("No results to aggregate!")
        return
    
    # Compute aggregated scores
    scores = compute_aggregated_scores(results, runs_dir)
    
    # Print summary
    print_summary(scores, missing, errors)
    
    # Save to JSON
    if args.out_json:
        out_path = Path(args.out_json)
    else:
        out_path = Path(f"runs/scores/{mode}_{model_name}_aggregated.json")
    
    ensure_dir(out_path.parent)
    write_json(out_path, {
        "model": model_name,
        "mode": mode,
        "aggregated_scores": scores,
        "missing_runs": missing,
        "error_runs": errors,
        "results": results,
    })
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
