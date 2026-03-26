#!/usr/bin/env python3
"""
统计 V 轴 checkpoint 的通过率 (增强版)
分为 tool-use 类和 visual-check 类
按 overall 和 level (L1, L2, L3) 分别统计
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def is_final_answer_checkpoint_from_config(cp_config):
    """从 task config 判断是否是 final answer checkpoint"""
    # final answer checkpoint 有 answer_check 字段
    return cp_config.get('axis') == 'V' and 'answer_check' in cp_config

def is_tool_use_checkpoint_from_config(cp_config):
    """从 task config 判断是否是 tool-use 类 V 轴 checkpoint"""
    # V 轴，不是 final answer，有 code_check 但没有 visual_check
    return (cp_config.get('axis') == 'V' and 
            not is_final_answer_checkpoint_from_config(cp_config) and
            'code_check' in cp_config and
            'visual_check' not in cp_config)

def is_visual_check_checkpoint_from_config(cp_config):
    """从 task config 判断是否是 visual-check 类 V 轴 checkpoint"""
    # V 轴，不是 final answer，有 visual_check 字段
    return (cp_config.get('axis') == 'V' and 
            not is_final_answer_checkpoint_from_config(cp_config) and
            'visual_check' in cp_config)

def is_final_answer_checkpoint_from_result(cp_data):
    """从评估结果判断是否是 final answer checkpoint"""
    # final answer checkpoint 只有 axis 字段，没有其他字段
    return cp_data.get('axis') == 'V' and len(cp_data) == 1

def is_tool_use_checkpoint_from_result(cp_data):
    """从评估结果判断是否是 tool-use 类 V 轴 checkpoint"""
    # V 轴，不是 final answer，没有 visual_check 字段
    return (cp_data.get('axis') == 'V' and 
            not is_final_answer_checkpoint_from_result(cp_data) and
            'visual_check' not in cp_data)

def is_visual_check_checkpoint_from_result(cp_data):
    """从评估结果判断是否是 visual-check 类 V 轴 checkpoint"""
    # V 轴，不是 final answer，有 visual_check 字段
    return (cp_data.get('axis') == 'V' and 
            not is_final_answer_checkpoint_from_result(cp_data) and
            'visual_check' in cp_data)

def get_level_from_task_file(task_file):
    """从 task_file 读取 level 信息"""
    try:
        task_data = read_json(task_file)
        level = task_data.get('meta', {}).get('level')
        if level in [1, 2, 3]:
            return f'L{level}'
    except Exception as e:
        print(f"Warning: Cannot read level from {task_file}: {e}", file=sys.stderr)
    return None

def find_task_config(run_id, task_config_dir):
    """根据 run_id 查找对应的 task config 文件"""
    task_config_path = Path(task_config_dir) / f"{run_id}.json"
    if task_config_path.exists():
        return task_config_path
    return None

def count_checkpoints_from_config(task_config_path):
    """从 task config 统计各类 checkpoint 数量"""
    try:
        task_cfg = read_json(task_config_path)
        pe = task_cfg.get('process_evaluation', {})
        checkpoints = pe.get('checkpoints', [])
        
        tool_use_count = 0
        visual_check_count = 0
        
        for cp in checkpoints:
            if is_tool_use_checkpoint_from_config(cp):
                tool_use_count += 1
            elif is_visual_check_checkpoint_from_config(cp):
                visual_check_count += 1
        
        level = task_cfg.get('meta', {}).get('level')
        level_name = f'L{level}' if level in [1, 2, 3] else None
        
        return {
            'tool_use': tool_use_count,
            'visual_check': visual_check_count,
            'level': level_name
        }
    except Exception as e:
        print(f"Warning: Cannot read checkpoints from {task_config_path}: {e}", file=sys.stderr)
        return None

def analyze_checkpoints(scored_json_path, task_config_dir=None):
    """分析 V 轴 checkpoint 的统计信息"""
    data = read_json(scored_json_path)
    results = data.get('results', [])
    
    # 按 level 分组统计
    stats = {
        'Overall': {
            'tool_use': {'total': 0, 'passed': 0},
            'visual_check': {'total': 0, 'passed': 0}
        },
        'L1': {
            'tool_use': {'total': 0, 'passed': 0},
            'visual_check': {'total': 0, 'passed': 0}
        },
        'L2': {
            'tool_use': {'total': 0, 'passed': 0},
            'visual_check': {'total': 0, 'passed': 0}
        },
        'L3': {
            'tool_use': {'total': 0, 'passed': 0},
            'visual_check': {'total': 0, 'passed': 0}
        }
    }
    
    missing_evaluations = []
    processed_runs = set()
    
    # 遍历所有 results
    for result in results:
        run_dir = result.get('run_dir', '')
        if run_dir:
            run_id = Path(run_dir).name
            processed_runs.add(run_id)
        
        task_file = result.get('task_file', '')
        
        # 检查是否是 incomplete run (缺失 result_scored.json)
        is_incomplete = result.get('incomplete_run', False)
        
        if is_incomplete and task_config_dir and run_dir:
            # 处理缺失评估的案例：从 task config 读取 checkpoint 信息
            run_id = Path(run_dir).name
            task_config_path = find_task_config(run_id, task_config_dir)
            
            if task_config_path:
                checkpoint_counts = count_checkpoints_from_config(task_config_path)
                if checkpoint_counts:
                    level = checkpoint_counts['level']
                    
                    # 将所有 checkpoint 标记为失败
                    tool_use_count = checkpoint_counts['tool_use']
                    visual_check_count = checkpoint_counts['visual_check']
                    
                    stats['Overall']['tool_use']['total'] += tool_use_count
                    stats['Overall']['visual_check']['total'] += visual_check_count
                    
                    if level:
                        stats[level]['tool_use']['total'] += tool_use_count
                        stats[level]['visual_check']['total'] += visual_check_count
                    
                    missing_evaluations.append(run_id)
                    continue
        
        # 正常处理有评估结果的案例
        level = get_level_from_task_file(task_file) if task_file else None
        checkpoint_results = result.get('eval', {}).get('checkpoint_results', {})
        
        for cp_id, cp_data in checkpoint_results.items():
            # 统计 tool-use 类 checkpoint
            if is_tool_use_checkpoint_from_result(cp_data):
                stats['Overall']['tool_use']['total'] += 1
                if cp_data.get('passed', False):
                    stats['Overall']['tool_use']['passed'] += 1
                
                if level:
                    stats[level]['tool_use']['total'] += 1
                    if cp_data.get('passed', False):
                        stats[level]['tool_use']['passed'] += 1
            
            # 统计 visual-check 类 checkpoint
            elif is_visual_check_checkpoint_from_result(cp_data):
                stats['Overall']['visual_check']['total'] += 1
                if cp_data.get('passed', False):
                    stats['Overall']['visual_check']['passed'] += 1
                
                if level:
                    stats[level]['visual_check']['total'] += 1
                    if cp_data.get('passed', False):
                        stats[level]['visual_check']['passed'] += 1
    
    return stats, missing_evaluations

def print_stats(stats, missing_evaluations):
    """打印统计结果"""
    print("="*70)
    print("V-AXIS CHECKPOINT ANALYSIS (Enhanced)")
    print("="*70)
    
    if missing_evaluations:
        print()
        print(f"⚠️  Missing Evaluations: {len(missing_evaluations)} runs")
        print(f"   (These runs are counted as all checkpoints failed)")
        if len(missing_evaluations) <= 10:
            print(f"   Run IDs: {', '.join(missing_evaluations)}")
        else:
            print(f"   Run IDs: {', '.join(missing_evaluations[:10])}... and {len(missing_evaluations)-10} more")
    
    print()
    
    for level_name in ['Overall', 'L1', 'L2', 'L3']:
        level_stats = stats[level_name]
        
        print(f"--- {level_name} ---")
        print()
        
        # Tool-use 类统计
        tool_use = level_stats['tool_use']
        tool_use_total = tool_use['total']
        tool_use_passed = tool_use['passed']
        tool_use_rate = (tool_use_passed / tool_use_total * 100) if tool_use_total > 0 else 0.0
        
        print(f"Tool-Use Checkpoints:")
        print(f"  Total: {tool_use_total}")
        print(f"  Passed: {tool_use_passed}")
        print(f"  Pass Rate: {tool_use_rate:.2f}%")
        print()
        
        # Visual-check 类统计
        visual_check = level_stats['visual_check']
        visual_check_total = visual_check['total']
        visual_check_passed = visual_check['passed']
        visual_check_rate = (visual_check_passed / visual_check_total * 100) if visual_check_total > 0 else 0.0
        
        print(f"Visual-Check Checkpoints:")
        print(f"  Total: {visual_check_total}")
        print(f"  Passed: {visual_check_passed}")
        print(f"  Pass Rate: {visual_check_rate:.2f}%")
        print()
        
        # 总计
        total_checkpoints = tool_use_total + visual_check_total
        total_passed = tool_use_passed + visual_check_passed
        total_rate = (total_passed / total_checkpoints * 100) if total_checkpoints > 0 else 0.0
        
        print(f"Total V-Axis Checkpoints:")
        print(f"  Total: {total_checkpoints}")
        print(f"  Passed: {total_passed}")
        print(f"  Pass Rate: {total_rate:.2f}%")
        print()
        print("-"*70)
        print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_v_checkpoints_v2.py <scored_json_path> [task_config_dir]")
        print()
        print("Arguments:")
        print("  scored_json_path: Path to the scored JSON file (e.g., runs/scores/general_deepeyes-rl_scored.json)")
        print("  task_config_dir:  (Optional) Directory containing task config JSON files (e.g., test_examples/QnA)")
        print()
        print("Examples:")
        print("  # Basic usage (without handling missing evaluations)")
        print("  python analyze_v_checkpoints_v2.py runs/scores/general_deepeyes-rl_scored.json")
        print()
        print("  # Enhanced usage (with task config directory to handle missing evaluations)")
        print("  python analyze_v_checkpoints_v2.py runs/scores/general_deepeyes-rl_scored.json test_examples/QnA")
        sys.exit(1)
    
    scored_json_path = sys.argv[1]
    task_config_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(scored_json_path).exists():
        print(f"Error: File not found: {scored_json_path}")
        sys.exit(1)
    
    if task_config_dir and not Path(task_config_dir).exists():
        print(f"Warning: Task config directory not found: {task_config_dir}")
        print(f"Proceeding without handling missing evaluations...")
        task_config_dir = None
    
    print(f"Analyzing: {scored_json_path}")
    if task_config_dir:
        print(f"Task Config Dir: {task_config_dir}")
    print()
    
    stats, missing_evaluations = analyze_checkpoints(scored_json_path, task_config_dir)
    print_stats(stats, missing_evaluations)

if __name__ == "__main__":
    main()
