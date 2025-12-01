#!/usr/bin/env python3
"""
Grid Search Result Analyzer
Select best parameters for each language (excluding baseline) and copy corresponding json files to new folder
"""

import os
import pandas as pd
import numpy as np
import fire
from typing import Dict, Tuple, List
import json
import shutil


def parse_strength_from_folder(folder_name: str) -> Tuple[float, float]:
    """
    Parse strength parameters from folder name.
    Example: strength_0.1_-0.1 -> (0.1, -0.1)
    """
    if not folder_name.startswith('strength_'):
        return None, None
    
    parts = folder_name.replace('strength_', '').split('_')
    if len(parts) != 2:
        return None, None
    
    try:
        strength1 = float(parts[0])
        strength2 = float(parts[1])
        return strength1, strength2
    except ValueError:
        return None, None


def read_results_from_folder(folder_path: str) -> Dict:
    """
    Read Results.csv file from a single strength folder.
    """
    results_file = os.path.join(folder_path, 'Results.csv')
    if not os.path.exists(results_file):
        return None
    
    try:
        df = pd.read_csv(results_file)
        
        # Check required columns
        required_cols = ['subtask', 'score', 'rep_fidelity', 'rea_fidelity']
        if not all(col in df.columns for col in required_cols):
            return None
        
        # Convert to dictionary format for easier processing
        results = {}
        for _, row in df.iterrows():
            subtask = row['subtask']
            results[subtask] = {
                'score': row['score'],
                'rep_fidelity': row['rep_fidelity'],
                'rea_fidelity': row['rea_fidelity']
            }
        
        return results
    
    except Exception as e:
        return None


def analyze_grid_search(input_folder: str) -> Dict:
    """
    Analyze grid search results.
    Select best parameters for each language, then calculate average metrics when using respective best parameters.
    """
    all_results = {}
    strength_folders = []
    
    for item in os.listdir(input_folder):
        item_path = os.path.join(input_folder, item)
        if os.path.isdir(item_path) and item.startswith('strength_'):
            strength1, strength2 = parse_strength_from_folder(item)
            if strength1 is not None and strength2 is not None:
                results = read_results_from_folder(item_path)
                if results is not None:
                    all_results[(strength1, strength2)] = results
                    strength_folders.append(item)
    
    if not all_results:
        return None
    
    # Get all languages (exclude FINAL)
    all_languages = set()
    for results in all_results.values():
        all_languages.update(results.keys())
    all_languages.discard('FINAL')  # Remove FINAL
    all_languages = sorted(list(all_languages))
    
    
    # Get baseline data (both steering strengths are 0)
    baseline_data = {}
    if (0, 0) in all_results:
        baseline_results = all_results[(0, 0)]
        for lang in all_languages:
            if lang in baseline_results:
                baseline_data[lang] = {
                    'score': baseline_results[lang]['score'],
                    'rep_fidelity': baseline_results[lang]['rep_fidelity'],
                    'rea_fidelity': baseline_results[lang]['rea_fidelity']
                }
        pass
    else:
        return None
    
    analysis_results = {
        'baseline_data': baseline_data,
        'best_params_per_language': {},
        'average_with_best_params': {},
        'comparison_with_unified_best': {},
        'summary_table': [],
        'all_languages': all_languages,
        'all_results': all_results
    }
    
    # 1. Find best parameter combination for each language (based on formula: 2*score_improvement + rep_improvement)
    for lang in all_languages:
        if lang not in baseline_data:
            continue
            
        baseline_score = baseline_data[lang]['score']
        baseline_rep_fidelity = baseline_data[lang]['rep_fidelity']
        
        best_improvement_score = -float('inf')
        best_params = None
        best_details = None
        best_score_improvement = 0
        best_rep_improvement = 0
        
        for (s1, s2), results in all_results.items():
            # Exclude baseline case (both steering strengths are 0)
            if s1 == 0 and s2 == 0:
                continue
                
            if lang in results:
                current_score = results[lang]['score']
                current_rep_fidelity = results[lang]['rep_fidelity']
                
                # Calculate improvements
                score_improvement = current_score - baseline_score
                rep_improvement = current_rep_fidelity - baseline_rep_fidelity
                
                # Apply formula: 2*score_improvement + rep_improvement
                improvement_score = 2 * score_improvement + rep_improvement
                
                if improvement_score > best_improvement_score:
                    best_improvement_score = improvement_score
                    best_params = (s1, s2)
                    best_details = results[lang]
                    best_score_improvement = score_improvement
                    best_rep_improvement = rep_improvement
        
        if best_params is not None:
            analysis_results['best_params_per_language'][lang] = {
                'strength_params': best_params,
                'score': best_details['score'],
                'rep_fidelity': best_details['rep_fidelity'],
                'rea_fidelity': best_details['rea_fidelity'],
                'baseline_score': baseline_score,
                'baseline_rep_fidelity': baseline_rep_fidelity,
                'score_improvement': best_score_improvement,
                'rep_improvement': best_rep_improvement,
                'improvement_score': best_improvement_score
            }
    
    if analysis_results['best_params_per_language']:
        total_score = 0
        total_rep_fidelity = 0
        total_rea_fidelity = 0
        valid_languages = 0
        
        for lang, info in analysis_results['best_params_per_language'].items():
            total_score += info['score']
            total_rep_fidelity += info['rep_fidelity']
            total_rea_fidelity += info['rea_fidelity']
            valid_languages += 1
        
        if valid_languages > 0:
            analysis_results['average_with_best_params'] = {
                'avg_score': total_score / valid_languages,
                'avg_rep_fidelity': total_rep_fidelity / valid_languages,
                'avg_rea_fidelity': total_rea_fidelity / valid_languages,
                'num_languages': valid_languages
            }
    
    best_unified_score = -1
    best_unified_params = None
    best_unified_details = None
    
    for (s1, s2), results in all_results.items():
        if s1 == 0 and s2 == 0:
            continue
            
        if 'FINAL' in results:
            final_score = results['FINAL']['score']
            if final_score > best_unified_score:
                best_unified_score = final_score
                best_unified_params = (s1, s2)
                best_unified_details = results['FINAL']
    
    if best_unified_params is not None:
        analysis_results['comparison_with_unified_best'] = {
            'unified_best_params': best_unified_params,
            'unified_final_score': best_unified_score,
            'unified_rep_fidelity': best_unified_details['rep_fidelity'],
            'unified_rea_fidelity': best_unified_details['rea_fidelity']
        }
    
    for (s1, s2), results in all_results.items():
        if 'FINAL' in results:
            row = {
                'strength_1': s1,
                'strength_2': s2,
                'final_score': results['FINAL']['score'],
                'final_rep_fidelity': results['FINAL']['rep_fidelity'],
                'final_rea_fidelity': results['FINAL']['rea_fidelity']
            }
            
            for lang in all_languages:
                if lang in results:
                    row[f'{lang}_score'] = results[lang]['score']
                    row[f'{lang}_rep_fidelity'] = results[lang]['rep_fidelity']
                    row[f'{lang}_rea_fidelity'] = results[lang]['rea_fidelity']
                else:
                    row[f'{lang}_score'] = None
                    row[f'{lang}_rep_fidelity'] = None
                    row[f'{lang}_rea_fidelity'] = None
            
            analysis_results['summary_table'].append(row)
    
    analysis_results['summary_table'].sort(key=lambda x: x['final_score'], reverse=True)
    
    return analysis_results


def print_analysis_results(results: Dict):
    """
    Print analysis results (based on improvement formula vs baseline)
    """
    pass


def copy_best_json_files(results: Dict, input_folder: str, output_folder: str):
    """
    Copy json files corresponding to best parameters for each language to new folder
    """
    best_json_folder = os.path.join(output_folder, 'best_json_files')
    os.makedirs(best_json_folder, exist_ok=True)
    
    
    copied_files = []
    for lang, info in results['best_params_per_language'].items():
        s1, s2 = info['strength_params']
        
        strength_folder = f"strength_{s1}_{s2}"
        source_folder = os.path.join(input_folder, strength_folder)
        
        json_filename = f"{lang}.json"
        source_json = os.path.join(source_folder, json_filename)
        
        if os.path.exists(source_json):
            target_filename = f"{lang}_strength_{s1}_{s2}.json"
            target_json = os.path.join(best_json_folder, target_filename)
            
            try:
                shutil.copy2(source_json, target_json)
                copied_files.append(target_filename)
            except Exception as e:
                pass
    return copied_files


def save_analysis_results(results: Dict, output_folder: str):
    """
    保存简化的分析结果到文件
    """
    os.makedirs(output_folder, exist_ok=True)
    
    if results['best_params_per_language']:
        best_params_data = []
        for lang, info in results['best_params_per_language'].items():
            s1, s2 = info['strength_params']
            best_params_data.append({
                'language': lang,
                'strength_1': s1,
                'strength_2': s2,
                'score': info['score'],
                'rep_fidelity': info['rep_fidelity'],
                'rea_fidelity': info['rea_fidelity'],
                'baseline_score': info['baseline_score'],
                'baseline_rep_fidelity': info['baseline_rep_fidelity']
            })
        
        if best_params_data:
            avg_score = sum(row['score'] for row in best_params_data) / len(best_params_data)
            avg_rep_fidelity = sum(row['rep_fidelity'] for row in best_params_data) / len(best_params_data)
            avg_rea_fidelity = sum(row['rea_fidelity'] for row in best_params_data) / len(best_params_data)
            avg_baseline_score = sum(row['baseline_score'] for row in best_params_data) / len(best_params_data)
            avg_baseline_rep_fidelity = sum(row['baseline_rep_fidelity'] for row in best_params_data) / len(best_params_data)
            
            best_params_data.append({
                'language': 'AVERAGE',
                'strength_1': 'mixed',
                'strength_2': 'mixed',
                'score': round(avg_score, 4),
                'rep_fidelity': round(avg_rep_fidelity, 4),
                'rea_fidelity': round(avg_rea_fidelity, 4),
                'baseline_score': round(avg_baseline_score, 4),
                'baseline_rep_fidelity': round(avg_baseline_rep_fidelity, 4)
            })
        
        df_best_params = pd.DataFrame(best_params_data)
        search_result_csv = os.path.join(output_folder, 'search_result.csv')
        df_best_params.to_csv(search_result_csv, index=False)


def main(input_folder: str, output_folder: str = None):
    """
    主函数
    
    Args:
        input_folder: 包含metric_summary.xlsx和各个strength文件夹的输入目录
        output_folder: 输出目录，如果不指定则使用输入目录
    """
    if not os.path.exists(input_folder):
        return
    
    if output_folder is None:
        output_folder = input_folder
    
    results = analyze_grid_search(input_folder)
    
    if results is None:
        return
    
    print_analysis_results(results)
    
    copied_files = copy_best_json_files(results, input_folder, output_folder)
    
    save_analysis_results(results, output_folder)
    


if __name__ == "__main__":
    fire.Fire(main)
