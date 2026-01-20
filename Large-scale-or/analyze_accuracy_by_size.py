# -*- coding: utf-8 -*-
"""
Script: Calculate model accuracy by size (L/M/S)

Usage:
    python analyze_accuracy_by_size.py [mode]
    
Arguments:
    mode: 'model' or 'code' (default: 'model')
          'model' - Use Final Count (Model) column for modeling accuracy
          'code'  - Use Final Count (Obj/Code) column for code accuracy
"""

import pandas as pd
import sys
import os

# ============================================================================
# Configuration
# ============================================================================

EXCEL_PATH = r'd:\Onedrive\CityU_Research\Lean_OPT_data_share\Large-scale-or-Results-0120-ALL.xlsx'
OUTPUT_PATH = r'd:\Onedrive\CityU_Research\Lean_OPT_data_share\model_accuracy_by_size.txt'

# Size column index (0-based) - "Type by size" column
SIZE_COL = 1

# Model configuration
MODELS_CONFIG = {
    'LEAN (GPT-4.1)':              {'model': 9,  'code': 10},
    'LEAN (gpt-oss-20b)':          {'model': 26, 'code': 27},
    'LEAN (GPT-4.1-Agnostic)':     {'model': 53, 'code': 54},
    'LEAN (gpt-oss-20b-Agnostic)': {'model': 59, 'code': 60},
    'Gemini3pro':                  {'model': 47, 'code': 48},
    'GPT-5.2':                     {'model': 41, 'code': 42},
    'ORLM':                        {'model': 14, 'code': 15},
    'GPT-5':                       {'model': 23, 'code': 22},
    'gpt-oss-20b':                 {'model': 31, 'code': 32},
    'GPT-4.1':                     {'model': 18, 'code': 19},
    'OptiMUS':                     {'model': 36, 'code': 37},
}

# Size categories
SIZE_CATEGORIES = ['L', 'M', 'S']

# ============================================================================
# Main Program
# ============================================================================

def analyze_accuracy(excel_path, mode='model'):
    """Analyze model accuracy by size"""
    df_raw = pd.read_excel(excel_path, header=None)
    df_data = df_raw.iloc[2:].copy()
    df_data.columns = range(len(df_data.columns))
    
    # Get size column
    sizes = df_data[SIZE_COL].astype(str).str.strip().str.upper()
    df_data['size_category'] = sizes
    df_valid = df_data[df_data['size_category'].isin(SIZE_CATEGORIES)].copy()
    
    # Calculate accuracy for each model
    results = []
    
    for model_name, cols in MODELS_CONFIG.items():
        col_idx = cols[mode]
        model_data = pd.to_numeric(df_valid[col_idx], errors='coerce').fillna(0)
        
        for size in SIZE_CATEGORIES:
            mask = df_valid['size_category'] == size
            cat_data = model_data[mask]
            total = len(cat_data)
            correct = (cat_data == 1).sum()
            accuracy = correct / total * 100 if total > 0 else 0
            
            results.append({
                'Model': model_name,
                'Size': size,
                'Correct': correct,
                'Total': total,
                'Accuracy': accuracy
            })
    
    return pd.DataFrame(results)

def generate_report(df_results, mode):
    """Generate report text"""
    pivot = df_results.pivot(index='Model', columns='Size', values='Accuracy')
    pivot = pivot[SIZE_CATEGORIES]
    
    # Calculate Overall
    overall = []
    for model in pivot.index:
        model_results = df_results[df_results['Model'] == model]
        total_correct = model_results['Correct'].sum()
        total_count = model_results['Total'].sum()
        overall.append(total_correct / total_count * 100 if total_count > 0 else 0)
    
    pivot['Overall'] = overall
    pivot = pivot.sort_values('Overall', ascending=False)
    
    # Generate report
    lines = []
    
    # Header
    header = f"{'Model':<35}"
    for size in SIZE_CATEGORIES:
        header += f" {size:>12}"
    header += f" {'Overall':>12}"
    lines.append(header)
    lines.append('-' * 85)
    
    # Data rows
    for model in pivot.index:
        row = pivot.loc[model]
        line = f"{model:<35}"
        for size in SIZE_CATEGORIES:
            line += f" {row[size]:>11.1f}%"
        line += f" {row['Overall']:>11.1f}%"
        lines.append(line)
    
    # Sample counts
    sample_counts = []
    for size in SIZE_CATEGORIES:
        count = df_results[df_results['Size'] == size]['Total'].iloc[0]
        sample_counts.append(f"{size} ({count})")
    lines.append('')
    lines.append(f"Sample count: {', '.join(sample_counts)}")
    
    return '\n'.join(lines)

def main():
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    
    mode = 'model'
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['model', 'code']:
            mode = arg
    
    if not os.path.exists(EXCEL_PATH):
        print(f"File not found: {EXCEL_PATH}")
        sys.exit(1)
    
    mode_desc = "Modeling Accuracy - Final Count (Model)" if mode == 'model' else "Code Accuracy - Final Count (Obj/Code)"
    
    print("=" * 85)
    print(f"Mode: {mode_desc}")
    print(f"Data file: {EXCEL_PATH}")
    print("=" * 85)
    
    df_results = analyze_accuracy(EXCEL_PATH, mode)
    report = generate_report(df_results, mode)
    
    print(f"\n{report}")
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nResults saved to: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
