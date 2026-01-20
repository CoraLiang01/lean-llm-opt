# -*- coding: utf-8 -*-
"""
Script: Calculate model accuracy by token range

Usage:
    python analyze_accuracy_by_token.py [mode]
    
Arguments:
    mode: 'model' or 'code' (default: 'code')
          'model' - Use Final Count (Model) column for modeling accuracy
          'code'  - Use Final Count (Obj/Code) column for code accuracy

Examples:
    python analyze_accuracy_by_token.py model   # Calculate modeling accuracy
    python analyze_accuracy_by_token.py code    # Calculate code accuracy
    python analyze_accuracy_by_token.py         # Default: code accuracy
"""

import pandas as pd
import sys
import os

# ============================================================================
# Configuration - Modify as needed
# ============================================================================

# Excel file path
EXCEL_PATH = r'd:\Onedrive\CityU_Research\Lean_OPT_data_share\Large-scale-or-Results-0120-ALL.xlsx'

# Output file path
OUTPUT_PATH = r'd:\Onedrive\CityU_Research\Lean_OPT_data_share\model_accuracy_by_token.txt'

# Token column index (0-based)
TOKEN_COL = 2

# Model configuration
# Format: 'Model Name': {'model': Final Count (Model) col index, 'code': Final Count (Obj/Code) col index}
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

# Token range categories
TOKEN_RANGES = [
    ('0-400', 0, 400),
    ('401-800', 401, 800),
    ('801+', 801, float('inf'))
]

# ============================================================================
# Main Program
# ============================================================================

def categorize_token(t):
    """Categorize token count into ranges"""
    if pd.isna(t):
        return None
    for name, low, high in TOKEN_RANGES:
        if low <= t <= high:
            return name
    return None

def analyze_accuracy(excel_path, mode='code'):
    """
    Analyze model accuracy by token range
    
    Args:
        excel_path: Path to Excel file
        mode: 'model' (modeling accuracy) or 'code' (code accuracy)
    
    Returns:
        DataFrame: Model accuracy for each token range
    """
    # Read data
    df_raw = pd.read_excel(excel_path, header=None)
    df_data = df_raw.iloc[2:].copy()
    df_data.columns = range(len(df_data.columns))
    
    # Token categorization
    tokens = pd.to_numeric(df_data[TOKEN_COL], errors='coerce')
    df_data['token_category'] = tokens.apply(categorize_token)
    df_valid = df_data[df_data['token_category'].notna()].copy()
    
    # Calculate accuracy for each model
    results = []
    
    for model_name, cols in MODELS_CONFIG.items():
        col_idx = cols[mode]
        # Get data, treat missing values as 0
        model_data = pd.to_numeric(df_valid[col_idx], errors='coerce').fillna(0)
        
        for cat_name, _, _ in TOKEN_RANGES:
            mask = df_valid['token_category'] == cat_name
            cat_data = model_data[mask]
            total = len(cat_data)
            correct = (cat_data == 1).sum()
            accuracy = correct / total * 100 if total > 0 else 0
            
            results.append({
                'Model': model_name,
                'Token_Range': cat_name,
                'Correct': correct,
                'Total': total,
                'Accuracy': accuracy
            })
    
    return pd.DataFrame(results)

def generate_report(df_results, mode):
    """Generate report text"""
    # Create pivot table
    pivot = df_results.pivot(index='Model', columns='Token_Range', values='Accuracy')
    pivot = pivot[[name for name, _, _ in TOKEN_RANGES]]
    
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
    
    # Header row
    header = f"{'Model':<35}"
    for name, _, _ in TOKEN_RANGES:
        header += f" {name:>12}"
    header += f" {'Overall':>12}"
    lines.append(header)
    lines.append('-' * 85)
    
    # Data rows
    for model in pivot.index:
        row = pivot.loc[model]
        line = f"{model:<35}"
        for name, _, _ in TOKEN_RANGES:
            line += f" {row[name]:>11.1f}%"
        line += f" {row['Overall']:>11.1f}%"
        lines.append(line)
    
    # Sample counts
    sample_counts = []
    for name, _, _ in TOKEN_RANGES:
        count = df_results[df_results['Token_Range'] == name]['Total'].iloc[0]
        sample_counts.append(f"{name} ({count})")
    lines.append('')
    lines.append(f"Sample count: {', '.join(sample_counts)}")
    
    return '\n'.join(lines)

def main():
    # Set encoding
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    
    # Parse command line arguments
    mode = 'code'  # Default mode
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['model', 'code']:
            mode = arg
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python analyze_accuracy_by_token.py [model|code]")
            sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(EXCEL_PATH):
        print(f"File not found: {EXCEL_PATH}")
        sys.exit(1)
    
    # Mode description
    mode_desc = "Modeling Accuracy - Final Count (Model)" if mode == 'model' else "Code Accuracy - Final Count (Obj/Code)"
    
    print("=" * 85)
    print(f"Mode: {mode_desc}")
    print(f"Data file: {EXCEL_PATH}")
    print("=" * 85)
    
    # Analyze data
    df_results = analyze_accuracy(EXCEL_PATH, mode)
    
    # Generate report
    report = generate_report(df_results, mode)
    
    # Output to console
    print(f"\n{report}")
    
    # Save to file
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nResults saved to: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
