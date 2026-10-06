import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError('Failed to read CSV with supported encodings.')
    required_cols = ['SKU', 'Revenue', 'Demand', 'Initial Inventory', 'Product Category']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing required columns in CSV: {missing_cols}')
    mask = df['Product Category'].astype(str).str.casefold() == 'zz'
    zz_df = df[mask].copy()
    if zz_df.empty:
        raise ValueError("No products found in category 'ZZ'.")
    I = zz_df['SKU'].astype(str).tolist()
    grouped = zz_df.groupby('SKU', as_index=False).agg({'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    a = dict(zip(grouped['SKU'].astype(str), grouped['Revenue']))
    d = dict(zip(grouped['SKU'].astype(str), grouped['Demand']))
    s = dict(zip(grouped['SKU'].astype(str), grouped['Initial Inventory']))
    for i in grouped['SKU'].astype(str):
        if i not in a or i not in d or i not in s:
            raise ValueError(f'Missing data for SKU {i}.')
        if pd.isnull(a[i]) or pd.isnull(d[i]) or pd.isnull(s[i]):
            raise ValueError(f'Null value for SKU {i}.')
        if not (isinstance(a[i], (int, float)) and isinstance(d[i], (int, float)) and isinstance(s[i], (int, float))):
            raise ValueError(f'Non-numeric value for SKU {i}.')
    for i in grouped['SKU'].astype(str):
        if d[i] < 0 or s[i] < 0:
            raise ValueError(f'Negative demand or inventory for SKU {i}.')
    m = gp.Model('RetailStore_ZZ_Revenue_Max')
    m.Params.MIPGap = 0.0001
    x = m.addVars(grouped['SKU'].astype(str), lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((a[i] * x[i] for i in grouped['SKU'].astype(str))), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= d[i] for i in grouped['SKU'].astype(str)), name='')
    m.addConstrs((x[i] <= s[i] for i in grouped['SKU'].astype(str)), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()