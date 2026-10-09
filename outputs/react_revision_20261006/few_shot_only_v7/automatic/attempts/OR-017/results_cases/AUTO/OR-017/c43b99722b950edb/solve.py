import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError('Failed to read CSV with supported encodings.')
    zz_mask = df['SKU'].str.casefold().str.contains('zz')
    zz_df = df[zz_mask].copy()
    required_cols = ['SKU', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in zz_df.columns:
            raise ValueError(f'Missing required column: {col}')
    zz_df['Revenue'] = pd.to_numeric(zz_df['Revenue'], errors='raise')
    zz_df['Demand'] = pd.to_numeric(zz_df['Demand'], errors='raise')
    zz_df['Initial Inventory'] = pd.to_numeric(zz_df['Initial Inventory'], errors='raise')
    zz_grouped = zz_df.groupby('SKU', as_index=False).agg({'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = zz_grouped['SKU'].tolist()
    revenue = dict(zip(zz_grouped['SKU'], zz_grouped['Revenue']))
    demand = dict(zip(zz_grouped['SKU'], zz_grouped['Demand']))
    inventory = dict(zip(zz_grouped['SKU'], zz_grouped['Initial Inventory']))
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing data for SKU: {i}')
    m = gp.Model('RetailStore_ZZ_Optimization')
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()