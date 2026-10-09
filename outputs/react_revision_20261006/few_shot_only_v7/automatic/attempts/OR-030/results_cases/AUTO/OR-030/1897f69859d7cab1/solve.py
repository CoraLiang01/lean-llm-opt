import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError('Failed to read CSV with supported encodings.')
    if 'Product Name' not in df.columns:
        raise ValueError("Missing 'Product Name' column in source data.")
    mask = df['Product Name'].str.casefold().str.contains('fdk57')
    df_fdk57 = df[mask].copy()
    if df_fdk57.empty:
        raise ValueError("No 'FDK57' car models found in source data.")
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_fdk57.columns:
            raise ValueError(f'Missing required column: {col}')
    items = df_fdk57['Product Name'].tolist()
    df_fdk57_grouped = df_fdk57.groupby('Product Name', as_index=False).agg({'Revenue': lambda x: list(x), 'Demand': lambda x: list(x), 'Initial Inventory': lambda x: list(x)})
    revenue = {}
    demand = {}
    inventory = {}
    for (idx, row) in df_fdk57_grouped.iterrows():
        key = row['Product Name']
        try:
            rev = sum((float(v) for v in row['Revenue'] if v.strip() != ''))
            dem = sum((float(v) for v in row['Demand'] if v.strip() != ''))
            inv = sum((float(v) for v in row['Initial Inventory'] if v.strip() != ''))
        except Exception:
            raise ValueError(f"Non-numeric value in coefficients for '{key}'.")
        revenue[key] = rev
        demand[key] = dem
        inventory[key] = inv
    for key in items:
        if key not in revenue or key not in demand or key not in inventory:
            raise ValueError(f"Missing coefficients for '{key}'.")
    m = gp.Model('BigMart_FDK57')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')