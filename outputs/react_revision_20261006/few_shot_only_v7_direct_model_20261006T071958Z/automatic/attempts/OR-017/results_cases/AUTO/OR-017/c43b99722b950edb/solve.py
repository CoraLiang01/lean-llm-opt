import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode CSV at {csv_path} with tried encodings.')
    if 'SKU' not in df.columns:
        raise ValueError("Missing required column 'SKU' in source data.")
    zz_mask = df['SKU'].str.casefold().str.startswith('zz')
    zz_df = df.loc[zz_mask].copy()
    required_cols = ['SKU', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in zz_df.columns:
            raise ValueError(f"Missing required column '{col}' in filtered data.")
    zz_df = zz_df.reset_index(drop=True)
    zz_keys = list(zz_df.index)
    try:
        revenue = zz_df['Revenue'].astype(float).to_dict()
        demand = zz_df['Demand'].astype(float).to_dict()
        inventory = zz_df['Initial Inventory'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting numeric columns: {e}')
    for k in zz_keys:
        if k not in revenue or k not in demand or k not in inventory:
            raise ValueError(f'Missing parameter for row {k}')
    ub = {k: min(demand[k], inventory[k]) for k in zz_keys}
    lb = {k: 0 for k in zz_keys}
    m = gp.Model('Retail_ZZ_Revenue_Max')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(zz_keys, lb=[lb[k] for k in zz_keys], ub=[ub[k] for k in zz_keys], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[k] * quantity_vars[k] for k in zz_keys)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[k] <= demand[k] for k in zz_keys), name='')
    m.addConstrs((quantity_vars[k] <= inventory[k] for k in zz_keys), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')