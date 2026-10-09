import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError('Failed to read CSV with supported encodings.')
    if 'Product Name' not in df.columns:
        raise ValueError("Missing 'Product Name' column in source data.")
    aalop_mask = df['Product Name'].str.casefold() == 'aalop'
    df_aalop = df[aalop_mask].copy()
    if df_aalop.empty:
        raise ValueError("No products classified under 'Aalop' found in source data.")
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_aalop.columns:
            raise ValueError(f"Missing required column '{col}' in source data.")
    df_aalop['key'] = df_aalop[required_cols].apply(lambda row: tuple(row), axis=1)
    keys = df_aalop['key'].tolist()

    def to_float_or_int(val):
        try:
            if '.' in val:
                return float(val)
            return int(val)
        except Exception:
            raise ValueError(f'Non-numeric value encountered: {val}')
    revenue = {}
    demand = {}
    inventory = {}
    for (_, row) in df_aalop.iterrows():
        k = row['key']
        try:
            revenue[k] = to_float_or_int(row['Revenue'])
            demand[k] = to_float_or_int(row['Demand'])
            inventory[k] = to_float_or_int(row['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Error parsing numeric fields for key {k}: {e}')
    for k in keys:
        if k not in revenue or k not in demand or k not in inventory:
            raise ValueError(f'Missing parameter(s) for key {k}')
    m = gp.Model('Aalop_Revenue_Max')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[k] * quantity_vars[k] for k in keys)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[k] <= inventory[k] for k in keys), name='')
    m.addConstrs((quantity_vars[k] <= demand[k] for k in keys), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')