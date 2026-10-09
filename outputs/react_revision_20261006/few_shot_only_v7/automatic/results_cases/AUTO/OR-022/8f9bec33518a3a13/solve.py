import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with supported encodings.')
    if 'Product Name' not in df.columns:
        raise ValueError("Missing 'Product Name' column in source data.")
    mask_27in = df['Product Name'].str.casefold().str.contains('27in')
    df_27in = df[mask_27in].copy()
    if df_27in.empty:
        raise ValueError("No products with '27in' in 'Product Name' found.")
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_27in.columns:
            raise ValueError(f'Missing required column: {col}')
    for col in ['Revenue', 'Demand', 'Initial Inventory']:
        df_27in[col] = pd.to_numeric(df_27in[col], errors='raise')
    df_agg = df_27in.groupby('Product Name', as_index=False).agg({'Revenue': 'first', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = df_agg['Product Name'].tolist()
    revenue = dict(zip(df_agg['Product Name'], df_agg['Revenue']))
    demand = dict(zip(df_agg['Product Name'], df_agg['Demand']))
    inventory = dict(zip(df_agg['Product Name'], df_agg['Initial Inventory']))
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing coefficients for product: {i}')
    m = gp.Model('NRM_27in')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()