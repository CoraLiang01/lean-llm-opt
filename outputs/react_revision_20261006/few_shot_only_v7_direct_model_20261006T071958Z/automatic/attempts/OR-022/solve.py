import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read CSV at {csv_path} with tried encodings.')
    mask = df['Product Name'].str.casefold().str.contains('27in')
    df_27in = df[mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_27in.columns:
            raise ValueError(f'Missing required column: {col}')
    items = df_27in['Product Name'].tolist()
    if len(items) == 0:
        raise ValueError("No products with '27in' found in 'Product Name'.")
    df_27in['Revenue'] = pd.to_numeric(df_27in['Revenue'], errors='raise')
    df_27in['Demand'] = pd.to_numeric(df_27in['Demand'], errors='raise')
    df_27in['Initial Inventory'] = pd.to_numeric(df_27in['Initial Inventory'], errors='raise')
    grouped = df_27in.groupby('Product Name', sort=False, as_index=False).agg({'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = grouped['Product Name'].tolist()
    revenue = dict(zip(items, grouped['Revenue']))
    demand = dict(zip(items, grouped['Demand']))
    inventory = dict(zip(items, grouped['Initial Inventory']))
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing parameter for product: {i}')
    m = gp.Model('NRM_27in')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()