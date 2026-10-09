import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError('Could not read CSV with supported encodings.')
    mask_27in = df['Product Name'].str.casefold().str.contains('27in')
    df_27in = df[mask_27in].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_27in.columns:
            raise ValueError(f'Missing required column: {col}')
    df_27in['Revenue'] = pd.to_numeric(df_27in['Revenue'], errors='raise')
    df_27in['Demand'] = pd.to_numeric(df_27in['Demand'], errors='raise')
    df_27in['Initial Inventory'] = pd.to_numeric(df_27in['Initial Inventory'], errors='raise')
    df_agg = df_27in.groupby('Product Name', as_index=False).agg({'Revenue': 'first', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = df_agg['Product Name'].tolist()
    revenue = dict(zip(df_agg['Product Name'], df_agg['Revenue']))
    demand = dict(zip(df_agg['Product Name'], df_agg['Demand']))
    inventory = dict(zip(df_agg['Product Name'], df_agg['Initial Inventory']))
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing data for product: {i}')
        if not (isinstance(revenue[i], (int, float)) and isinstance(demand[i], (int, float)) and isinstance(inventory[i], (int, float))):
            raise ValueError(f'Non-numeric data for product: {i}')
    m = gp.Model('Supermarket_27in_Revenue')
    m.Params.MIPGap = 0.0001
    ub_dict = {i: min(demand[i], inventory[i]) for i in items}
    quantity_vars = m.addVars(items, lb=0, ub=ub_dict, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')