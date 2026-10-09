import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not read CSV with supported encodings.')
    baby_mask = df['Product Name'].str.casefold().str.contains('baby')
    baby_df = df[baby_mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in baby_df.columns:
            raise ValueError(f'Missing required column: {col}')
    for col in ['Revenue', 'Demand', 'Initial Inventory']:
        baby_df[col] = pd.to_numeric(baby_df[col], errors='raise')
    baby_df = baby_df.groupby('Product Name', as_index=False).agg({'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = list(baby_df['Product Name'])
    revenue = dict(zip(baby_df['Product Name'], baby_df['Revenue']))
    demand = dict(zip(baby_df['Product Name'], baby_df['Demand']))
    inventory = dict(zip(baby_df['Product Name'], baby_df['Initial Inventory']))
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing data for product: {i}')
    m = gp.Model('Europe_Baby_Product_Revenue_Max')
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
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')