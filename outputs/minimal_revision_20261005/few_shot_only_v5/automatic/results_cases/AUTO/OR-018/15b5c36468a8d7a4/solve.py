import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not read CSV with supported encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    baby_mask = df['Product Name'].str.casefold().str.contains('baby')
    baby_df = df[baby_mask].copy()
    if baby_df.empty:
        raise ValueError("No 'Baby' products found in the data.")
    items = list(baby_df['Product Name'])
    revenue = dict(zip(baby_df['Product Name'], baby_df['Revenue']))
    inventory = dict(zip(baby_df['Product Name'], baby_df['Initial Inventory']))
    demand = dict(zip(baby_df['Product Name'], baby_df['Demand']))
    for i in items:
        if pd.isnull(revenue[i]) or pd.isnull(inventory[i]) or pd.isnull(demand[i]):
            raise ValueError(f'Missing data for product: {i}')
        try:
            revenue[i] = float(revenue[i])
            inventory[i] = int(inventory[i])
            demand[i] = int(demand[i])
        except Exception:
            raise ValueError(f'Non-numeric data for product: {i}')
    m = gp.Model('Supermarket_Baby_Revenue')
    m.Params.MIPGap = 0.0001
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()