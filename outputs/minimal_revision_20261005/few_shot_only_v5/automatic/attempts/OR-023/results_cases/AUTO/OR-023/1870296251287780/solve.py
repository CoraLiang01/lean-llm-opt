import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not read CSV with supported encodings.')
    if 'Product_Reference' not in df.columns:
        raise ValueError('Missing required column: Product_Reference')
    ele_s_mask = df['Product_Reference'].astype(str).str.casefold().str.contains('ele-s')
    df_ele_s = df[ele_s_mask].copy()
    if df_ele_s.empty:
        raise ValueError("No 'ELE-S' products found in the data.")
    required_cols = ['Product_Reference', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_ele_s.columns:
            raise ValueError(f'Missing required column: {col}')
    items = df_ele_s['Product_Reference'].astype(str).tolist()
    revenue = {}
    inventory = {}
    demand = {}
    for (_, row) in df_ele_s.iterrows():
        key = str(row['Product_Reference'])
        try:
            a_i = float(row['Revenue'])
            s_i = float(row['Initial Inventory'])
            d_i = float(row['Demand'])
        except Exception:
            raise ValueError(f'Non-numeric data in required columns for product {key}')
        revenue[key] = a_i
        inventory[key] = s_i
        demand[key] = d_i
    for key in items:
        if key not in revenue or key not in inventory or key not in demand:
            raise ValueError(f'Missing coefficients for product {key}')
    m = gp.Model('Supermarket_ELE_S_Revenue')
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()