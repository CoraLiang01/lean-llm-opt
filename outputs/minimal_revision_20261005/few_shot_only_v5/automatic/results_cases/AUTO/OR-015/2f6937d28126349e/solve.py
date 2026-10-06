import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError('Could not read CSV with supported encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    mask = df['Product Name'].astype(str).str.casefold().str.contains('aalop')
    df_aalop = df[mask].copy()
    if df_aalop.empty:
        raise ValueError("No 'Aalop' products found in the data.")
    df_aalop = df_aalop.groupby('Product Name', as_index=False).agg({'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = df_aalop['Product Name'].tolist()
    revenue = {}
    demand = {}
    inventory = {}
    for (_, row) in df_aalop.iterrows():
        key = row['Product Name']
        try:
            revenue[key] = float(row['Revenue'])
            demand[key] = int(row['Demand'])
            inventory[key] = int(row['Initial Inventory'])
        except Exception:
            raise ValueError(f"Non-numeric data for product '{key}'.")
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f"Missing coefficients for product '{i}'.")
    m = gp.Model('Restaurant_Aalop_RevMax')
    m.setParam('MIPGap', 0.0001)
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