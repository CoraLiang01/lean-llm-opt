import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM19/WomenClothingEcommerceSalesData.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read CSV at {csv_path} with tried encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df = df.dropna(subset=required_cols)
    grouped = df.groupby('Product Name', as_index=False).agg({'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = grouped['Product Name'].tolist()
    revenue = dict(zip(grouped['Product Name'], grouped['Revenue']))
    demand = dict(zip(grouped['Product Name'], grouped['Demand']))
    inventory = dict(zip(grouped['Product Name'], grouped['Initial Inventory']))
    for i in items:
        if not (i in revenue and i in demand and (i in inventory)):
            raise ValueError(f'Missing data for product: {i}')
        if not (pd.api.types.is_number(revenue[i]) and pd.api.types.is_number(demand[i]) and pd.api.types.is_number(inventory[i])):
            raise ValueError(f'Non-numeric data for product: {i}')
    m = gp.Model('WomenClothingEcommerceSales')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()