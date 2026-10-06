import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read CSV file at {csv_path} with tried encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df_grouped = df.groupby('Product Name', as_index=False).agg({'Revenue': 'first', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = df_grouped['Product Name'].tolist()
    revenue = dict(zip(df_grouped['Product Name'], df_grouped['Revenue']))
    demand = dict(zip(df_grouped['Product Name'], df_grouped['Demand']))
    inventory = dict(zip(df_grouped['Product Name'], df_grouped['Initial Inventory']))
    for i in items:
        if pd.isnull(revenue[i]) or pd.isnull(demand[i]) or pd.isnull(inventory[i]):
            raise ValueError(f'Missing data for product: {i}')
        if not (isinstance(revenue[i], (int, float)) and isinstance(demand[i], (int, float)) and isinstance(inventory[i], (int, float))):
            raise ValueError(f'Non-numeric data for product: {i}')
        if demand[i] < 0 or inventory[i] < 0:
            raise ValueError(f'Negative demand or inventory for product: {i}')
    m = gp.Model('Pizza_Fulfillment')
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