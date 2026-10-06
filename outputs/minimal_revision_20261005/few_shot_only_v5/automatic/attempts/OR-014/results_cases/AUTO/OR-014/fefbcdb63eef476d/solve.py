import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with tried encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    if not all((col in df.columns for col in required_cols)):
        raise ValueError(f'Missing required columns in CSV. Found columns: {df.columns.tolist()}')
    df = df.drop_duplicates(subset=required_cols)
    items = df['Product Name'].tolist()
    if len(set(items)) != len(items):
        raise ValueError('Duplicate Product Name entries found; identifiers must be unique.')
    revenue = dict(zip(df['Product Name'], df['Revenue']))
    inventory = dict(zip(df['Product Name'], df['Initial Inventory']))
    demand = dict(zip(df['Product Name'], df['Demand']))
    for i in items:
        if i not in revenue or i not in inventory or i not in demand:
            raise ValueError(f'Missing coefficient for item {i}.')
        if pd.isnull(revenue[i]) or pd.isnull(inventory[i]) or pd.isnull(demand[i]):
            raise ValueError(f'Null coefficient for item {i}.')
        if not (isinstance(revenue[i], (int, float)) and isinstance(inventory[i], (int, float)) and isinstance(demand[i], (int, float))):
            raise ValueError(f'Non-numeric coefficient for item {i}.')
    m = gp.Model('Pizza_Fulfillment')
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