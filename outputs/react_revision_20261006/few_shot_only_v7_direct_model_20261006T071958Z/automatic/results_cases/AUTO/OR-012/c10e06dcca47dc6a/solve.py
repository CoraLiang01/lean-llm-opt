import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM3/OnlineSalesDataset.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {csv_path} with tried encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df['Product Name'] = df['Product Name'].str.strip()
    items = df['Product Name'].tolist()
    if len(set(items)) != len(items):
        raise ValueError('Duplicate product names found in source data.')

    def to_numeric(series, name):
        try:
            return pd.to_numeric(series, errors='raise')
        except Exception:
            raise ValueError(f'Non-numeric or missing values in column: {name}')
    revenue_series = to_numeric(df['Revenue'], 'Revenue')
    demand_series = to_numeric(df['Demand'], 'Demand')
    inventory_series = to_numeric(df['Initial Inventory'], 'Initial Inventory')
    revenue = dict(zip(items, revenue_series))
    demand = dict(zip(items, demand_series))
    inventory = dict(zip(items, inventory_series))
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing parameter for product: {i}')
    m = gp.Model('OnlineRetailerRevenueMax')
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