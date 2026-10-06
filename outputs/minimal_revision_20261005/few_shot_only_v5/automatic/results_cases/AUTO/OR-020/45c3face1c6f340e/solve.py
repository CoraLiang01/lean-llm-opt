import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv'
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
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df = df.drop_duplicates(subset=required_cols)
    items = df['Product Name'].astype(str).tolist()
    if len(set(items)) != len(items):
        raise ValueError('Duplicate product names found; product names must be unique.')
    try:
        revenue = dict(zip(items, df['Revenue']))
        demand = dict(zip(items, df['Demand']))
        inventory = dict(zip(items, df['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Error mapping coefficients: {e}')
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing coefficient for product: {i}')
        for (name, d) in [('Revenue', revenue), ('Demand', demand), ('Initial Inventory', inventory)]:
            if pd.isnull(d[i]):
                raise ValueError(f'Null {name} for product: {i}')
            try:
                float(d[i])
            except Exception:
                raise ValueError(f'Non-numeric {name} for product: {i}')
    m = gp.Model('Supermarket_Fulfillment')
    m.Params.MIPGap = 0.0001
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((float(revenue[i]) * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= float(demand[i]) for i in items), name='')
    m.addConstrs((x[i] <= float(inventory[i]) for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()