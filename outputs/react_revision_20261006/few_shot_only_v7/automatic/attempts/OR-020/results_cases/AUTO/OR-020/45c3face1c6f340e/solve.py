import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {csv_path} with tried encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df = df.drop_duplicates(subset=['Product Name'])
    items = df['Product Name'].tolist()
    if len(set(items)) != len(items):
        raise ValueError('Duplicate product names found in source data.')
    try:
        revenue = pd.to_numeric(df.set_index('Product Name')['Revenue'], errors='raise')
        demand = pd.to_numeric(df.set_index('Product Name')['Demand'], errors='raise')
        inventory = pd.to_numeric(df.set_index('Product Name')['Initial Inventory'], errors='raise')
    except Exception as e:
        raise ValueError(f'Error converting numeric columns: {e}')
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing coefficient for product: {i}')
    m = gp.Model('Supermarket_Revenue_Max')
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
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')