import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with supported encodings.')
    required_cols = ['Product_Reference', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df = df.groupby('Product_Reference', as_index=False).agg({'Revenue': 'first', 'Demand': 'first', 'Initial Inventory': 'first'})
    items = df['Product_Reference'].tolist()
    revenue = {}
    inventory = {}
    demand = {}
    for (_, row) in df.iterrows():
        key = row['Product_Reference']
        try:
            revenue[key] = float(row['Revenue'])
        except Exception:
            raise ValueError(f'Non-numeric or missing Revenue for {key}')
        try:
            inventory[key] = int(float(row['Initial Inventory']))
        except Exception:
            raise ValueError(f'Non-numeric or missing Initial Inventory for {key}')
        try:
            demand[key] = int(float(row['Demand']))
        except Exception:
            raise ValueError(f'Non-numeric or missing Demand for {key}')
    for key in items:
        if key not in revenue or key not in inventory or key not in demand:
            raise ValueError(f'Missing coefficients for {key}')
    m = gp.Model('Supermarket_ELE_S_Revenue')
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