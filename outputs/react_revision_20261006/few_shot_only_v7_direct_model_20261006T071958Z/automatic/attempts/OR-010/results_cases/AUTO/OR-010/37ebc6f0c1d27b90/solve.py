import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode CSV at {csv_path} with tried encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df = df.drop_duplicates(subset=required_cols)
    products = df['Product Name'].tolist()
    if len(set(products)) != len(products):
        raise ValueError('Duplicate product names found; product names must be unique for indexing.')
    try:
        revenue = pd.to_numeric(df.set_index('Product Name')['Revenue'], errors='raise').to_dict()
        demand = pd.to_numeric(df.set_index('Product Name')['Demand'], errors='raise').to_dict()
        inventory = pd.to_numeric(df.set_index('Product Name')['Initial Inventory'], errors='raise').to_dict()
    except Exception as e:
        raise ValueError(f'Error converting numeric columns: {e}')
    for p in products:
        if p not in revenue or p not in demand or p not in inventory:
            raise ValueError(f'Missing parameter(s) for product: {p}')
    m = gp.Model('MobileDeviceOrderFulfillment')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[p] * quantity_vars[p] for p in products)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[p] <= demand[p] for p in products), name='')
    m.addConstrs((quantity_vars[p] <= inventory[p] for p in products), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()