import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with tried encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    faux_mask = df['Product Name'].str.casefold().str.contains('faux')
    faux_df = df[faux_mask].copy()
    if faux_df.empty:
        raise ValueError("No products with 'FAUX' in 'Product Name' found.")
    items = []
    revenue = {}
    inventory = {}
    demand = {}
    for (idx, row) in faux_df.iterrows():
        prod = row['Product Name']
        try:
            rev = float(row['Revenue'])
        except Exception:
            raise ValueError(f"Non-numeric Revenue for product '{prod}'")
        try:
            inv = float(row['Initial Inventory'])
        except Exception:
            raise ValueError(f"Non-numeric Initial Inventory for product '{prod}'")
        try:
            dem = float(row['Demand'])
        except Exception:
            raise ValueError(f"Non-numeric Demand for product '{prod}'")
        items.append(prod)
        revenue[prod] = rev
        inventory[prod] = inv
        demand[prod] = dem
    for prod in items:
        if prod not in revenue or prod not in inventory or prod not in demand:
            raise ValueError(f"Missing coefficients for product '{prod}'")
    m = gp.Model('ZARA_FAUX_Revenue_Max')
    quantity_vars = m.addVars(items, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()