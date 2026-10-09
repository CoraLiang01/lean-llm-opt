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
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read CSV at {csv_path} with tried encodings.')
    mask = df['Product Name'].str.casefold().str.contains('faux')
    faux_df = df[mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in faux_df.columns:
            raise ValueError(f'Missing required column: {col}')
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for (idx, row) in faux_df.iterrows():
        prod_name = row['Product Name']
        if prod_name in items:
            continue
        try:
            A_i = float(row['Revenue'])
            d_i = float(row['Demand'])
            I_i = float(row['Initial Inventory'])
        except Exception as e:
            raise ValueError(f"Non-numeric parameter for product '{prod_name}': {e}")
        if any(pd.isna([A_i, d_i, I_i])):
            raise ValueError(f"Missing parameter for product '{prod_name}'")
        items.append(prod_name)
        revenue[prod_name] = A_i
        demand[prod_name] = d_i
        inventory[prod_name] = I_i
    if not items:
        raise ValueError("No products with 'FAUX' in Product Name found in the data.")
    for prod in items:
        if prod not in revenue or prod not in demand or prod not in inventory:
            raise ValueError(f"Missing parameter for product '{prod}'.")
    m = gp.Model('ZARA_FAUX_RevMax')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()