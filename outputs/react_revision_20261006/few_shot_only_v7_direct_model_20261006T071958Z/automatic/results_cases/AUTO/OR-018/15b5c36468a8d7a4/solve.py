import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            if enc == encodings[-1]:
                raise
            continue
    mask = df['Product Name'].str.casefold().str.contains('baby')
    baby_df = df[mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in baby_df.columns:
            raise ValueError(f'Missing required column: {col}')
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for (idx, row) in baby_df.iterrows():
        prod = row['Product Name']
        try:
            A_i = float(row['Revenue'])
            d_i = float(row['Demand'])
            I_i = float(row['Initial Inventory'])
        except Exception:
            raise ValueError(f"Non-numeric or missing data for product '{prod}' in one of the required columns.")
        if prod in items:
            raise ValueError(f'Duplicate product name found: {prod}')
        items.append(prod)
        revenue[prod] = A_i
        demand[prod] = d_i
        inventory[prod] = I_i
    if not set(items) == set(revenue) == set(demand) == set(inventory):
        raise ValueError('Mismatch in index sets for items, revenue, demand, or inventory.')
    m = gp.Model('Baby_Product_Fulfillment')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()