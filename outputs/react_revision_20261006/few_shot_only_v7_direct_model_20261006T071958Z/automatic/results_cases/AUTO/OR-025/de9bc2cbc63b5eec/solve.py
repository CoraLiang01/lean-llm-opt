import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with tried encodings.')
    if 'Product Name' not in df.columns:
        raise ValueError("Missing 'Product Name' column in source data.")
    tablet_mask = df['Product Name'].str.casefold().str.startswith('tablet')
    tablet_df = df.loc[tablet_mask].copy()
    for col in ['Revenue', 'Demand', 'Initial Inventory']:
        if col not in tablet_df.columns:
            raise ValueError(f"Missing '{col}' column in source data.")
    items = tablet_df['Product Name'].tolist()
    if len(items) == 0:
        raise ValueError("No products found with 'Product Name' starting with 'TABLET'.")
    revenue = {}
    demand = {}
    inventory = {}
    for (idx, row) in tablet_df.iterrows():
        key = row['Product Name']
        try:
            A_i = float(row['Revenue'])
        except Exception:
            raise ValueError(f"Non-numeric or missing Revenue for product '{key}'.")
        try:
            d_i = int(float(row['Demand']))
        except Exception:
            raise ValueError(f"Non-numeric or missing Demand for product '{key}'.")
        try:
            s_i = int(float(row['Initial Inventory']))
        except Exception:
            raise ValueError(f"Non-numeric or missing Initial Inventory for product '{key}'.")
        if key in revenue:
            revenue[key] += A_i
            demand[key] += d_i
            inventory[key] += s_i
        else:
            revenue[key] = A_i
            demand[key] = d_i
            inventory[key] = s_i
    for key in items:
        if key not in revenue or key not in demand or key not in inventory:
            raise ValueError(f"Missing parameter(s) for product '{key}'.")
    m = gp.Model('Tablet_Revenue_Maximization')
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