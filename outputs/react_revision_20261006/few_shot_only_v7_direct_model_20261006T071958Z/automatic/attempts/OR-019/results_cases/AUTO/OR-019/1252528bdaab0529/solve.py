import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            if enc == tried_encodings[-1]:
                raise
            continue
    mask = df['Product Name'].str.casefold().str.contains('27in')
    df_27in = df[mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_27in.columns:
            raise ValueError(f'Missing required column: {col}')
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for (idx, row) in df_27in.iterrows():
        prod_name = row['Product Name']
        try:
            A_i = float(row['Revenue'])
            d_i = int(float(row['Demand']))
            I_i = int(float(row['Initial Inventory']))
        except Exception as e:
            raise ValueError(f"Non-numeric or missing data for product '{prod_name}': {e}")
        items.append(prod_name)
        revenue[prod_name] = A_i
        demand[prod_name] = d_i
        inventory[prod_name] = I_i
    for prod_name in items:
        if prod_name not in revenue or prod_name not in demand or prod_name not in inventory:
            raise ValueError(f'Missing parameter for product: {prod_name}')
    m = gp.Model('27in_Product_Fulfillment')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')