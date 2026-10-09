import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM3/OnlineSalesDataset.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            if enc == tried_encodings[-1]:
                raise
            continue
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df_grouped = df.groupby('Product Name', as_index=False).agg({'Revenue': 'first', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    products = df_grouped['Product Name'].tolist()
    revenue = {}
    demand = {}
    inventory = {}
    for (_, row) in df_grouped.iterrows():
        key = row['Product Name']
        try:
            revenue[key] = float(row['Revenue'])
        except Exception:
            raise ValueError(f'Non-numeric or missing Revenue for product: {key}')
        try:
            demand[key] = int(float(row['Demand']))
        except Exception:
            raise ValueError(f'Non-numeric or missing Demand for product: {key}')
        try:
            inventory[key] = int(float(row['Initial Inventory']))
        except Exception:
            raise ValueError(f'Non-numeric or missing Initial Inventory for product: {key}')
    for key in products:
        if key not in revenue or key not in demand or key not in inventory:
            raise ValueError(f'Missing coefficients for product: {key}')
    m = gp.Model('OnlineRetailerRevenueMax')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in products)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in products), name='')
    m.addConstrs((quantity_vars[i] <= demand[i] for i in products), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')