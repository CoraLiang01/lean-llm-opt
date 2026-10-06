import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read CSV at {csv_path} with tried encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df = df.dropna(subset=required_cols)
    df_grouped = df.groupby('Product Name', as_index=False).agg({'Revenue': 'first', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    products = df_grouped['Product Name'].tolist()
    revenue = {}
    demand = {}
    inventory = {}
    for (_, row) in df_grouped.iterrows():
        prod = row['Product Name']
        try:
            rev = float(row['Revenue'])
            dem = float(row['Demand'])
            inv = float(row['Initial Inventory'])
        except Exception:
            raise ValueError(f'Non-numeric data for product {prod}')
        revenue[prod] = rev
        demand[prod] = dem
        inventory[prod] = inv
    for prod in products:
        if prod not in revenue or prod not in demand or prod not in inventory:
            raise ValueError(f'Missing coefficients for product {prod}')
    m = gp.Model('MobileSalesOrderFulfillment')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= inventory[i] for i in products), name='')
    m.addConstrs((x[i] <= demand[i] for i in products), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()