import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    csv_path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv'
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required_cols = ['Full_Product_Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    products = df['Full_Product_Name'].tolist()
    rev = {}
    dem = {}
    inv = {}
    for (_, row) in df.iterrows():
        prod = row['Full_Product_Name']
        try:
            rev_val = float(row['Revenue'])
            dem_val = int(float(row['Demand']))
            inv_val = int(float(row['Initial Inventory']))
        except ValueError as e:
            raise ValueError(f'Invalid numeric value for product {prod}: {e}')
        rev[prod] = rev_val
        dem[prod] = dem_val
        inv[prod] = inv_val
    for prod in products:
        if prod not in rev or prod not in dem or prod not in inv:
            raise ValueError(f'Missing parameter for product {prod}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((rev[p] * x[p] for p in products)), GRB.MAXIMIZE)
    m.addConstrs((x[p] <= dem[p] for p in products), name='')
    m.addConstrs((x[p] <= inv[p] for p in products), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()