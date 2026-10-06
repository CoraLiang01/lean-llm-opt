import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Unable to read CSV with supported encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    mask = df['Product Name'].str.casefold().str.contains('tablet')
    df_tablet = df[mask].copy()
    if df_tablet.empty:
        raise ValueError("No 'TABLET' products found in the source data.")
    items = df_tablet['Product Name'].tolist()
    revenue = {}
    inventory = {}
    demand = {}
    for (_, row) in df_tablet.iterrows():
        key = row['Product Name']
        try:
            revenue[key] = float(row['Revenue'])
            inventory[key] = float(row['Initial Inventory'])
            demand[key] = float(row['Demand'])
        except Exception as e:
            raise ValueError(f"Invalid data for product '{key}': {e}")
    for i in items:
        if i not in revenue or i not in inventory or i not in demand:
            raise ValueError(f"Missing coefficients for product '{i}'.")
    m = gp.Model('SmartphoneRetailOutlet_TABLET')
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()