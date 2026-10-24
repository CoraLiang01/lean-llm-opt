import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    csv_path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv'
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df['Product Name'] = df['Product Name'].str.strip()
    tablet_df = df[df['Product Name'].str.startswith('TABLET_')].copy()
    for col in ['Revenue', 'Demand', 'Initial Inventory']:
        tablet_df[col] = pd.to_numeric(tablet_df[col], errors='raise')
    rev = {}
    inv = {}
    dem = {}
    for (_, row) in tablet_df.iterrows():
        prod = row['Product Name']
        rev[prod] = float(row['Revenue'])
        inv[prod] = int(row['Initial Inventory'])
        dem[prod] = int(row['Demand'])
    for prod in rev:
        if prod not in inv or prod not in dem:
            raise ValueError(f'Missing parameters for product {prod}')
    model = gurobipy.Model('TabletRevenueMaximization')
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(rev.keys(), lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((rev[p] * x[p] for p in rev)), GRB.MAXIMIZE)
    for p in rev:
        model.addConstr(x[p] <= inv[p], name=f'inv_{p}')
        model.addConstr(x[p] <= dem[p], name=f'dem_{p}')
    model.optimize()
    return model
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')