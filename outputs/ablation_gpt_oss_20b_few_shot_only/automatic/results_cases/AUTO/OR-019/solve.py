import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    csv_path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv'
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df['Revenue'] = df['Revenue'].astype(float)
    df['Demand'] = df['Demand'].astype(int)
    df['Initial Inventory'] = df['Initial Inventory'].astype(int)
    mask = df['Product Name'].str.casefold().str.contains('27in')
    df_27 = df[mask].copy()
    I = list(df_27['Product Name'])
    rev = dict(zip(df_27['Product Name'], df_27['Revenue']))
    dem = dict(zip(df_27['Product Name'], df_27['Demand']))
    inv = dict(zip(df_27['Product Name'], df_27['Initial Inventory']))
    for key in I:
        if key not in rev:
            raise ValueError(f'Missing revenue for product {key!r}')
        if key not in dem:
            raise ValueError(f'Missing demand for product {key!r}')
        if key not in inv:
            raise ValueError(f'Missing inventory for product {key!r}')
    m = gurobipy.Model('27in_Product_Sales')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= dem[i] for i in I), name='')
    m.addConstrs((x[i] <= inv[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()