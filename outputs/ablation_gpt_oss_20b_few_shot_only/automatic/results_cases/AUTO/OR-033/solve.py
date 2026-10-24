import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df_baby = df[df['Product Name'].str.startswith('Baby')].copy()
    for col in ['Revenue', 'Demand', 'Initial Inventory']:
        df_baby[col] = pd.to_numeric(df_baby[col], errors='raise')
    revenue = dict(zip(df_baby['Product Name'], df_baby['Revenue']))
    demand = dict(zip(df_baby['Product Name'], df_baby['Demand']))
    inventory = dict(zip(df_baby['Product Name'], df_baby['Initial Inventory']))
    ids = list(revenue.keys())
    for i in ids:
        if i not in demand or i not in inventory:
            raise ValueError(f'Missing parameter for {i}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(ids, lb=1, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in ids)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in ids), name='')
    m.addConstrs((x[i] <= inventory[i] for i in ids), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')