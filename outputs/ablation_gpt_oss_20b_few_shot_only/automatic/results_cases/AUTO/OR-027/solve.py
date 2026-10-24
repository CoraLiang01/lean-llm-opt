import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    csv_path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv'
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='raise')
    df['Demand'] = pd.to_numeric(df['Demand'], errors='raise')
    df['Initial Inventory'] = pd.to_numeric(df['Initial Inventory'], errors='raise')
    products = df['Sub Category'].tolist()
    revenue = dict(zip(products, df['Revenue']))
    demand = dict(zip(products, df['Demand']))
    inventory = dict(zip(products, df['Initial Inventory']))
    for p in products:
        if p not in revenue:
            raise ValueError(f"Missing revenue for product '{p}'")
        if p not in demand:
            raise ValueError(f"Missing demand for product '{p}'")
        if p not in inventory:
            raise ValueError(f"Missing inventory for product '{p}'")
    m = gurobipy.Model('MaxRevenue')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in products), name='')
    m.addConstrs((x[i] <= inventory[i] for i in products), name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')