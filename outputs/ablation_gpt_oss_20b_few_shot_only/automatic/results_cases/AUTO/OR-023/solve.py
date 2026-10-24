import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    csv_path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv'
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df['Revenue'] = df['Revenue'].astype(float)
    df['Demand'] = df['Demand'].astype(int)
    df['Initial Inventory'] = df['Initial Inventory'].astype(int)
    mask = df['Product_Reference'].str.casefold().str.startswith('ele-s')
    df_filtered = df[mask]
    indices = df_filtered['Product_Reference'].tolist()
    revenue = dict(zip(indices, df_filtered['Revenue'].tolist()))
    demand = dict(zip(indices, df_filtered['Demand'].tolist()))
    inventory = dict(zip(indices, df_filtered['Initial Inventory'].tolist()))
    if not set(indices).issubset(revenue.keys()):
        missing = set(indices) - set(revenue.keys())
        raise ValueError(f'Missing revenue for products: {missing}')
    if not set(indices).issubset(demand.keys()):
        missing = set(indices) - set(demand.keys())
        raise ValueError(f'Missing demand for products: {missing}')
    if not set(indices).issubset(inventory.keys()):
        missing = set(indices) - set(inventory.keys())
        raise ValueError(f'Missing inventory for products: {missing}')
    m = gurobipy.Model('ELE_S_Revenue_Maximization')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(indices, lb=1, vtype=GRB.INTEGER, name='')
    for i in indices:
        m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inventory[i], name=f'inventory_{i}')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in indices)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()