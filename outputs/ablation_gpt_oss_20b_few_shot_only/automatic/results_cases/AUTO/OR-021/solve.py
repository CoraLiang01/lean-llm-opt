import os
import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM12/Salesofsummerclothes.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(f'CSV file not found: {path}')
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required_cols = ['Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')

    def clean_numeric(series):
        return pd.to_numeric(series.str.replace(',', '').str.strip(), errors='coerce')
    revenue_series = clean_numeric(df['Revenue'])
    demand_series = clean_numeric(df['Demand'])
    inventory_series = clean_numeric(df['Initial Inventory'])
    if revenue_series.isna().any() or demand_series.isna().any() or inventory_series.isna().any():
        raise ValueError('Non-numeric data found in required columns.')
    if (demand_series < 1).any() or (inventory_series < 1).any():
        raise ValueError('Demand or inventory less than 1 for some products.')
    I = df.index.tolist()
    revenue = dict(zip(I, revenue_series))
    demand = dict(zip(I, demand_series))
    inventory = dict(zip(I, inventory_series))
    m = gurobipy.Model('RevenueMaximization')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=1, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inventory[i], name=f'inventory_{i}')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')