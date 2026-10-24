import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='raise')
    df['Demand'] = pd.to_numeric(df['Demand'], errors='raise')
    df['Initial Inventory'] = pd.to_numeric(df['Initial Inventory'], errors='raise')
    mask = df['Product Name'].str.contains('Aalop')
    df_aalop = df[mask].copy()
    I = list(df_aalop['Product Name'])
    revenue = dict(zip(df_aalop['Product Name'], df_aalop['Revenue']))
    demand = dict(zip(df_aalop['Product Name'], df_aalop['Demand']))
    inventory = dict(zip(df_aalop['Product Name'], df_aalop['Initial Inventory']))
    for key in I:
        if key not in revenue:
            raise ValueError(f'Missing revenue for product {key}')
        if key not in demand:
            raise ValueError(f'Missing demand for product {key}')
        if key not in inventory:
            raise ValueError(f'Missing inventory for product {key}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=1, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in I), name='')
    m.addConstrs((x[i] <= inventory[i] for i in I), name='')
    m.optimize()
    return m
m = solve_problem()