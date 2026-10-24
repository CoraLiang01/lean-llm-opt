import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    csv_path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv'
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df['Revenue'] = df['Revenue'].astype(float)
    df['Demand'] = df['Demand'].astype(float)
    df['Initial Inventory'] = df['Initial Inventory'].astype(float)
    df_faux = df[df['Product Name'].str.contains('FAUX')]
    products = df_faux['Product Name'].tolist()
    revenue = dict(zip(products, df_faux['Revenue']))
    demand = dict(zip(products, df_faux['Demand']))
    inv = dict(zip(products, df_faux['Initial Inventory']))
    for p in products:
        if p not in revenue:
            raise ValueError(f'Missing Revenue for product {p}')
        if p not in demand:
            raise ValueError(f'Missing Demand for product {p}')
        if p not in inv:
            raise ValueError(f'Missing Initial Inventory for product {p}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[p] * x[p] for p in products)), GRB.MAXIMIZE)
    m.addConstrs((x[p] <= demand[p] for p in products), name='')
    m.addConstrs((x[p] <= inv[p] for p in products), name='')
    m.optimize()
    return m
m = solve_problem()