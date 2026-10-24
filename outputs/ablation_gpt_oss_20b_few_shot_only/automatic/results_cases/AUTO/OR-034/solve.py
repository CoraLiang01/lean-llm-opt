import pandas as pd
import gurobipy
from gurobipy import GRB
DATA_PATH = 'Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv'
df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f'Missing required column: {col}')
df['Revenue'] = pd.to_numeric(df['Revenue'], errors='raise')
df['Demand'] = pd.to_numeric(df['Demand'], errors='raise')
df['Initial Inventory'] = pd.to_numeric(df['Initial Inventory'], errors='raise')
products = df['Product Name'].tolist()
if len(set(products)) != len(products):
    raise ValueError('Duplicate product identifiers found.')
revenue = dict(zip(products, df['Revenue']))
demand = dict(zip(products, df['Demand']))
inventory = dict(zip(products, df['Initial Inventory']))
for p in products:
    if p not in revenue or p not in demand or p not in inventory:
        raise ValueError(f'Missing parameter for product: {p}')

def solve_problem():
    m = gurobipy.Model('FrenchBakeryRevenueMax')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[p] * x[p] for p in products)), GRB.MAXIMIZE)
    for p in products:
        m.addConstr(x[p] <= demand[p])
        m.addConstr(x[p] <= inventory[p])
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')