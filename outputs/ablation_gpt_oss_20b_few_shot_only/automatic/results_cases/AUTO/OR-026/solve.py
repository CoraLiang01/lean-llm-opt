import gurobipy
from gurobipy import GRB
import pandas as pd

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='raise')
    df['Demand'] = pd.to_numeric(df['Demand'], errors='raise')
    df['Initial Inventory'] = pd.to_numeric(df['Initial Inventory'], errors='raise')
    mask = df['Product Name'].str.strip().str.startswith('Fashion')
    df_fashion = df[mask].copy()
    if df_fashion.empty:
        raise ValueError("No products with category 'Fashion' found.")
    ids = df_fashion.index.tolist()
    revenue = df_fashion.loc[ids, 'Revenue'].to_dict()
    demand = df_fashion.loc[ids, 'Demand'].to_dict()
    init_inv = df_fashion.loc[ids, 'Initial Inventory'].to_dict()
    for i in ids:
        if i not in revenue or i not in demand or i not in init_inv:
            raise ValueError(f'Missing parameter for product index {i}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(ids, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in ids)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in ids), name='')
    m.addConstrs((x[i] <= init_inv[i] for i in ids), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()