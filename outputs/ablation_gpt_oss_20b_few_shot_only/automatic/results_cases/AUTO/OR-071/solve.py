import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture2/41.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required_cols = ['Product Name', 'Labor per unit', 'Material per unit', 'Selling Price', 'Variable Cost']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f'Missing columns in CSV: {missing_cols}')
    for col in required_cols[1:]:
        df[col] = pd.to_numeric(df[col], errors='raise')
    for col in required_cols[1:]:
        if df[col].isna().any():
            raise ValueError(f'NaN values found in column {col}')
    products = df['Product Name'].tolist()
    L = dict(zip(products, df['Labor per unit']))
    M = dict(zip(products, df['Material per unit']))
    S = dict(zip(products, df['Selling Price']))
    C = dict(zip(products, df['Variable Cost']))
    for p in products:
        if p not in L or p not in M or p not in S or (p not in C):
            raise ValueError(f'Missing parameter for product {p}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    fixed_cost = 4500
    m.setObjective(gurobipy.quicksum(((S[p] - C[p]) * x[p] for p in products)) - fixed_cost, GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((L[p] * x[p] for p in products)) <= 1650, name='labor')
    m.addConstr(gurobipy.quicksum((M[p] * x[p] for p in products)) <= 1850, name='material')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()