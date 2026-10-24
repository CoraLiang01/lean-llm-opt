import gurobipy
from gurobipy import GRB
import pandas as pd

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv'
    df_capacity = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    df_capacity['SectionID'] = df_capacity['SectionID'].astype(str).str.strip()
    df_capacity['Capacity'] = df_capacity['Capacity'].astype(int)
    df_products = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    df_products['ProductName'] = df_products['ProductName'].astype(str).str.strip()
    df_products['Value'] = df_products['Value'].astype(int)
    df_products['Weight'] = df_products['Weight'].astype(int)
    C = dict(zip(df_capacity['SectionID'], df_capacity['Capacity']))
    V = dict(zip(df_products['ProductName'], df_products['Value']))
    W = dict(zip(df_products['ProductName'], df_products['Weight']))
    I = list(C.keys())
    J = list(V.keys())
    if set(V.keys()) != set(W.keys()):
        missing = set(V.keys()).symmetric_difference(set(W.keys()))
        raise ValueError(f'Product parameters missing for: {missing}')
    for i in I:
        if i not in C:
            raise ValueError(f'Missing capacity for section {i}')
    for j in J:
        if j not in V or j not in W:
            raise ValueError(f'Missing parameters for product {j}')
    m = gurobipy.Model('SupermarketStocking')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((V[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gurobipy.quicksum((W[j] * x[i, j] for j in J)) <= C[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()