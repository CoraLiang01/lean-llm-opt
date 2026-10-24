import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv'
    df_cap = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    df_cap['Capacity'] = df_cap['Capacity'].astype(int)
    C = {row['DisplayID']: row['Capacity'] for (_, row) in df_cap.iterrows()}
    df_prod = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    df_prod['Value'] = df_prod['Value'].astype(int)
    df_prod['Weight'] = df_prod['Weight'].astype(int)
    V = {row['ProductName']: row['Value'] for (_, row) in df_prod.iterrows()}
    W = {row['ProductName']: row['Weight'] for (_, row) in df_prod.iterrows()}
    I = list(C.keys())
    J = list(V.keys())
    for i in I:
        if i not in C:
            raise ValueError(f'Missing capacity for DisplayID {i}')
    for j in J:
        if j not in V:
            raise ValueError(f'Missing value for ProductName {j}')
        if j not in W:
            raise ValueError(f'Missing weight for ProductName {j}')
    m = gurobipy.Model('BoatDisplayAssignment')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((V[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(gurobipy.quicksum((W[j] * x[i, j] for j in J)) <= C[i])
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()