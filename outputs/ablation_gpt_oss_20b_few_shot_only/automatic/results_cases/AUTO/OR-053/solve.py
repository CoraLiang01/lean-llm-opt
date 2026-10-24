import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv'
    df_cap = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    df_cap['Capacity'] = pd.to_numeric(df_cap['Capacity'], errors='raise')
    df_prod = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    df_prod['Value'] = pd.to_numeric(df_prod['Value'], errors='raise')
    df_prod['Weight'] = pd.to_numeric(df_prod['Weight'], errors='raise')
    Cap = {row['ShelfID']: row['Capacity'] for (_, row) in df_cap.iterrows()}
    Val = {row['ProductName']: row['Value'] for (_, row) in df_prod.iterrows()}
    Wt = {row['ProductName']: row['Weight'] for (_, row) in df_prod.iterrows()}
    I = list(Cap.keys())
    J = list(Val.keys())
    for i in I:
        if i not in Cap:
            raise ValueError(f'Missing capacity for shelf {i}')
    for j in J:
        if j not in Val:
            raise ValueError(f'Missing value for product {j}')
        if j not in Wt:
            raise ValueError(f'Missing weight for product {j}')
    m = gurobipy.Model('BigMartAllocation')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((Val[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gurobipy.quicksum((Wt[j] * x[i, j] for j in J)) <= Cap[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()