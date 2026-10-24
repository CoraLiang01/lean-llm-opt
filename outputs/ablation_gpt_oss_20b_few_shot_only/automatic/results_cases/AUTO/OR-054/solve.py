import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv'
    df_capacity = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    df_capacity['Capacity'] = pd.to_numeric(df_capacity['Capacity'], errors='raise')
    df_products = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    df_products['Value'] = pd.to_numeric(df_products['Value'], errors='raise')
    df_products['Weight'] = pd.to_numeric(df_products['Weight'], errors='raise')
    c = dict(zip(df_capacity['ShelfID'], df_capacity['Capacity']))
    w = dict(zip(df_products['ProductName'], df_products['Weight']))
    v = dict(zip(df_products['ProductName'], df_products['Value']))
    I = list(c.keys())
    J = list(w.keys())
    missing_c = set(I) - set(c.keys())
    if missing_c:
        raise ValueError(f'Missing capacity for shelves: {missing_c}')
    missing_w = set(J) - set(w.keys())
    if missing_w:
        raise ValueError(f'Missing weight for products: {missing_w}')
    missing_v = set(J) - set(v.keys())
    if missing_v:
        raise ValueError(f'Missing value for products: {missing_v}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in I for j in J], vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gurobipy.quicksum((v[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gurobipy.quicksum((w[j] * x[i, j] for j in J)) <= c[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()