import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    capacity_path = 'Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv'
    products_path = 'Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv'
    df_cap = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
    if 'Capacity' not in df_cap.columns:
        raise ValueError('Capacity column missing in capacity.csv')
    if df_cap.empty:
        raise ValueError('capacity.csv is empty')
    cap_str = df_cap['Capacity'].iloc[0]
    try:
        C = int(cap_str)
    except ValueError:
        raise ValueError(f'Invalid capacity value: {cap_str}')
    df_prod = pd.read_csv(products_path, dtype=str, keep_default_na=False)
    required_cols = {'ProductName', 'Value', 'Weight'}
    if not required_cols.issubset(df_prod.columns):
        missing = required_cols - set(df_prod.columns)
        raise ValueError(f'Missing columns in products.csv: {missing}')
    if df_prod.empty:
        raise ValueError('products.csv is empty')
    I = []
    v = {}
    w = {}
    for (_, row) in df_prod.iterrows():
        name = row['ProductName']
        if name in I:
            raise ValueError(f'Duplicate product name: {name}')
        try:
            val = int(row['Value'])
            wt = int(row['Weight'])
        except ValueError:
            raise ValueError(f'Non-integer Value or Weight for product {name}')
        I.append(name)
        v[name] = val
        w[name] = wt
    m = gurobipy.Model('PharmaRestock')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((v[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((w[i] * x[i] for i in I)) <= C, name='capacity')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')