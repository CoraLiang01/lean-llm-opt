import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
    decode_errors = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in decode_errors:
        try:
            products_df = pd.read_csv(products_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {products_path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
    for enc in decode_errors:
        try:
            capacity_df = pd.read_csv(capacity_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {capacity_path} with tried encodings.')
    required_prod_cols = {'ProductName', 'Value', 'Weight'}
    if not required_prod_cols.issubset(products_df.columns):
        raise ValueError(f'products.csv missing columns: {required_prod_cols - set(products_df.columns)}')
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("capacity.csv missing column: 'Capacity'")
    grouped = products_df.groupby('ProductName', sort=False, as_index=False).agg({'Value': 'sum', 'Weight': 'sum'})
    I = list(grouped['ProductName'])
    p = dict(zip(grouped['ProductName'], grouped['Value']))
    w = dict(zip(grouped['ProductName'], grouped['Weight']))
    C = capacity_df['Capacity'].sum()
    if any((pd.isnull(p[i]) or pd.isnull(w[i]) for i in I)):
        raise ValueError('Missing Value or Weight for some ProductName in products.csv')
    if any((w[i] < 0 for i in I)):
        raise ValueError('Negative Weight found in products.csv')
    if any((p[i] < 0 for i in I)):
        raise ValueError('Negative Value found in products.csv')
    if C < 0:
        raise ValueError('Negative Capacity found in capacity.csv')
    m = gp.Model('NYC_Development')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((p[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * x[i] for i in I)) <= C, name='cap')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()