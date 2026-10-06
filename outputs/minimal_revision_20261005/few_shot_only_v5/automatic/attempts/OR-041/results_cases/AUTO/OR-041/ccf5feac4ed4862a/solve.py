import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv'
    decode_errors = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in decode_errors:
        try:
            capacity_df = pd.read_csv(capacity_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {capacity_path} with tried encodings.')
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Missing 'Capacity' column in capacity.csv")
    if len(capacity_df) != 1:
        raise ValueError('Expected exactly one row in capacity.csv')
    C = capacity_df['Capacity'].iloc[0]
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv'
    for enc in decode_errors:
        try:
            products_df = pd.read_csv(products_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {products_path} with tried encodings.')
    required_cols = {'ProductName', 'Value', 'Weight'}
    if not required_cols.issubset(products_df.columns):
        raise ValueError(f'Missing columns in products.csv: {required_cols - set(products_df.columns)}')
    grouped = products_df.groupby('ProductName', as_index=False).agg({'Value': 'sum', 'Weight': 'sum'})
    areas = list(grouped['ProductName'])
    v = dict(zip(grouped['ProductName'], grouped['Value']))
    w = dict(zip(grouped['ProductName'], grouped['Weight']))
    if any((pd.isnull(v[a]) or pd.isnull(w[a]) for a in areas)):
        raise ValueError('Null value in Value or Weight columns for some area.')
    m = gp.Model('NYC_Development')
    x = m.addVars(areas, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[a] * x[a] for a in areas)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[a] * x[a] for a in areas)) <= C, name='capacity')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()