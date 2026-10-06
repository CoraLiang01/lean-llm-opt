import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv'
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
    if capacity_df.shape[0] != 1:
        raise ValueError('capacity.csv must have exactly one row for total capacity.')
    C = capacity_df['Capacity'].iloc[0]
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv'
    for enc in decode_errors:
        try:
            products_df = pd.read_csv(products_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {products_path} with tried encodings.')
    required_cols = {'ProductName', 'Weight', 'Value'}
    if not required_cols.issubset(products_df.columns):
        raise ValueError(f'products.csv missing columns: {required_cols - set(products_df.columns)}')
    grouped = products_df.groupby('ProductName', as_index=False).agg({'Weight': 'first', 'Value': 'first'})
    I = list(grouped['ProductName'])
    w = dict(zip(grouped['ProductName'], grouped['Weight']))
    v = dict(zip(grouped['ProductName'], grouped['Value']))
    for i in I:
        if pd.isnull(w[i]) or pd.isnull(v[i]):
            raise ValueError(f'Missing coefficient for product {i}')
        if not (isinstance(w[i], (int, float)) and isinstance(v[i], (int, float))):
            raise ValueError(f'Non-numeric coefficient for product {i}')
    if pd.isnull(C) or not isinstance(C, (int, float)):
        raise ValueError('Capacity is missing or not numeric.')
    m = gp.Model('SupermarketRestock')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * x[i] for i in I)) <= C, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()