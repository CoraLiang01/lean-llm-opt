import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv'
    decode_errors = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in decode_errors:
        try:
            capacity_df = pd.read_csv(capacity_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {capacity_path} with supported encodings.')
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Missing 'Capacity' column in capacity.csv")
    if capacity_df.shape[0] != 1:
        raise ValueError('capacity.csv must have exactly one row for total capacity.')
    C = capacity_df['Capacity'].iloc[0]
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv'
    for enc in decode_errors:
        try:
            products_df = pd.read_csv(products_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {products_path} with supported encodings.')
    required_cols = {'ProductName', 'Value', 'Weight'}
    if not required_cols.issubset(products_df.columns):
        raise ValueError(f'products.csv missing columns: {required_cols - set(products_df.columns)}')
    products_df = products_df.groupby('ProductName', as_index=False).agg({'Value': 'sum', 'Weight': 'sum'})
    items = products_df['ProductName'].tolist()
    v = dict(zip(products_df['ProductName'], products_df['Value']))
    w = dict(zip(products_df['ProductName'], products_df['Weight']))
    for i in items:
        if i not in v or i not in w:
            raise ValueError(f'Missing value or weight for product {i}')
    m = gp.Model('PharmacyDrugOrder')
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * x[i] for i in items)) <= C, name='capacity')
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