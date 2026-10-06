import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    product_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv'
    decode_errors = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in decode_errors:
        try:
            products = pd.read_csv(product_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {product_path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv'
    for enc in decode_errors:
        try:
            capacity_df = pd.read_csv(capacity_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {capacity_path} with tried encodings.')
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in products.columns:
            raise ValueError(f"Column '{col}' missing from products.csv")
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Column 'Capacity' missing from capacity.csv")
    I = products['ProductName'].astype(str).tolist()
    v = products.set_index('ProductName')['Value'].to_dict()
    w = products.set_index('ProductName')['Weight'].to_dict()
    if set(I) - set(v.keys()):
        raise ValueError('Some ProductName in I missing from Value column')
    if set(I) - set(w.keys()):
        raise ValueError('Some ProductName in I missing from Weight column')
    C = capacity_df['Capacity'].sum()
    m = gp.Model('NYC_Dev')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * x[i] for i in I)) <= C, name='capacity')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')