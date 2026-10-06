import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    product_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            products = pd.read_csv(product_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {product_path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv'
    for enc in encodings:
        try:
            capacity_df = pd.read_csv(capacity_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {capacity_path} with tried encodings.')
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in products.columns:
            raise ValueError(f"Missing column '{col}' in products.csv")
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Missing column 'Capacity' in capacity.csv")
    I = products['ProductName'].tolist()
    v = products.set_index('ProductName')['Value'].to_dict()
    w = products.set_index('ProductName')['Weight'].to_dict()
    if set(v.keys()) != set(I) or set(w.keys()) != set(I):
        raise ValueError('Mismatch in ProductName keys between Value/Weight and I.')
    if len(capacity_df) != 1:
        raise ValueError('capacity.csv must have exactly one row.')
    C = float(capacity_df.iloc[0]['Capacity'])
    m = gp.Model('CarSales_Inventory')
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * x[i] for i in I)) <= C, name='capacity')
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