import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv'
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv'
    decode_errors = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for encoding in decode_errors:
        try:
            products_df = pd.read_csv(products_path, encoding=encoding)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {products_path} with tried encodings.')
    for encoding in decode_errors:
        try:
            capacity_df = pd.read_csv(capacity_path, encoding=encoding)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {capacity_path} with tried encodings.')
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in products_df.columns:
            raise ValueError(f"Column '{col}' missing from products.csv")
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Column 'Capacity' missing from capacity.csv")
    I = products_df['ProductName'].astype(str).tolist()
    if len(set(I)) != len(I):
        raise ValueError('Duplicate ProductName entries found in products.csv')
    v_i = dict(zip(products_df['ProductName'].astype(str), products_df['Value']))
    w_i = dict(zip(products_df['ProductName'].astype(str), products_df['Weight']))
    C = capacity_df['Capacity'].sum()
    if pd.isnull(C) or not (isinstance(C, int) or isinstance(C, float)):
        raise ValueError('Invalid or missing Capacity value(s) in capacity.csv')
    for i in I:
        if i not in v_i or i not in w_i:
            raise ValueError(f"Missing Value or Weight for ProductName '{i}'")
    m = gp.Model('CarSales_Inventory')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w_i[i] * x[i] for i in I)) <= C, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()