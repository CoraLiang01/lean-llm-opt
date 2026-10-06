import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    product_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv'
    decode_errors = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in decode_errors:
        try:
            products = pd.read_csv(product_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {product_path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv'
    for enc in decode_errors:
        try:
            capacity_df = pd.read_csv(capacity_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {capacity_path} with tried encodings.')
    for col in ['ProductName', 'Weight', 'Value']:
        if col not in products.columns:
            raise ValueError(f"Column '{col}' missing from products.csv")
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Column 'Capacity' missing from capacity.csv")
    I = products['ProductName'].astype(str).unique().tolist()
    v_i = products.groupby('ProductName')['Value'].sum().to_dict()
    w_i = products.groupby('ProductName')['Weight'].sum().to_dict()
    if set(I) - set(v_i.keys()):
        raise ValueError('Some ProductName in I missing from v_i')
    if set(I) - set(w_i.keys()):
        raise ValueError('Some ProductName in I missing from w_i')
    if len(capacity_df) != 1:
        raise ValueError('capacity.csv must have exactly one row')
    C = float(capacity_df.iloc[0]['Capacity'])
    m = gp.Model('Supermarket_Produce_Order')
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