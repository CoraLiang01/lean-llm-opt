import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    cap_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv'
    prod_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv'
    decode_errors = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in decode_errors:
        try:
            cap_df = pd.read_csv(cap_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {cap_path} with supported encodings')
    for enc in decode_errors:
        try:
            prod_df = pd.read_csv(prod_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {prod_path} with supported encodings')
    if 'Capacity' not in cap_df.columns:
        raise ValueError("Missing 'Capacity' column in capacity.csv")
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in prod_df.columns:
            raise ValueError(f"Missing '{col}' column in products.csv")
    if len(cap_df) != 1:
        raise ValueError('capacity.csv must have exactly one row')
    C = cap_df.iloc[0]['Capacity']
    I = list(prod_df['ProductName'])
    v = dict(zip(prod_df['ProductName'], prod_df['Value']))
    w = dict(zip(prod_df['ProductName'], prod_df['Weight']))
    if len(set(I)) != len(I):
        raise ValueError('Duplicate ProductName entries in products.csv')
    if any((pd.isnull(v[i]) or pd.isnull(w[i]) for i in I)):
        raise ValueError('Null Value or Weight in products.csv')
    m = gp.Model('PharmacyDrugOrder')
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