import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    cap_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv'
    prod_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            cap_df = pd.read_csv(cap_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {cap_path} with tried encodings.')
    for enc in encodings:
        try:
            prod_df = pd.read_csv(prod_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {prod_path} with tried encodings.')
    if 'Capacity' not in cap_df.columns:
        raise ValueError("Missing 'Capacity' column in capacity.csv")
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in prod_df.columns:
            raise ValueError(f"Missing '{col}' column in products.csv")
    if len(cap_df) != 1:
        raise ValueError('capacity.csv must have exactly one row for total capacity.')
    C = cap_df.iloc[0]['Capacity']
    prod_df = prod_df.copy()
    prod_df['ProductName'] = prod_df['ProductName'].astype(str)
    items = prod_df['ProductName'].tolist()
    v = dict(zip(prod_df['ProductName'], prod_df['Value']))
    w = dict(zip(prod_df['ProductName'], prod_df['Weight']))
    if any((pd.isnull(v[i]) or pd.isnull(w[i]) for i in items)):
        raise ValueError('Missing Value or Weight for some products.')
    m = gp.Model('CarSales_Inventory')
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