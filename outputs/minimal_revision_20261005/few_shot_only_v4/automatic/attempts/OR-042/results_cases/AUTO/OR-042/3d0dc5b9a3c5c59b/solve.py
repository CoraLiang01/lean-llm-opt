import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv'
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            products_df = pd.read_csv(products_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {products_path} with tried encodings.')
    for enc in encodings:
        try:
            capacity_df = pd.read_csv(capacity_path, encoding=enc)
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
    v = products_df.set_index('ProductName')['Value'].to_dict()
    w = products_df.set_index('ProductName')['Weight'].to_dict()
    for i in I:
        if i not in v or i not in w:
            raise ValueError(f"Missing value or weight for product '{i}'")
        if pd.isnull(v[i]) or pd.isnull(w[i]):
            raise ValueError(f"Null value or weight for product '{i}'")
        try:
            v[i] = float(v[i])
            w[i] = float(w[i])
        except Exception:
            raise ValueError(f"Non-numeric value or weight for product '{i}'")
    if capacity_df['Capacity'].isnull().any():
        raise ValueError('Null value in Capacity column of capacity.csv')
    try:
        C = float(capacity_df['Capacity'].sum())
    except Exception:
        raise ValueError('Non-numeric value in Capacity column of capacity.csv')
    m = gp.Model('Pharmacy_Inventory')
    m.Params.MIPGap = 0.0001
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