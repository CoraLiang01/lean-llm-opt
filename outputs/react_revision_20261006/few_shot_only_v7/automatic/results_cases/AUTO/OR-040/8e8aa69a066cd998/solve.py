import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
    capacity_df = read_csv_robust(capacity_path, dtype=str, keep_default_na=False)
    products_df = read_csv_robust(products_path, dtype=str, keep_default_na=False)
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Missing 'Capacity' column in capacity.csv")
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in products_df.columns:
            raise ValueError(f"Missing '{col}' column in products.csv")
    if len(capacity_df) != 1:
        raise ValueError('capacity.csv must contain exactly one row for total capacity.')
    try:
        C = float(capacity_df.iloc[0]['Capacity'])
    except Exception:
        raise ValueError('Capacity value in capacity.csv is not a valid number.')
    products_df = products_df.copy()
    products_df['Value'] = pd.to_numeric(products_df['Value'], errors='raise')
    products_df['Weight'] = pd.to_numeric(products_df['Weight'], errors='raise')
    I = list(products_df['ProductName'])
    if len(set(I)) != len(I):
        raise ValueError('Duplicate ProductName entries found in products.csv.')
    b = dict(zip(products_df['ProductName'], products_df['Value']))
    w = dict(zip(products_df['ProductName'], products_df['Weight']))
    for i in I:
        if i not in b or i not in w:
            raise ValueError(f'Missing coefficients for product {i}.')
    m = gp.Model('NYC_RealEstate_Development')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((b[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * quantity_vars[i] for i in I)) <= C, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()