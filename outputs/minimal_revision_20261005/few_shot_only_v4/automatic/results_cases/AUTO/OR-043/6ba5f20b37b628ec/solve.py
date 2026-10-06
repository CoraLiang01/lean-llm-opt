import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    product_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv'
    try:
        products = pd.read_csv(product_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            products = pd.read_csv(product_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                products = pd.read_csv(product_path, encoding='gbk')
            except UnicodeDecodeError:
                products = pd.read_csv(product_path, encoding='latin-1')
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in products.columns:
            raise ValueError(f"Missing required column '{col}' in products.csv")
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv'
    try:
        capacity_df = pd.read_csv(capacity_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            capacity_df = pd.read_csv(capacity_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                capacity_df = pd.read_csv(capacity_path, encoding='gbk')
            except UnicodeDecodeError:
                capacity_df = pd.read_csv(capacity_path, encoding='latin-1')
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Missing required column 'Capacity' in capacity.csv")
    if len(capacity_df['Capacity']) == 0:
        raise ValueError('No capacity value found in capacity.csv')
    C = capacity_df['Capacity'].iloc[0]
    I = products['ProductName'].tolist()
    v = dict(zip(products['ProductName'], products['Value']))
    w = dict(zip(products['ProductName'], products['Weight']))
    if set(v.keys()) != set(I) or set(w.keys()) != set(I):
        raise ValueError('Mismatch in product identifiers between value/weight and index set.')
    m = gp.Model('PharmacyDrugOrder')
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