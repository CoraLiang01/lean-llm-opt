import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
    try:
        cap_df = pd.read_csv(capacity_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            cap_df = pd.read_csv(capacity_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                cap_df = pd.read_csv(capacity_path, encoding='gbk')
            except UnicodeDecodeError:
                cap_df = pd.read_csv(capacity_path, encoding='latin-1')
    if 'Capacity' not in cap_df.columns:
        raise ValueError("Missing 'Capacity' column in capacity.csv")
    if cap_df.shape[0] != 1:
        raise ValueError('capacity.csv must have exactly one row')
    C = cap_df['Capacity'].iloc[0]
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
    try:
        prod_df = pd.read_csv(products_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            prod_df = pd.read_csv(products_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                prod_df = pd.read_csv(products_path, encoding='gbk')
            except UnicodeDecodeError:
                prod_df = pd.read_csv(products_path, encoding='latin-1')
    required_cols = {'ProductName', 'Value', 'Weight'}
    if not required_cols.issubset(prod_df.columns):
        raise ValueError(f'products.csv missing columns: {required_cols - set(prod_df.columns)}')
    prod_df = prod_df.groupby('ProductName', as_index=False).agg({'Value': 'sum', 'Weight': 'sum'})
    areas = prod_df['ProductName'].tolist()
    v = dict(zip(prod_df['ProductName'], prod_df['Value']))
    w = dict(zip(prod_df['ProductName'], prod_df['Weight']))
    for i in areas:
        if pd.isnull(v[i]) or pd.isnull(w[i]):
            raise ValueError(f'Missing Value or Weight for area {i}')
    m = gp.Model('NYC_Development')
    x = m.addVars(areas, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * x[i] for i in areas)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * x[i] for i in areas)) <= C, name='capacity')
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