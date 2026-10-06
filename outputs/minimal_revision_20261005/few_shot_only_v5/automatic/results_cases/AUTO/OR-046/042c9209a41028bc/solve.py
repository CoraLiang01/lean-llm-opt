import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv'
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
        raise ValueError('capacity.csv must have exactly one row for total capacity')
    C = cap_df['Capacity'].iloc[0]
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv'
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
    required_cols = {'ProductName', 'Weight', 'Value'}
    if not required_cols.issubset(prod_df.columns):
        raise ValueError(f'Missing columns in products.csv: {required_cols - set(prod_df.columns)}')
    prod_df = prod_df.groupby('ProductName', as_index=False).agg({'Weight': 'sum', 'Value': 'sum'})
    items = prod_df['ProductName'].tolist()
    w = dict(zip(prod_df['ProductName'], prod_df['Weight']))
    v = dict(zip(prod_df['ProductName'], prod_df['Value']))
    for i in items:
        if pd.isnull(w[i]) or pd.isnull(v[i]):
            raise ValueError(f'Missing coefficient for product {i}')
    m = gp.Model('Supermarket_Replenishment')
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