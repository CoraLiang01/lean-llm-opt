import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv'

    def read_csv_with_fallback(path):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    capacity_df = read_csv_with_fallback(capacity_path)
    products_df = read_csv_with_fallback(products_path)
    if 'ShelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
        raise ValueError('capacity.csv must contain columns: ShelfID, Capacity')
    if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
        raise ValueError('products.csv must contain columns: ProductName, Value, Weight')
    S = capacity_df['ShelfID'].astype(str).unique().tolist()
    P = products_df['ProductName'].astype(str).unique().tolist()
    C_s = {}
    for (_, row) in capacity_df.iterrows():
        key = str(row['ShelfID'])
        if key in C_s:
            C_s[key] += float(row['Capacity'])
        else:
            C_s[key] = float(row['Capacity'])
    v_p = {}
    w_p = {}
    for (_, row) in products_df.iterrows():
        key = str(row['ProductName'])
        v_p[key] = float(row['Value'])
        w_p[key] = float(row['Weight'])
    for s in S:
        if s not in C_s:
            raise ValueError(f'Missing capacity for shelf {s}')
    for p in P:
        if p not in v_p or p not in w_p:
            raise ValueError(f'Missing value or weight for product {p}')
    keys = [(s, p) for s in S for p in P]
    m = gp.Model('BigMart_Shelf_Allocation')
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * x[s, p] for (s, p) in keys)), GRB.MAXIMIZE)
    for s in S:
        m.addConstr(gp.quicksum((w_p[p] * x[s, p] for p in P)) <= C_s[s], name=f'cap_{s}')
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