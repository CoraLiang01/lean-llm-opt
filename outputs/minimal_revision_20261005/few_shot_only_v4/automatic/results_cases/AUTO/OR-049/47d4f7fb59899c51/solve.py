import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f'Could not decode {path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv'
    cap_df = read_csv_with_encodings(capacity_path)
    prod_df = read_csv_with_encodings(products_path)
    if not {'ShelfID', 'Capacity'}.issubset(cap_df.columns):
        raise ValueError('capacity.csv must contain columns: ShelfID, Capacity')
    if not {'ProductName', 'Value', 'Weight'}.issubset(prod_df.columns):
        raise ValueError('products.csv must contain columns: ProductName, Value, Weight')
    S = cap_df['ShelfID'].astype(str).unique().tolist()
    P = prod_df['ProductName'].astype(str).unique().tolist()
    c_s = {}
    for (_, row) in cap_df.iterrows():
        key = str(row['ShelfID'])
        if key in c_s:
            c_s[key] += row['Capacity']
        else:
            c_s[key] = row['Capacity']
    v_p = {}
    w_p = {}
    for (_, row) in prod_df.iterrows():
        key = str(row['ProductName'])
        if key in v_p:
            v_p[key] += row['Value']
            w_p[key] += row['Weight']
        else:
            v_p[key] = row['Value']
            w_p[key] = row['Weight']
    for s in S:
        if s not in c_s:
            raise ValueError(f'Missing capacity for shelf {s}')
    for p in P:
        if p not in v_p or p not in w_p:
            raise ValueError(f'Missing value or weight for product {p}')
    m = gp.Model('Retail_Shelf_Allocation')
    m.setParam('MIPGap', 0.0001)
    keys = [(s, p) for s in S for p in P]
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * x[s, p] for s in S for p in P)), GRB.MAXIMIZE)
    for s in S:
        m.addConstr(gp.quicksum((w_p[p] * x[s, p] for p in P)) <= c_s[s], name=f'cap_{s}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()