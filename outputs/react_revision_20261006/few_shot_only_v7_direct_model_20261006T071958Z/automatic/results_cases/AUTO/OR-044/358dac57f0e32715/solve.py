import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_encodings(path):
        for enc in csv_encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv'
    cap_df = read_csv_with_encodings(capacity_path)
    prod_df = read_csv_with_encodings(products_path)
    S = cap_df['SectionID'].tolist()
    P = prod_df['ProductName'].tolist()
    if cap_df['SectionID'].duplicated().any():
        raise ValueError('Duplicate SectionID found in capacity.csv')
    C_s = dict(zip(cap_df['SectionID'], cap_df['Capacity'].astype(float)))
    if prod_df['ProductName'].duplicated().any():
        raise ValueError('Duplicate ProductName found in products.csv')
    v_p = dict(zip(prod_df['ProductName'], prod_df['Value'].astype(float)))
    w_p = dict(zip(prod_df['ProductName'], prod_df['Weight'].astype(float)))
    for s in S:
        if s not in C_s:
            raise ValueError(f'Missing capacity for section {s}')
    for p in P:
        if p not in v_p or p not in w_p:
            raise ValueError(f'Missing value or weight for product {p}')
    m = gp.Model('Supermarket_Section_Stocking')
    quantity_keys = [(s, p) for s in S for p in P]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * quantity_vars[s, p] for s in S for p in P)), GRB.MAXIMIZE)
    for s in S:
        m.addConstr(gp.quicksum((w_p[p] * quantity_vars[s, p] for p in P)) <= C_s[s], name=f'cap_{s}')
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