import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with supported encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv'
    cap_df = read_csv_robust(capacity_path)
    prod_df = read_csv_robust(products_path)
    if not {'SectionID', 'Capacity'}.issubset(cap_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(prod_df.columns):
        raise ValueError('products.csv missing required columns.')
    S = cap_df['SectionID'].astype(str).unique().tolist()
    P = prod_df['ProductName'].astype(str).unique().tolist()
    c_s = cap_df.groupby('SectionID')['Capacity'].sum().to_dict()
    v_p = prod_df.groupby('ProductName')['Value'].sum().to_dict()
    w_p = prod_df.groupby('ProductName')['Weight'].sum().to_dict()
    for s in S:
        if s not in c_s:
            raise ValueError(f'Missing capacity for section {s}')
    for p in P:
        if p not in v_p or p not in w_p:
            raise ValueError(f'Missing value or weight for product {p}')
    m = gp.Model('Supermarket_Section_Stocking')
    m.setParam('MIPGap', 0.0001)
    keys = [(s, p) for s in S for p in P]
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * x[s, p] for s in S for p in P)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_p[p] * x[s, p] for p in P)) <= c_s[s] for s in S), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()