import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv'
    try:
        df_capacity = pd.read_csv(capacity_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            df_capacity = pd.read_csv(capacity_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df_capacity = pd.read_csv(capacity_path, encoding='gbk')
            except UnicodeDecodeError:
                df_capacity = pd.read_csv(capacity_path, encoding='latin-1')
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv'
    try:
        df_products = pd.read_csv(products_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            df_products = pd.read_csv(products_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df_products = pd.read_csv(products_path, encoding='gbk')
            except UnicodeDecodeError:
                df_products = pd.read_csv(products_path, encoding='latin-1')
    if not {'BookshelfID', 'Capacity'}.issubset(df_capacity.columns):
        raise ValueError('capacity.csv missing required columns')
    if not {'ProductName', 'Value', 'Weight'}.issubset(df_products.columns):
        raise ValueError('products.csv missing required columns')
    B = df_capacity['BookshelfID'].astype(str).unique().tolist()
    P = df_products['ProductName'].astype(str).unique().tolist()
    C_b = {}
    for (_, row) in df_capacity.iterrows():
        key = str(row['BookshelfID'])
        if key in C_b:
            C_b[key] += row['Capacity']
        else:
            C_b[key] = row['Capacity']
    v_p = {}
    w_p = {}
    for (_, row) in df_products.iterrows():
        key = str(row['ProductName'])
        if key in v_p:
            v_p[key] += row['Value']
            w_p[key] += row['Weight']
        else:
            v_p[key] = row['Value']
            w_p[key] = row['Weight']
    for b in B:
        if b not in C_b:
            raise ValueError(f'BookshelfID {b} missing capacity')
    for p in P:
        if p not in v_p or p not in w_p:
            raise ValueError(f'ProductName {p} missing value or weight')
    keys = [(b, p) for b in B for p in P]
    m = gp.Model('Bookshelf_Allocation')
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * x[b, p] for b in B for p in P)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_p[p] * x[b, p] for p in P)) <= C_b[b] for b in B), name='')
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