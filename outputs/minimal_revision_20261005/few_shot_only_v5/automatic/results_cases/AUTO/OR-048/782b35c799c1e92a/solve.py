import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv'
    try:
        cap = pd.read_csv(capacity_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            cap = pd.read_csv(capacity_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                cap = pd.read_csv(capacity_path, encoding='gbk')
            except UnicodeDecodeError:
                cap = pd.read_csv(capacity_path, encoding='latin-1')
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv'
    try:
        prod = pd.read_csv(products_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            prod = pd.read_csv(products_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                prod = pd.read_csv(products_path, encoding='gbk')
            except UnicodeDecodeError:
                prod = pd.read_csv(products_path, encoding='latin-1')
    if not {'StorageID', 'Capacity'}.issubset(cap.columns):
        raise ValueError('capacity.csv missing required columns')
    if not {'ProductName', 'Value', 'Weight'}.issubset(prod.columns):
        raise ValueError('products.csv missing required columns')
    S = cap['StorageID'].astype(str).unique().tolist()
    P = prod['ProductName'].astype(str).unique().tolist()
    C_s = cap.groupby('StorageID')['Capacity'].sum().to_dict()
    v_p = prod.groupby('ProductName')['Value'].sum().to_dict()
    w_p = prod.groupby('ProductName')['Weight'].sum().to_dict()
    for s in S:
        if s not in C_s:
            raise ValueError(f'Missing capacity for storage area {s}')
    for p in P:
        if p not in v_p or p not in w_p:
            raise ValueError(f'Missing value or weight for product {p}')
    keys = [(s, p) for s in S for p in P]
    m = gp.Model('Amazon_AC_Storage')
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