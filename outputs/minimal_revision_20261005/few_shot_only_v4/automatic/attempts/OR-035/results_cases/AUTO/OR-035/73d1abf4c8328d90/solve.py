import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv'
    try:
        products = pd.read_csv(products_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            products = pd.read_csv(products_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                products = pd.read_csv(products_path, encoding='gbk')
            except UnicodeDecodeError:
                products = pd.read_csv(products_path, encoding='latin-1')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv'
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
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in products.columns:
            raise ValueError(f"Missing column '{col}' in products.csv")
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Missing column 'Capacity' in capacity.csv")
    I = products['ProductName'].astype(str).tolist()
    p = products.set_index('ProductName')['Value'].to_dict()
    w = products.set_index('ProductName')['Weight'].to_dict()
    if len(capacity_df) != 1:
        raise ValueError('capacity.csv must contain exactly one row for total capacity.')
    C = float(capacity_df['Capacity'].iloc[0])
    for i in I:
        if i not in p or pd.isnull(p[i]):
            raise ValueError(f"Missing or null Value for product '{i}'")
        if i not in w or pd.isnull(w[i]):
            raise ValueError(f"Missing or null Weight for product '{i}'")
    m = gp.Model('Bakery_Bread_Stocking')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((p[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * x[i] for i in I)) <= C, name='storage')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()