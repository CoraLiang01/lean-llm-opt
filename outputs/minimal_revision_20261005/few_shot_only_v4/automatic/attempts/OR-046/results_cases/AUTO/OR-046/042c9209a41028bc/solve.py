import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv'
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
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in products.columns:
            raise ValueError(f"Missing required column '{col}' in products.csv")
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv'
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
    if capacity_df['Capacity'].isnull().any():
        raise ValueError("Null value found in 'Capacity' column of capacity.csv")
    if len(capacity_df['Capacity']) == 0:
        raise ValueError('No capacity value found in capacity.csv')
    C = float(capacity_df['Capacity'].iloc[0])
    products = products.drop_duplicates(subset=['ProductName'])
    I = list(products['ProductName'])
    if len(I) == 0:
        raise ValueError('No products found in products.csv')
    v = products.set_index('ProductName')['Value'].to_dict()
    w = products.set_index('ProductName')['Weight'].to_dict()
    for i in I:
        if pd.isnull(v[i]) or pd.isnull(w[i]):
            raise ValueError(f"Missing value or weight for product '{i}'")
        try:
            v[i] = float(v[i])
            w[i] = float(w[i])
        except Exception:
            raise ValueError(f"Non-numeric value or weight for product '{i}'")
    m = gp.Model('SupermarketStockReplenishment')
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