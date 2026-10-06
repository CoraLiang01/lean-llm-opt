import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv'
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
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv'
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
    if not {'PlatformId', 'Capacity'}.issubset(cap.columns):
        raise ValueError('capacity.csv missing required columns')
    if not {'ProductName', 'Value', 'Weight'}.issubset(prod.columns):
        raise ValueError('products.csv missing required columns')
    platforms = cap['PlatformId'].astype(str).unique().tolist()
    genres = prod['ProductName'].astype(str).unique().tolist()
    capacity = dict(zip(cap['PlatformId'].astype(str), cap['Capacity']))
    value = dict(zip(prod['ProductName'].astype(str), prod['Value']))
    weight = dict(zip(prod['ProductName'].astype(str), prod['Weight']))
    for p in platforms:
        if p not in capacity:
            raise ValueError(f'Missing capacity for platform {p}')
    for g in genres:
        if g not in value or g not in weight:
            raise ValueError(f'Missing value or weight for genre {g}')
    m = gp.Model('VideoGameStore')
    x = m.addVars(platforms, genres, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[j] * x[i, j] for i in platforms for j in genres)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((weight[j] * x[i, j] for j in genres)) <= capacity[i] for i in platforms), name='')
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