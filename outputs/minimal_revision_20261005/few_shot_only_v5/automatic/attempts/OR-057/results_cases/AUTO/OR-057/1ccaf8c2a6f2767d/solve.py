import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not read {path} with supported encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv'
    cap_df = read_csv_with_encodings(capacity_path)
    prod_df = read_csv_with_encodings(products_path)
    if not {'PlatformID', 'Capacity'}.issubset(cap_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(prod_df.columns):
        raise ValueError('products.csv missing required columns.')
    cap_df = cap_df.drop_duplicates(subset=['PlatformID'])
    prod_df = prod_df.drop_duplicates(subset=['ProductName'])
    platforms = cap_df['PlatformID'].astype(str).tolist()
    games = prod_df['ProductName'].astype(str).tolist()
    capacity = dict(zip(cap_df['PlatformID'].astype(str), cap_df['Capacity']))
    value = dict(zip(prod_df['ProductName'].astype(str), prod_df['Value']))
    weight = dict(zip(prod_df['ProductName'].astype(str), prod_df['Weight']))
    for p in platforms:
        if p not in capacity:
            raise ValueError(f'Missing capacity for platform {p}')
    for g in games:
        if g not in value or g not in weight:
            raise ValueError(f'Missing value or weight for game {g}')
    m = gp.Model('VideoGame_Platform_Allocation')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in platforms for j in games]
    x = m.addVars(x_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[j] * x[i, j] for i in platforms for j in games)), GRB.MAXIMIZE)
    for i in platforms:
        m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in games)) <= capacity[i], name=f'cap_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()