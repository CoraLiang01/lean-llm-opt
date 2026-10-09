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
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv'
    capacity_df = read_csv_with_encodings(capacity_path)
    products_df = read_csv_with_encodings(products_path)
    if not {'PlatformId', 'Capacity'}.issubset(capacity_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
        raise ValueError('products.csv missing required columns.')
    platforms = capacity_df['PlatformId'].unique().tolist()
    games = products_df['ProductName'].unique().tolist()
    try:
        capacity = {row['PlatformId']: float(row['Capacity']) for (_, row) in capacity_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing Capacity: {e}')
    try:
        value = {row['ProductName']: float(row['Value']) for (_, row) in products_df.iterrows()}
        weight = {row['ProductName']: float(row['Weight']) for (_, row) in products_df.iterrows()}
    except Exception as e:
        raise ValueError(f'Error parsing Value/Weight: {e}')
    for p in platforms:
        if p not in capacity:
            raise ValueError(f'Missing capacity for platform {p}')
    for g in games:
        if g not in value or g not in weight:
            raise ValueError(f'Missing value/weight for game {g}')
    keys = [(i, j) for i in platforms for j in games]
    m = gp.Model('VideoGameStore')
    quantity_vars = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[j] * quantity_vars[i, j] for i in platforms for j in games)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((weight[j] * quantity_vars[i, j] for j in games)) <= capacity[i] for i in platforms), name='')
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