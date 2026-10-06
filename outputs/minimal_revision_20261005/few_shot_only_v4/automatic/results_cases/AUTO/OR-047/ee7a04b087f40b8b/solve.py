import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            capacity_df = pd.read_csv(capacity_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {capacity_path} with tried encodings.')
    for enc in encodings:
        try:
            products_df = pd.read_csv(products_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {products_path} with tried encodings.')
    if not {'PlatformId', 'Capacity'}.issubset(capacity_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
        raise ValueError('products.csv missing required columns.')
    platforms = list(capacity_df['PlatformId'].astype(str))
    genres = list(products_df['ProductName'].astype(str))
    c = {}
    for (_, row) in capacity_df.iterrows():
        pid = str(row['PlatformId'])
        if pid in c:
            raise ValueError(f'Duplicate PlatformId {pid} in capacity.csv')
        c[pid] = row['Capacity']
    v = {}
    w = {}
    for (_, row) in products_df.iterrows():
        pname = str(row['ProductName'])
        if pname in v or pname in w:
            raise ValueError(f'Duplicate ProductName {pname} in products.csv')
        v[pname] = row['Value']
        w[pname] = row['Weight']
    if set(platforms) != set(c.keys()):
        raise ValueError('Mismatch in platform identifiers between index set and capacity dictionary.')
    if set(genres) != set(v.keys()) or set(genres) != set(w.keys()):
        raise ValueError('Mismatch in genre identifiers between index set and value/weight dictionaries.')
    keys = [(i, j) for i in platforms for j in genres]
    m = gp.Model('VideoGameStore')
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[j] * x[i, j] for i in platforms for j in genres)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w[j] * x[i, j] for j in genres)) <= c[i] for i in platforms), name='')
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