import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if not {'PlatformId', 'Capacity'}.issubset(capacity_df.columns):
    raise ValueError('Missing required columns in capacity.csv')
platform_ids = capacity_df['PlatformId'].astype(str).tolist()
platform_capacities = {}
for (idx, row) in capacity_df.iterrows():
    pid = str(row['PlatformId'])
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f'Invalid Capacity value for PlatformId {pid}')
    platform_capacities[pid] = cap
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise ValueError('Missing required columns in products.csv')
genre_names = products_df['ProductName'].astype(str).tolist()
genre_values = {}
genre_weights = {}
for (idx, row) in products_df.iterrows():
    gname = str(row['ProductName'])
    try:
        val = int(row['Value'])
        wt = int(row['Weight'])
    except Exception:
        raise ValueError(f'Invalid Value or Weight for ProductName {gname}')
    genre_values[gname] = val
    genre_weights[gname] = wt
platforms = platform_ids
genres = genre_names
decision_keys = [(i, j) for i in platforms for j in genres]

def solve_problem():
    m = gp.Model('VideoGamePlatformListing')
    x_vars = m.addVars(decision_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((genre_values[j] * x_vars[i, j] for (i, j) in decision_keys)), gp.GRB.MAXIMIZE)
    for i in platforms:
        m.addConstr(gp.quicksum((genre_weights[j] * x_vars[i, j] for j in genres)) <= platform_capacities[i], name=f'cap_{i}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')