import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'PlatformId' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise ValueError('capacity.csv must contain PlatformId and Capacity columns')
platforms = list(capacity_df['PlatformId'])
capacity = {}
for idx, row in capacity_df.iterrows():
    try:
        capacity[row['PlatformId']] = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for PlatformId {row['PlatformId']}")
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError('products.csv must contain ProductName, Value, and Weight columns')
genres = list(products_df['ProductName'])
value = {}
weight = {}
for idx, row in products_df.iterrows():
    try:
        value[row['ProductName']] = int(row['Value'])
        weight[row['ProductName']] = int(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Value or Weight for ProductName {row['ProductName']}")
if len(platforms) == 0 or len(genres) == 0:
    raise ValueError('No platforms or genres found in input data')
for p in platforms:
    if p not in capacity:
        raise ValueError(f'Missing capacity for platform {p}')
for g in genres:
    if g not in value or g not in weight:
        raise ValueError(f'Missing value or weight for genre {g}')
m = gp.Model('VideoGameStore_Listing')
x_vars = m.addVars(platforms, genres, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[g] * x_vars[p, g] for p in platforms for g in genres)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[g] * x_vars[p, g] for g in genres)) <= capacity[p] for p in platforms), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')