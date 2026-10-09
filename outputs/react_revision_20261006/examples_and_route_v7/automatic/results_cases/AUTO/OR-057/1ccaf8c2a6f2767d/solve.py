import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv', dtype=str, keep_default_na=False)
if 'PlatformID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise ValueError('Missing required columns in capacity.csv')
platforms = capacity_df['PlatformID'].astype(str).tolist()
capacity_dict = {}
for (idx, row) in capacity_df.iterrows():
    pid = str(row['PlatformID'])
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f'Non-integer Capacity for PlatformID {pid}')
    capacity_dict[pid] = cap
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise ValueError('Missing required columns in products.csv')
games = products_df['ProductName'].astype(str).tolist()
value_dict = {}
weight_dict = {}
for (idx, row) in products_df.iterrows():
    pname = str(row['ProductName'])
    try:
        val = int(row['Value'])
        wt = int(row['Weight'])
    except Exception:
        raise ValueError(f'Non-integer Value or Weight for ProductName {pname}')
    value_dict[pname] = val
    weight_dict[pname] = wt
if set(platforms) != set(capacity_dict.keys()):
    raise ValueError('Mismatch between platforms and capacity_dict keys')
if set(games) != set(value_dict.keys()) or set(games) != set(weight_dict.keys()):
    raise ValueError('Mismatch between games and value/weight dict keys')
decision_keys = [(i, j) for i in platforms for j in games]
m = gp.Model('GameListingKnapsack')
x_vars = m.addVars(decision_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in platforms for j in games)), gp.GRB.MAXIMIZE)
for i in platforms:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in games)) <= capacity_dict[i], name='')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (i, j) in decision_keys:
        var = x_vars[i, j]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')