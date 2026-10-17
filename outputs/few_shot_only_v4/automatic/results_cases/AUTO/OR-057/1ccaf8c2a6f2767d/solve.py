import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv', dtype=str, keep_default_na=False)
platforms = list(capacity_df['PlatformID'])
capacity = {}
for idx, row in capacity_df.iterrows():
    try:
        capacity[row['PlatformID']] = int(row['Capacity'])
    except Exception as e:
        raise ValueError(f"Invalid Capacity for PlatformID {row['PlatformID']}: {row['Capacity']}") from e
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv', dtype=str, keep_default_na=False)
genres = list(products_df['ProductName'])
value = {}
weight = {}
for idx, row in products_df.iterrows():
    try:
        value[row['ProductName']] = int(row['Value'])
        weight[row['ProductName']] = int(row['Weight'])
    except Exception as e:
        raise ValueError(f"Invalid Value/Weight for ProductName {row['ProductName']}: Value={row['Value']}, Weight={row['Weight']}") from e
if set(capacity.keys()) != set(platforms):
    raise ValueError('Mismatch in platform identifiers between capacity.csv and loaded platforms.')
if set(value.keys()) != set(genres) or set(weight.keys()) != set(genres):
    raise ValueError('Mismatch in product identifiers between products.csv and loaded genres.')
m = gp.Model('VideoGameStore_Allocation')
x_vars = m.addVars(platforms, genres, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in platforms for j in genres)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x_vars[i, j] for j in genres)) <= capacity[i] for i in platforms), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')