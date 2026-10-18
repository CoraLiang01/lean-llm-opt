import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', dtype=str, keep_default_na=False)
capacity_df['PlatformId'] = capacity_df['PlatformId'].astype(str).str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
platforms = capacity_df['PlatformId'].tolist()
capacity = dict(zip(capacity_df['PlatformId'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', dtype=str, keep_default_na=False)
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
genres = products_df['ProductName'].tolist()
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
m = gp.Model('VideoGameStore_Listing')
x_vars = m.addVars(platforms, genres, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[genre] * x_vars[platform, genre] for platform in platforms for genre in genres)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[genre] * x_vars[platform, genre] for genre in genres)) <= capacity[platform] for platform in platforms), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')