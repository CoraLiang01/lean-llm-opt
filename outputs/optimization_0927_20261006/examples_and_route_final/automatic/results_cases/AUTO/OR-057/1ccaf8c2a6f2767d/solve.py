import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
capacity_df['PlatformID'] = capacity_df['PlatformID'].astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv', sep=',', dtype=str, keep_default_na=False)
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
platforms = capacity_df['PlatformID'].tolist()
games = products_df['ProductName'].tolist()
capacity = dict(zip(capacity_df['PlatformID'], capacity_df['Capacity']))
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(platforms) != set(capacity.keys()):
    raise ValueError('Mismatch between platform index set and capacity keys.')
if set(games) != set(value.keys()) or set(games) != set(weight.keys()):
    raise ValueError('Mismatch between game index set and value/weight keys.')
m = gp.Model('GamePlatformListing')
x_vars = m.addVars(platforms, games, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in platforms for j in games)), gp.GRB.MAXIMIZE)
for i in platforms:
    m.addConstr(gp.quicksum((weight[j] * x_vars[i, j] for j in games)) <= capacity[i], name=f'cap_{i}')
m.optimize()