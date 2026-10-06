import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv', sep=',')
capacity_df['PlatformID'] = capacity_df['PlatformID'].astype(int)
platforms = capacity_df['PlatformID'].tolist()
capacity = dict(zip(capacity_df['PlatformID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv', sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
games = products_df['ProductName'].tolist()
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(platforms) == 0:
    raise ValueError('No platforms found in capacity.csv')
if len(games) == 0:
    raise ValueError('No games found in products.csv')
if set(capacity.keys()) != set(platforms):
    raise ValueError('Mismatch in platform IDs between list and capacity dictionary')
if set(value.keys()) != set(games) or set(weight.keys()) != set(games):
    raise ValueError('Mismatch in game names between list and value/weight dictionaries')
m = gp.Model('GameListingKnapsack')
x = m.addVars(platforms, games, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in platforms for j in games)), gp.GRB.MAXIMIZE)
for i in platforms:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in games)) <= capacity[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('--- Game Listing Plan ---')
    for i in platforms:
        platform_total_value = 0
        platform_total_weight = 0
        print(f'Platform {i}:')
        for j in games:
            units = int(round(x[i, j].X))
            if units > 0:
                print(f"  Game '{j}': {units} units (Value: {value[j]}, Weight: {weight[j]})")
                platform_total_value += value[j] * units
                platform_total_weight += weight[j] * units
        print(f'  >> Total value: {platform_total_value}, Total memory used: {platform_total_weight} / {capacity[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')