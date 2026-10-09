import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', sep=',')
capacity_df['PlatformId'] = capacity_df['PlatformId'].astype(int)
products_df['ProductName'] = products_df['ProductName'].astype(str)
platforms = capacity_df['PlatformId'].tolist()
genres = products_df['ProductName'].tolist()
platform_capacity = dict(zip(capacity_df['PlatformId'], capacity_df['Capacity']))
genre_value = dict(zip(products_df['ProductName'], products_df['Value']))
genre_weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(platforms) != set(platform_capacity.keys()):
    raise ValueError('Mismatch between platform index set and capacity data keys.')
if set(genres) != set(genre_value.keys()) or set(genres) != set(genre_weight.keys()):
    raise ValueError('Mismatch between genre index set and value/weight data keys.')
m = gp.Model('VideoGameStore_MultiKnapsack')
x = m.addVars(platforms, genres, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((genre_value[j] * x[i, j] for i in platforms for j in genres)), gp.GRB.MAXIMIZE)
for i in platforms:
    m.addConstr(gp.quicksum((genre_weight[j] * x[i, j] for j in genres)) <= platform_capacity[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('--- Listing Plan (platform, genre, units) ---')
    for i in platforms:
        for j in genres:
            units = x[i, j].X
            if units > 0.5:
                print(f"Platform {i}, Genre '{j}': {int(round(units))} units")
else:
    print(f'No optimal solution found. Status: {m.status}')