import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', sep=',')
platforms = capacity_df['PlatformId'].astype(int).tolist()
genres = products_df['ProductName'].astype(str).tolist()
capacity = dict(zip(capacity_df['PlatformId'].astype(int), capacity_df['Capacity'].astype(int)))
value = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(int)))
weight = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(int)))
if set(platforms) != set(capacity.keys()):
    raise ValueError('Mismatch between platform index set and capacity keys.')
if set(genres) != set(value.keys()) or set(genres) != set(weight.keys()):
    raise ValueError('Mismatch between genre index set and value/weight keys.')
m = gp.Model('VideoGamePlatformListing')
x = m.addVars(platforms, genres, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in platforms for j in genres)), gp.GRB.MAXIMIZE)
for i in platforms:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in genres)) <= capacity[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('\n--- Listing Plan (units of each genre per platform) ---')
    for i in platforms:
        print(f'Platform {i} (Capacity: {capacity[i]}):')
        used = 0
        for j in genres:
            units = int(round(x[i, j].X))
            if units > 0:
                print(f'  {j}: {units} units (Value/unit: {value[j]}, Weight/unit: {weight[j]})')
                used += weight[j] * units
        print(f'  Total memory used: {used} / {capacity[i]}\n')
else:
    print(f'No optimal solution found. Status: {m.status}')