import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', sep=',')
platforms = capacity_df['PlatformId'].astype(int).tolist()
games = products_df['ProductName'].astype(str).tolist()
capacity = dict(zip(capacity_df['PlatformId'].astype(int), capacity_df['Capacity'].astype(int)))
value = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(int)))
weight = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(int)))
if set(platforms) != set(capacity.keys()):
    raise ValueError('Mismatch between platform index set and capacity keys.')
if set(games) != set(value.keys()) or set(games) != set(weight.keys()):
    raise ValueError('Mismatch between game index set and value/weight keys.')
m = gp.Model('VideoGameStoreListing')
x = m.addVars(platforms, games, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in platforms for j in games)), gp.GRB.MAXIMIZE)
for i in platforms:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in games)) <= capacity[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('--- Listing Plan (platform, game, units) ---')
    for i in platforms:
        for j in games:
            units = x[i, j].X
            if units > 0.5:
                print(f'Platform {i}, Game "{j}": {int(round(units))} units')
else:
    print(f'No optimal solution found. Status: {m.status}')