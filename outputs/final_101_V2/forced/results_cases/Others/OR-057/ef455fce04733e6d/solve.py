import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv', sep=',')
capacity_df['PlatformID'] = capacity_df['PlatformID'].astype(int)
products_df['ProductName'] = products_df['ProductName'].astype(str)
platforms = capacity_df['PlatformID'].tolist()
games = products_df['ProductName'].tolist()
capacity = dict(zip(capacity_df['PlatformID'], capacity_df['Capacity']))
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(platforms) != len(capacity):
    raise ValueError('Mismatch in number of platforms and capacities.')
if len(games) != len(value) or len(games) != len(weight):
    raise ValueError('Mismatch in number of games and their value/weight.')
m = gp.Model('GameListingKnapsack')
x = m.addVars(platforms, games, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in platforms for j in games)), gp.GRB.MAXIMIZE)
for i in platforms:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in games)) <= capacity[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('Game listing plan (platform, game, units):')
    for i in platforms:
        for j in games:
            units = x[i, j].X
            if units > 0.5:
                print(f"  Platform {i}, Game '{j}': {int(round(units))} units")
else:
    print(f'No optimal solution found. Status: {m.status}')