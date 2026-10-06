import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', sep=',')
platforms = capacity_df['PlatformId'].astype(int).tolist()
platform_cap = dict(zip(capacity_df['PlatformId'].astype(int), capacity_df['Capacity'].astype(int)))
games = products_df['ProductName'].astype(str).tolist()
game_value = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(int)))
game_weight = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(int)))
if set(platforms) != set(capacity_df['PlatformId'].astype(int)):
    raise ValueError('Mismatch in platform indices.')
if set(games) != set(products_df['ProductName'].astype(str)):
    raise ValueError('Mismatch in game/genre indices.')
if set(game_value.keys()) != set(games) or set(game_weight.keys()) != set(games):
    raise ValueError('Missing value or weight data for some games.')
m = gp.Model('VideoGameStoreListing')
x = m.addVars(platforms, games, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((game_value[j] * x[i, j] for i in platforms for j in games)), gp.GRB.MAXIMIZE)
for i in platforms:
    m.addConstr(gp.quicksum((game_weight[j] * x[i, j] for j in games)) <= platform_cap[i], name='')
m.optimize()