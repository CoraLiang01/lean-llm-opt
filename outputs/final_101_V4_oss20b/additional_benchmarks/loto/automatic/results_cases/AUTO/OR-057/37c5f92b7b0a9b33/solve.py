import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
platforms = capacity_df['PlatformID'].tolist()
platform_capacity = dict(zip(capacity_df['PlatformID'], capacity_df['Capacity']))
games = products_df['ProductName'].tolist()
game_value = dict(zip(products_df['ProductName'], products_df['Value']))
game_weight = dict(zip(products_df['ProductName'], products_df['Weight']))
m = gp.Model('GameListing')
x = m.addVars(platforms, games, vtype=gp.GRB.INTEGER)
obj = gp.quicksum((game_value[j] * x[i, j] for i in platforms for j in games))
m.setObjective(obj, gp.GRB.MAXIMIZE)
for i in platforms:
    m.addConstr(gp.quicksum((game_weight[j] * x[i, j] for j in games)) <= platform_capacity[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal}')
    for i in platforms:
        for j in games:
            if x[i, j].X > 0:
                print(f"Platform {i}, Game '{j}': {int(x[i, j].X)} units")
else:
    print(f'No optimal solution found. Status: {m.status}')