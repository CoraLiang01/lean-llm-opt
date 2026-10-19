import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', sep=',')
products_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', sep=',')
platforms = capacity_df['PlatformId'].tolist()
platform_capacity = dict(zip(capacity_df['PlatformId'], capacity_df['Capacity']))
genres = products_df['ProductName'].tolist()
genre_value = dict(zip(products_df['ProductName'], products_df['Value']))
genre_weight = dict(zip(products_df['ProductName'], products_df['Weight']))
m = gp.Model('GameListingOptimization')
x = m.addVars(platforms, genres, vtype=gp.GRB.INTEGER)
obj = gp.quicksum((genre_value[j] * x[i, j] for i in platforms for j in genres))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((genre_weight[j] * x[i, j] for j in genres)) <= platform_capacity[i] for i in platforms))
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    for i in platforms:
        for j in genres:
            if x[i, j].X > 0:
                print(f'Platform {i}, Genre {j}: {int(x[i, j].X)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')