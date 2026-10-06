import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', sep=',')
capacity_df['PlatformId'] = capacity_df['PlatformId'].astype(int)
platforms = capacity_df['PlatformId'].tolist()
capacity = dict(zip(capacity_df['PlatformId'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str)
genres = products_df['ProductName'].tolist()
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(platforms) == 0:
    raise ValueError('No platforms found in capacity.csv')
if len(genres) == 0:
    raise ValueError('No genres/games found in products.csv')
if set(capacity.keys()) != set(platforms):
    raise ValueError('Mismatch in platform IDs between capacity.csv and loaded platforms')
if set(value.keys()) != set(genres) or set(weight.keys()) != set(genres):
    raise ValueError('Mismatch in genre keys between products.csv and loaded genres')
m = gp.Model('VideoGameStore_MultiKnapsack')
x = m.addVars(platforms, genres, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in platforms for j in genres)), gp.GRB.MAXIMIZE)
for i in platforms:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in genres)) <= capacity[i], name='')
m.optimize()