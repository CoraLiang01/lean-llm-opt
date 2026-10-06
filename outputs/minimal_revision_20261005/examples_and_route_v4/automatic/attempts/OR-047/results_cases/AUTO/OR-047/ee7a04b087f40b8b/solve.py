import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', sep=',')
capacity_df['PlatformId'] = capacity_df['PlatformId'].astype(int)
platforms = capacity_df['PlatformId'].tolist()
platform_cap = dict(zip(capacity_df['PlatformId'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str)
genres = products_df['ProductName'].tolist()
genre_value = dict(zip(products_df['ProductName'], products_df['Value']))
genre_weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(platforms) != len(platform_cap):
    raise ValueError('Mismatch in number of platforms and capacities.')
if len(genres) != len(genre_value) or len(genres) != len(genre_weight):
    raise ValueError('Mismatch in number of genres and their value/weight.')
index_set = [(i, j) for i in platforms for j in genres]

def solve_problem():
    m = gp.Model('VideoGameStoreMultiKnapsack')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(index_set, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((genre_value[j] * x[i, j] for (i, j) in index_set)), gp.GRB.MAXIMIZE)
    for i in platforms:
        m.addConstr(gp.quicksum((genre_weight[j] * x[i, j] for j in genres)) <= platform_cap[i], name=f'cap_{i}')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')