import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
capacity_df['PlatformID'] = capacity_df['PlatformID'].astype(int)
products_df['ProductName'] = products_df['ProductName'].astype(str)
platforms = list(capacity_df['PlatformID'])
games = list(products_df['ProductName'])
capacity = dict(zip(capacity_df['PlatformID'], capacity_df['Capacity']))
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(platforms) != set(capacity.keys()):
    raise ValueError('Mismatch between platform index set and capacity keys.')
if set(games) != set(value.keys()) or set(games) != set(weight.keys()):
    raise ValueError('Mismatch between game index set and value/weight keys.')
index_keys = [(i, j) for i in platforms for j in games]

def solve_problem():
    m = gp.Model('VideoGamePlatformListing')
    x = m.addVars(index_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value[j] * x[i, j] for i in platforms for j in games)), gp.GRB.MAXIMIZE)
    for i in platforms:
        m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in games)) <= capacity[i], name=f'cap_{i}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')