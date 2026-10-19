import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
shelves = capacity_df['ShelfID'].tolist()
capacities = dict(zip(capacity_df['ShelfID'], capacity_df['Capacity']))
products = products_df['ProductName'].tolist()
values = dict(zip(products_df['ProductName'], products_df['Value']))
weights = dict(zip(products_df['ProductName'], products_df['Weight']))
product_index = {p: idx for idx, p in enumerate(products)}
m = gp.Model('RetailShelfAllocation')
x = {}
for i in shelves:
    for j in products:
        var_name = f'x_{i}_{product_index[j]}'
        x[i, j] = m.addVar(vtype=gp.GRB.INTEGER)
objective = gp.quicksum((values[j] * x[i, j] for i in shelves for j in products))
m.setObjective(objective, gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weights[j] * x[i, j] for j in products)) <= capacities[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    for i in shelves:
        for j in products:
            val = x[i, j].X
            if val > 0:
                print(f'Shelf {i}, Product "{j}": {int(val)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')