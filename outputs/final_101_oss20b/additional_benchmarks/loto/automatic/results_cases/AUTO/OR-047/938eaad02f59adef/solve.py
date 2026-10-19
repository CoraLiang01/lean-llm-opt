import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_df = pd.read_csv('/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv')
products_df = pd.read_csv('/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv')
platforms = capacity_df['PlatformId'].tolist()
platform_capacity = dict(zip(capacity_df['PlatformId'], capacity_df['Capacity']))
products = products_df['ProductName'].tolist()
product_value = dict(zip(products_df['ProductName'], products_df['Value']))
product_weight = dict(zip(products_df['ProductName'], products_df['Weight']))
m = gp.Model('GameListingOptimization')
x = m.addVars(platforms, products, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((product_value[j] * x[i, j] for i in platforms for j in products)), gp.GRB.MAXIMIZE)
for i in platforms:
    m.addConstr(gp.quicksum((product_weight[j] * x[i, j] for j in products)) <= platform_capacity[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal}')
else:
    print(f'No optimal solution found. Status: {m.status}')