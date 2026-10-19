import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv', sep=',')
products_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', sep=',')
capacity_limit = int(capacity_df['Capacity'].iloc[0])
product_names = products_df['ProductName'].tolist()
values = products_df['Value'].tolist()
weights = products_df['Weight'].tolist()
m = gp.Model('RealEstateDevelopment')
var_map = {}
for idx, name in enumerate(product_names):
    var_map[name] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0)
obj = gp.quicksum((values[i] * var_map[product_names[i]] for i in range(len(product_names))))
m.setObjective(obj, gp.GRB.MAXIMIZE)
capacity_expr = gp.quicksum((weights[i] * var_map[product_names[i]] for i in range(len(product_names))))
m.addConstr(capacity_expr <= capacity_limit)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.2f}')
    print('--- Development Plan ---')
    for name, var in var_map.items():
        if var.X > 1e-06:
            print(f'{name}: {var.X:.2f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')