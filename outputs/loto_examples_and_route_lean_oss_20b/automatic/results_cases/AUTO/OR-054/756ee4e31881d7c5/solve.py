import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
shelves = capacity_df['ShelfID'].tolist()
products = products_df['ProductName'].tolist()
capacity = dict(zip(capacity_df['ShelfID'], capacity_df['Capacity']))
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
m = gp.Model('BigMartShelfAllocation')
x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('\n--- Allocation of products to shelves ---')
    for i in shelves:
        for j in products:
            if x[i, j].X > 0:
                print(f'Shelf {i} -> Product {j}: {int(x[i, j].X)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')