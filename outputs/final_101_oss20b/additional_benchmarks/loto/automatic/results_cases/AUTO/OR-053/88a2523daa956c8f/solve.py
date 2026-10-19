import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_products = pd.read_csv(products_path, sep=',')
shelves = df_capacity['ShelfID'].tolist()
products = df_products['ProductName'].tolist()
capacity_dict = dict(zip(df_capacity['ShelfID'], df_capacity['Capacity']))
weight_dict = dict(zip(df_products['ProductName'], df_products['Weight']))
value_dict = dict(zip(df_products['ProductName'], df_products['Value']))
m = gp.Model('BigMartSalesAllocation')
x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in products)) <= capacity_dict[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('\nNon‑zero allocations (shelf, product, units):')
    for i in shelves:
        for j in products:
            if x[i, j].X > 0:
                print(f'  Shelf {i}, Product {j}: {int(x[i, j].X)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')