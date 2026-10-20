import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
capacity = int(capacity_df.iloc[0]['Capacity'])
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv'
products_df = pd.read_csv(products_path, sep=',')
product_names = products_df['ProductName'].tolist()
values = products_df['Value'].tolist()
weights = products_df['Weight'].tolist()
n = len(product_names)
m = gp.Model('PharmaInventoryOptimization')
x = m.addVars(range(n), vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((values[i] * x[i] for i in range(n))), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x[i] for i in range(n))) <= capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.2f}')
    print('\nOrder quantities per product:')
    for i in range(n):
        qty = x[i].X
        if qty > 0:
            print(f'  {product_names[i]}: {int(qty)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')