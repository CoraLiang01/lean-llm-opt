import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
capacity = int(capacity_df.iloc[0]['Capacity'])
products_df = pd.read_csv(products_path, sep=',')
product_names = products_df['ProductName'].tolist()
values = products_df['Value'].tolist()
weights = products_df['Weight'].tolist()
m = gp.Model('NYCPropertyDevelopment')
x = m.addVars(len(product_names), vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((values[i] * x[i] for i in range(len(product_names)))), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x[i] for i in range(len(product_names)))) <= capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.2f}')
    print('--- Development Plan per area ---')
    for i, name in enumerate(product_names):
        if x[i].X > 0:
            print(f'{name}: {int(x[i].X)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')