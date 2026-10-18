import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
capacity = int(capacity_df['Capacity'].iloc[0])
products = products_df['ProductName'].tolist()
values = dict(zip(products_df['ProductName'], products_df['Value']))
weights = dict(zip(products_df['ProductName'], products_df['Weight']))
m = gp.Model('CarSalesInventory')
x = m.addVars(products, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((values[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x[p] for p in products)) <= capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.2f}')
    print('\nOptimal ordering plan:')
    for p in products:
        qty = int(x[p].X)
        if qty > 0:
            print(f'  {p}: {qty} units (Benefit per unit: {values[p]}, Weight per unit: {weights[p]})')
else:
    print(f'No optimal solution found. Status: {m.status}')