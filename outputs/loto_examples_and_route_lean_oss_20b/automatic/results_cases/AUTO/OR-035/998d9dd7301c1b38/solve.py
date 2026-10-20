import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_df = pd.read_csv('/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', sep=',')
capacity = int(capacity_df['Capacity'].iloc[0])
products = products_df['ProductName'].tolist()
value_dict = dict(zip(products, products_df['Value']))
weight_dict = dict(zip(products, products_df['Weight']))
m = gp.Model('BakeryProfitMaximization')
x = {}
for idx, prod in enumerate(products):
    var = m.addVar(vtype=gp.GRB.INTEGER)
    x[prod] = var
m.setObjective(gp.quicksum((value_dict[prod] * x[prod] for prod in products)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[prod] * x[prod] for prod in products)) <= capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected profit: {m.objVal:.2f}')
    print('\nOrder plan (units per day):')
    for prod in products:
        qty = x[prod].X
        if qty > 0:
            print(f'  {prod}: {int(qty)}')
else:
    print(f'No optimal solution found. Status: {m.status}')