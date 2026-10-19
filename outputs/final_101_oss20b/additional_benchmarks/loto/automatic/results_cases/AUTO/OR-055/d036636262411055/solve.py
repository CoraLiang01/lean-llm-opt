import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_products = pd.read_csv(products_path, sep=',')
display_ids = df_capacity['DisplayID'].tolist()
capacity_dict = dict(zip(df_capacity['DisplayID'], df_capacity['Capacity']))
product_names = df_products['ProductName'].tolist()
product_indices = list(range(len(product_names)))
value_dict = dict(zip(product_names, df_products['Value']))
weight_dict = dict(zip(product_names, df_products['Weight']))
m = gp.Model('BoatDisplayAllocation')
x = m.addVars(display_ids, product_indices, vtype=gp.GRB.INTEGER)
obj = gp.quicksum((value_dict[product_names[j]] * x[i, j] for i in display_ids for j in product_indices))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight_dict[product_names[j]] * x[i, j] for j in product_indices)) <= capacity_dict[i] for i in display_ids))
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('Allocation:')
    for i in display_ids:
        for j in product_indices:
            val = x[i, j].X
            if val > 0:
                print(f'  Display {i}, Boat {product_names[j]}: {int(val)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')