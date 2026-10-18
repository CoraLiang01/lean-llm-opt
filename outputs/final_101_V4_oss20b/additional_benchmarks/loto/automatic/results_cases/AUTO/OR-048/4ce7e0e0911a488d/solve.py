import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_products = pd.read_csv(products_path, sep=',')
storage_ids = df_capacity['StorageID'].tolist()
capacity_dict = dict(zip(df_capacity['StorageID'], df_capacity['Capacity']))
product_names = df_products['ProductName'].tolist()
value_dict = dict(zip(df_products['ProductName'], df_products['Value']))
weight_dict = dict(zip(df_products['ProductName'], df_products['Weight']))
product_index_to_name = {idx: name for idx, name in enumerate(product_names)}
product_name_to_index = {name: idx for idx, name in enumerate(product_names)}
m = gp.Model('AirConditionerAllocation')
x = m.addVars(storage_ids, range(len(product_names)), vtype=gp.GRB.INTEGER)
obj = gp.quicksum((value_dict[product_index_to_name[j]] * x[i, j] for i in storage_ids for j in range(len(product_names))))
m.setObjective(obj, gp.GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight_dict[product_index_to_name[j]] * x[i, j] for j in range(len(product_names)))) <= capacity_dict[i] for i in storage_ids))
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('\nAllocation of units per storage area:')
    for i in storage_ids:
        for j in range(len(product_names)):
            if x[i, j].X > 0:
                prod_name = product_index_to_name[j]
                print(f'  Storage {i} - {prod_name}: {int(x[i, j].X)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')