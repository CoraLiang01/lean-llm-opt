import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
capacity_df['CabinetID'] = capacity_df['CabinetID'].astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
products_df['ProductName'] = products_df['ProductName'].astype(str)
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(float)
cabinet_ids = capacity_df['CabinetID'].tolist()
num_products = len(products_df)
product_indices = list(range(num_products))
capacity = dict(zip(capacity_df['CabinetID'], capacity_df['Capacity']))
values = products_df['Value'].tolist()
weights = products_df['Weight'].tolist()
m = gp.Model('CoffeeCabinetAllocation')
x = m.addVars(cabinet_ids, product_indices, vtype=gp.GRB.INTEGER)
obj = gp.quicksum((values[j] * x[i, j] for i in cabinet_ids for j in product_indices))
m.setObjective(obj, gp.GRB.MAXIMIZE)
for i in cabinet_ids:
    m.addConstr(gp.quicksum((weights[j] * x[i, j] for j in product_indices)) <= capacity[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    for i in cabinet_ids:
        total_val = 0
        total_weight = 0
        print(f'Cabinet {i}:')
        for j in product_indices:
            qty = int(round(x[i, j].X))
            if qty > 0:
                prod_name = products_df.at[j, 'ProductName']
                val = values[j] * qty
                wt = weights[j] * qty
                total_val += val
                total_weight += wt
                print(f'  {prod_name} x {qty} (value {val}, weight {wt:.2f})')
        print(f'  Total weight: {total_weight:.2f} / {capacity[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')