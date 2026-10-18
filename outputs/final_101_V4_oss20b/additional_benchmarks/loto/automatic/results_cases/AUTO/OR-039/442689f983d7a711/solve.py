import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
capacity_df['Warehouse ID'] = capacity_df['Warehouse ID'].astype(str)
warehouse_capacity = dict(zip(capacity_df['Warehouse ID'], capacity_df['Capacity']))
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv'
products_df = pd.read_csv(products_path, sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str)
product_value = dict(zip(products_df['ProductName'], products_df['Value']))
product_weight = dict(zip(products_df['ProductName'], products_df['Weight']))
products = list(product_value.keys())
warehouses = list(warehouse_capacity.keys())
m = gp.Model('WarehouseInventory')
x = m.addVars(products, warehouses, vtype=gp.GRB.INTEGER)
for w in warehouses:
    m.addConstr(gp.quicksum((product_weight[i] * x[i, w] for i in products)) <= warehouse_capacity[w])
m.setObjective(gp.quicksum((product_value[i] * x[i, w] for i in products for w in warehouses)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal}')
    print('--- Allocation per product and warehouse ---')
    for i in products:
        for w in warehouses:
            val = x[i, w].X
            if val > 0:
                print(f'Product "{i}" in Warehouse "{w}": {int(val)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')