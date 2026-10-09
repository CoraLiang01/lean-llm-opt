import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv', dtype=str, keep_default_na=False)
product_ids = products_df['ProductName'].tolist()
weight_param = {}
value_param = {}
for (idx, row) in products_df.iterrows():
    pid = str(row['ProductName'])
    try:
        weight_param[pid] = int(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Weight for product '{pid}': {row['Weight']}")
    try:
        value_param[pid] = int(row['Value'])
    except Exception:
        raise ValueError(f"Invalid Value for product '{pid}': {row['Value']}")
if capacity_df.shape[0] != 1:
    raise ValueError('capacity.csv must contain exactly one row with the total capacity.')
try:
    total_capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception:
    raise ValueError(f"Invalid Capacity value: {capacity_df.iloc[0]['Capacity']}")
m = gp.Model('SupermarketRestockKnapsack')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_param[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[pid] * x_vars[pid] for pid in product_ids)) <= total_capacity, name='TotalWeight')
m.optimize()