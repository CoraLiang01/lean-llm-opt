import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if capacity_df.shape[0] != 1:
    raise ValueError('capacity.csv must have exactly one row')
try:
    capacity = int(capacity_df.loc[0, 'Capacity'])
except Exception as e:
    raise ValueError(f'Could not convert capacity value to int: {e}')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', sep=',', dtype=str, keep_default_na=False)
for col in ['ProductName', 'Value', 'Weight']:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
product_ids = products_df['ProductName'].tolist()
if len(set(product_ids)) != len(product_ids):
    raise ValueError('Duplicate ProductName entries found in products.csv')
try:
    value_param = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
    weight_param = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f'Could not convert Value or Weight to int: {e}')
if set(value_param.keys()) != set(product_ids) or set(weight_param.keys()) != set(product_ids):
    raise ValueError('Mismatch in product identifiers between Value/Weight and ProductName')
m = gp.Model('BakeryBreadOrder')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_param[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[i] * x_vars[i] for i in product_ids)) <= capacity, name='storage_capacity')
m.optimize()