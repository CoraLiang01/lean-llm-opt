import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError('products.csv must contain columns: ProductName, Value, Weight')
product_ids = products_df['ProductName'].tolist()
try:
    value_dict = dict(zip(product_ids, products_df['Value'].astype(int)))
    weight_dict = dict(zip(product_ids, products_df['Weight'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting Value or Weight columns to int: {e}')
if 'Capacity' not in capacity_df.columns:
    raise KeyError('capacity.csv must contain column: Capacity')
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row')
try:
    capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to int: {e}')
m = gp.Model('CarSalesInventory')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in product_ids)) <= capacity, name='inventory_capacity')
m.optimize()