import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv', sep=',', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain columns: 'ProductName', 'Value', 'Weight'.")
product_ids = products_df['ProductName'].tolist()
try:
    value_param = {pid: int(products_df.loc[products_df['ProductName'] == pid, 'Value'].values[0]) for pid in product_ids}
    weight_param = {pid: int(products_df.loc[products_df['ProductName'] == pid, 'Weight'].values[0]) for pid in product_ids}
except Exception as e:
    raise ValueError(f'Error extracting Value or Weight for products: {e}')
if 'Capacity' not in capacity_df.columns or len(capacity_df) != 1:
    raise ValueError("capacity.csv must contain exactly one row with column 'Capacity'.")
try:
    capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error extracting Capacity value: {e}')
for pid in product_ids:
    if not isinstance(value_param[pid], int) or not isinstance(weight_param[pid], int):
        raise ValueError(f'Product {pid} has non-integer Value or Weight.')
m = gp.Model('PharmacyRestock')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_param[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[pid] * x_vars[pid] for pid in product_ids)) <= capacity, name='capacity')
m.optimize()