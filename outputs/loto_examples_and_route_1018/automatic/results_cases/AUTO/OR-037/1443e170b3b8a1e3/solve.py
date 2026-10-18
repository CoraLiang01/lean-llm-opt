import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
required_product_cols = ['ProductName', 'Value', 'Weight']
for col in required_product_cols:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
product_ids = products_df['ProductName'].tolist()
try:
    value_dict = dict(zip(product_ids, products_df['Value'].astype(int)))
    weight_dict = dict(zip(product_ids, products_df['Weight'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting Value or Weight to int: {e}')
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row')
try:
    total_capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to int: {e}')
m = gp.Model('CarSalesInventoryReplenishment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[pid] * x_vars[pid] for pid in product_ids)) <= total_capacity, name='InventoryCapacity')
m.optimize()