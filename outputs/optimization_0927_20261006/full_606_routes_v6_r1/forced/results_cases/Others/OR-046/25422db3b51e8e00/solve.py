import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
required_product_cols = ['ProductName', 'Weight', 'Value']
for col in required_product_cols:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
product_ids = products_df['ProductName'].tolist()
try:
    weights = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
    values = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'Weight' or 'Value' columns to int: {e}")
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row')
try:
    capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f"Error converting 'Capacity' to int: {e}")
for pid in product_ids:
    if pid not in weights or pid not in values:
        raise KeyError(f"Missing weight or value for product '{pid}'")
m = gp.Model('SupermarketStockReplenishment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[pid] * x_vars[pid] for pid in product_ids)) <= capacity, name='StockCapacity')
m.optimize()