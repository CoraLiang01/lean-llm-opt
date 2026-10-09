import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv', dtype=str, keep_default_na=False)
if 'ShelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'ShelfID' and 'Capacity' columns.")
shelf_ids = capacity_df['ShelfID'].astype(str).str.strip()
if shelf_ids.duplicated().any():
    raise ValueError('Duplicate ShelfID found in capacity.csv.')
shelf_ids = shelf_ids.tolist()
capacity_dict = dict(zip(shelf_ids, capacity_df['Capacity'].astype(float)))
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
product_ids = products_df['ProductName'].astype(str).str.strip()
if product_ids.duplicated().any():
    raise ValueError('Duplicate ProductName found in products.csv.')
product_ids = product_ids.tolist()
value_dict = dict(zip(product_ids, products_df['Value'].astype(float)))
weight_dict = dict(zip(product_ids, products_df['Weight'].astype(float)))
m = gp.Model('BigMart_MultiShelf_Knapsack')
x_vars = m.addVars(shelf_ids, product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in shelf_ids for j in product_ids)), gp.GRB.MAXIMIZE)
for i in shelf_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_ids)) <= capacity_dict[i], name=f'shelf_capacity_{i}')
m.optimize()