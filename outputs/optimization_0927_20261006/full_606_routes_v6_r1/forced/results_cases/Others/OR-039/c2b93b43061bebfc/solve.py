import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError('products.csv missing required columns.')
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
products = list(products_df['ProductName'])
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if not {'Warehouse ID', 'Capacity'}.issubset(capacity_df.columns):
    raise KeyError('capacity.csv missing required columns.')
capacity_df['Warehouse ID'] = capacity_df['Warehouse ID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
warehouses = list(capacity_df['Warehouse ID'])
capacity_dict = dict(zip(capacity_df['Warehouse ID'], capacity_df['Capacity']))
if len(products) == 0 or len(warehouses) == 0:
    raise ValueError('No products or warehouses found in input data.')
for p in products:
    if p not in value_dict or p not in weight_dict:
        raise KeyError(f"Missing value or weight for product '{p}'.")
for w in warehouses:
    if w not in capacity_dict:
        raise KeyError(f"Missing capacity for warehouse '{w}'.")
m = gp.Model('NewCarSalesInNorway_InventoryReplenishment')
x_vars = m.addVars(products, warehouses, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[p] * x_vars[p, w] for p in products for w in warehouses)), gp.GRB.MAXIMIZE)
for w in warehouses:
    m.addConstr(gp.quicksum((weight_dict[p] * x_vars[p, w] for p in products)) <= capacity_dict[w], name=f'capacity_{w}')
m.optimize()