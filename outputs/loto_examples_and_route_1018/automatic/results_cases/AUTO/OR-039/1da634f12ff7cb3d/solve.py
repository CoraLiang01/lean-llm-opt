import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain columns: 'ProductName', 'Value', 'Weight'.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
product_ids = products_df['ProductName'].tolist()
try:
    value_dict = dict(zip(products_df['ProductName'], products_df['Value'].astype(int)))
    weight_dict = dict(zip(products_df['ProductName'], products_df['Weight'].astype(int)))
except Exception as e:
    raise ValueError(f"Error converting 'Value' or 'Weight' to int in products.csv: {e}")
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if 'Warehouse ID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain columns: 'Warehouse ID', 'Capacity'.")
capacity_df['Warehouse ID'] = capacity_df['Warehouse ID'].str.strip()
warehouse_ids = capacity_df['Warehouse ID'].tolist()
try:
    capacity_dict = dict(zip(capacity_df['Warehouse ID'], capacity_df['Capacity'].astype(int)))
except Exception as e:
    raise ValueError(f"Error converting 'Capacity' to int in capacity.csv: {e}")
if len(product_ids) == 0 or len(warehouse_ids) == 0:
    raise ValueError('No products or warehouses found in the input files.')
m = gp.Model('CarWarehouseInventory')
x_vars = m.addVars(warehouse_ids, product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[w, i] for w in warehouse_ids for i in product_ids)), gp.GRB.MAXIMIZE)
for w in warehouse_ids:
    m.addConstr(gp.quicksum((weight_dict[i] * x_vars[w, i] for i in product_ids)) <= capacity_dict[w], name=f'capacity_{w}')
m.optimize()