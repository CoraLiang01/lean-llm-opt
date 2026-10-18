import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'StorageID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'StorageID' and 'Capacity' columns.")
capacity_df['StorageID'] = capacity_df['StorageID'].str.strip().astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip().astype(int)
storage_ids = capacity_df['StorageID'].tolist()
capacity_dict = dict(zip(capacity_df['StorageID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv', sep=',', dtype=str, keep_default_na=False)
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip().astype(int)
products_df['Weight'] = products_df['Weight'].str.strip().astype(int)
product_names = products_df['ProductName'].tolist()
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(storage_ids) == 0 or len(product_names) == 0:
    raise ValueError('No storage areas or products found in the input files.')
m = gp.Model('Amazon_AirConditioner_Allocation')
x_vars = m.addVars(storage_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in storage_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in storage_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_names)) <= capacity_dict[i], name=f'capacity_{i}')
m.optimize()