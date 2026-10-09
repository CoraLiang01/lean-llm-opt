import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv', sep=',', dtype=str, keep_default_na=False)
capacity_df['StorageID'] = capacity_df['StorageID'].astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv', sep=',', dtype=str, keep_default_na=False)
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
storage_ids = capacity_df['StorageID'].tolist()
product_names = products_df['ProductName'].tolist()
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
capacity_dict = dict(zip(capacity_df['StorageID'], capacity_df['Capacity']))
if len(storage_ids) != len(set(storage_ids)):
    raise ValueError('Duplicate StorageID found in capacity.csv')
if len(product_names) != len(set(product_names)):
    raise ValueError('Duplicate ProductName found in products.csv')
if set(value_dict.keys()) != set(product_names) or set(weight_dict.keys()) != set(product_names):
    raise ValueError('Mismatch in product parameter keys')
if set(capacity_dict.keys()) != set(storage_ids):
    raise ValueError('Mismatch in storage parameter keys')
m = gp.Model('Amazon_AirConditioner_Allocation')
x_vars = m.addVars(storage_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in storage_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in storage_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_names)) <= capacity_dict[i], name=f'capacity_{i}')
m.optimize()