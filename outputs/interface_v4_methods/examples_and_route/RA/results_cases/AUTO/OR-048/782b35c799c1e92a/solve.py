import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
if capacity_df.isnull().any().any():
    raise ValueError('Missing values detected in capacity.csv')
capacity_df['StorageID'] = capacity_df['StorageID'].astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
storage_ids = capacity_df['StorageID'].tolist()
storage_capacities = dict(zip(capacity_df['StorageID'], capacity_df['Capacity']))
products_df = pd.read_csv(products_path, sep=',')
if products_df.isnull().any().any():
    raise ValueError('Missing values detected in products.csv')
products_df['ProductName'] = products_df['ProductName'].astype(str)
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
product_names = products_df['ProductName'].tolist()
product_values = dict(zip(products_df['ProductName'], products_df['Value']))
product_weights = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(storage_ids) == 0 or len(product_names) == 0:
    raise ValueError('No storage areas or products found in the input files.')
m = gp.Model('Amazon_AC_Storage_Allocation')
x = m.addVars(storage_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[j] * x[i, j] for i in storage_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in storage_ids:
    m.addConstr(gp.quicksum((product_weights[j] * x[i, j] for j in product_names)) <= storage_capacities[i], name=f'capacity_{i}')
m.optimize()