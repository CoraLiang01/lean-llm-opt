import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'PlatformID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'PlatformID' and 'Capacity' columns.")
capacity_df['PlatformID'] = capacity_df['PlatformID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
platform_ids = capacity_df['PlatformID'].tolist()
if len(set(platform_ids)) != len(platform_ids):
    raise ValueError('PlatformID values in capacity.csv must be unique.')
platform_id_ints = [int(pid) for pid in platform_ids]
platform_id_map = dict(zip(platform_ids, platform_id_ints))
capacity_dict = dict(zip(platform_ids, capacity_df['Capacity'].astype(int)))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv', sep=',', dtype=str, keep_default_na=False)
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_names = products_df['ProductName'].tolist()
if len(set(product_names)) != len(product_names):
    raise ValueError('ProductName values in products.csv must be unique.')
value_dict = dict(zip(product_names, products_df['Value'].astype(int)))
weight_dict = dict(zip(product_names, products_df['Weight'].astype(int)))
I = platform_ids
J = product_names
for i in I:
    if i not in capacity_dict:
        raise KeyError(f'Missing capacity for PlatformID {i}')
for j in J:
    if j not in value_dict or j not in weight_dict:
        raise KeyError(f'Missing value or weight for ProductName {j}')
m = gp.Model('VideoGameStoreListing')
x_vars = m.addVars(I, J, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in I for j in J)), gp.GRB.MAXIMIZE)
for i in I:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in J)) <= capacity_dict[i], name=f'capacity_{i}')
m.optimize()