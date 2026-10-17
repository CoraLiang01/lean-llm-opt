import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if not {'PlatformID', 'Capacity'}.issubset(capacity_df.columns):
    raise KeyError('capacity.csv must contain columns: PlatformID, Capacity')
capacity_df['PlatformID'] = capacity_df['PlatformID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
platform_ids = capacity_df['PlatformID'].tolist()
if len(set(platform_ids)) != len(platform_ids):
    raise ValueError('PlatformID values in capacity.csv must be unique')
platform_ids_set = set(platform_ids)
capacity_dict = dict(zip(platform_ids, capacity_df['Capacity'].astype(int)))
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError('products.csv must contain columns: ProductName, Value, Weight')
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_names = products_df['ProductName'].tolist()
if len(set(product_names)) != len(product_names):
    raise ValueError('ProductName values in products.csv must be unique')
product_names_set = set(product_names)
value_dict = dict(zip(product_names, products_df['Value'].astype(int)))
weight_dict = dict(zip(product_names, products_df['Weight'].astype(int)))
m = gp.Model('VideoGameStoreListing')
x_vars = m.addVars(platform_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in platform_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in platform_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_names)) <= capacity_dict[i], name=f'capacity_{i}')
m.optimize()