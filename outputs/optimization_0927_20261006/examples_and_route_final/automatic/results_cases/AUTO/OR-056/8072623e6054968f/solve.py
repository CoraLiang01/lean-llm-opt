import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
capacity_df['DisplayID'] = capacity_df['DisplayID'].str.strip().astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip().astype(int)
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip().astype(int)
products_df['Weight'] = products_df['Weight'].str.strip().astype(int)
display_ids = capacity_df['DisplayID'].tolist()
product_names = products_df['ProductName'].tolist()
capacity_dict = dict(zip(capacity_df['DisplayID'], capacity_df['Capacity']))
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(display_ids) != set(capacity_dict.keys()):
    raise ValueError('Mismatch between display_ids and capacity_dict keys.')
if set(product_names) != set(value_dict.keys()) or set(product_names) != set(weight_dict.keys()):
    raise ValueError('Mismatch between product_names and value/weight dict keys.')
m = gp.Model('BoatDisplayAssignment')
x_vars = m.addVars(display_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in display_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in display_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_names)) <= capacity_dict[i], name=f'capacity_{i}')
m.optimize()