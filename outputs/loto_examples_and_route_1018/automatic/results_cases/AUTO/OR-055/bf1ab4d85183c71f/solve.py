import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'DisplayID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'DisplayID' and 'Capacity' columns.")
capacity_df['DisplayID'] = capacity_df['DisplayID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
display_ids = list(capacity_df['DisplayID'])
capacity_dict = dict(zip(display_ids, capacity_df['Capacity'].astype(int)))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv', sep=',', dtype=str, keep_default_na=False)
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_names = list(products_df['ProductName'])
value_dict = dict(zip(product_names, products_df['Value'].astype(int)))
weight_dict = dict(zip(product_names, products_df['Weight'].astype(int)))
if len(display_ids) == 0:
    raise ValueError('No display areas found in capacity.csv.')
if len(product_names) == 0:
    raise ValueError('No products found in products.csv.')
m = gp.Model('BoatDisplayAllocation')
x_vars = m.addVars(display_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in display_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in display_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_names)) <= capacity_dict[i], name=f'capacity_{i}')
m.optimize()