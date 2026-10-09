import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'PlatformID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'PlatformID' and 'Capacity' columns.")
capacity_df['PlatformID'] = capacity_df['PlatformID'].astype(str).str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
platform_ids = capacity_df['PlatformID'].tolist()
platform_capacities = dict(zip(capacity_df['PlatformID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv', sep=',', dtype=str, keep_default_na=False)
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
product_names = products_df['ProductName'].tolist()
product_values = dict(zip(products_df['ProductName'], products_df['Value']))
product_weights = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(platform_ids) == 0 or len(product_names) == 0:
    raise ValueError('No platforms or products found in the input data.')
m = gp.Model('GamePlatformListing')
x_vars = m.addVars(platform_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[prod] * x_vars[plat, prod] for plat in platform_ids for prod in product_names)), gp.GRB.MAXIMIZE)
for plat in platform_ids:
    m.addConstr(gp.quicksum((product_weights[prod] * x_vars[plat, prod] for prod in product_names)) <= platform_capacities[plat], name=f'capacity_{plat}')
m.optimize()