import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if not {'PlatformId', 'Capacity'}.issubset(capacity_df.columns):
    raise KeyError('Missing required columns in capacity.csv')
capacity_df['PlatformId'] = capacity_df['PlatformId'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
platform_ids = capacity_df['PlatformId'].tolist()
if len(set(platform_ids)) != len(platform_ids):
    raise ValueError('Duplicate PlatformId found in capacity.csv')
platform_ids = [int(pid) for pid in platform_ids]
platform_capacities = dict(zip(platform_ids, capacity_df['Capacity'].astype(int)))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', sep=',', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError('Missing required columns in products.csv')
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_names = products_df['ProductName'].tolist()
if len(set(product_names)) != len(product_names):
    raise ValueError('Duplicate ProductName found in products.csv')
product_values = dict(zip(product_names, products_df['Value'].astype(int)))
product_weights = dict(zip(product_names, products_df['Weight'].astype(int)))
I = platform_ids
J = product_names
if not all((pid in platform_capacities for pid in I)):
    raise ValueError('Missing platform capacity for some PlatformId')
if not all((j in product_values and j in product_weights for j in J)):
    raise ValueError('Missing value or weight for some ProductName')
m = gp.Model('VideoGameStoreListing')
x_vars = m.addVars(I, J, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[j] * x_vars[i, j] for i in I for j in J)), gp.GRB.MAXIMIZE)
for i in I:
    m.addConstr(gp.quicksum((product_weights[j] * x_vars[i, j] for j in J)) <= platform_capacities[i], name=f'capacity_{i}')
m.optimize()