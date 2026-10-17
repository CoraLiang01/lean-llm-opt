import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', dtype=str, keep_default_na=False)
if 'PlatformId' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError('Missing required columns in capacity.csv')
capacity_df['PlatformId'] = capacity_df['PlatformId'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
platform_ids = list(capacity_df['PlatformId'])
platform_capacities = {}
for idx, row in capacity_df.iterrows():
    pid = row['PlatformId']
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for PlatformId {pid}: {row['Capacity']}")
    platform_capacities[pid] = cap
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', dtype=str, keep_default_na=False)
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError('Missing required columns in products.csv')
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_names = list(products_df['ProductName'])
product_values = {}
product_weights = {}
for idx, row in products_df.iterrows():
    pname = row['ProductName']
    try:
        val = int(row['Value'])
    except Exception:
        raise ValueError(f"Invalid Value for ProductName {pname}: {row['Value']}")
    try:
        wt = int(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Weight for ProductName {pname}: {row['Weight']}")
    product_values[pname] = val
    product_weights[pname] = wt
if set(platform_ids) != set(capacity_df['PlatformId']):
    raise ValueError('PlatformId mismatch in capacity data')
if set(product_names) != set(products_df['ProductName']):
    raise ValueError('ProductName mismatch in products data')
m = gp.Model('VideoGameStoreListing')
x_vars = m.addVars(platform_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[j] * x_vars[i, j] for i in platform_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in platform_ids:
    m.addConstr(gp.quicksum((product_weights[j] * x_vars[i, j] for j in product_names)) <= platform_capacities[i])
m.optimize()