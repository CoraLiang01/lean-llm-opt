import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if not {'PlatformId', 'Capacity'}.issubset(capacity_df.columns):
    raise KeyError('capacity.csv must contain columns: PlatformId, Capacity')
capacity_df['PlatformId'] = capacity_df['PlatformId'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
platform_ids = capacity_df['PlatformId'].tolist()
if len(set(platform_ids)) != len(platform_ids):
    raise ValueError('PlatformId values in capacity.csv must be unique')
platform_capacities = {}
for (idx, row) in capacity_df.iterrows():
    pid = row['PlatformId']
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for PlatformId {pid}: {row['Capacity']}")
    platform_capacities[pid] = cap
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv', sep=',', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError('products.csv must contain columns: ProductName, Value, Weight')
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_names = products_df['ProductName'].tolist()
if len(set(product_names)) != len(product_names):
    raise ValueError('ProductName values in products.csv must be unique')
product_values = {}
product_weights = {}
for (idx, row) in products_df.iterrows():
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
I = platform_ids
J = product_names
m = gp.Model('VideoGameStoreListing')
x_vars = m.addVars(I, J, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[j] * x_vars[i, j] for i in I for j in J)), gp.GRB.MAXIMIZE)
for i in I:
    m.addConstr(gp.quicksum((product_weights[j] * x_vars[i, j] for j in J)) <= platform_capacities[i], name=f'capacity_{i}')
m.optimize()