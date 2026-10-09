import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
capacity_df['PlatformID'] = capacity_df['PlatformID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
platform_ids = capacity_df['PlatformID'].tolist()
if len(set(platform_ids)) != len(platform_ids):
    raise ValueError('Duplicate PlatformID found in capacity.csv')
platform_ids_set = set(platform_ids)
capacity_dict = {}
for (idx, row) in capacity_df.iterrows():
    pid = row['PlatformID']
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f'Invalid Capacity value for PlatformID {pid}')
    capacity_dict[pid] = cap
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv', sep=',', dtype=str, keep_default_na=False)
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_names = products_df['ProductName'].tolist()
if len(set(product_names)) != len(product_names):
    raise ValueError('Duplicate ProductName found in products.csv')
product_names_set = set(product_names)
value_dict = {}
weight_dict = {}
for (idx, row) in products_df.iterrows():
    pname = row['ProductName']
    try:
        val = int(row['Value'])
        wt = int(row['Weight'])
    except Exception:
        raise ValueError(f'Invalid Value or Weight for ProductName {pname}')
    value_dict[pname] = val
    weight_dict[pname] = wt
I = platform_ids
J = product_names
if set(capacity_dict.keys()) != set(I):
    raise ValueError('Mismatch in PlatformID between index set and capacity_dict')
if set(value_dict.keys()) != set(J) or set(weight_dict.keys()) != set(J):
    raise ValueError('Mismatch in ProductName between index set and value/weight dicts')
m = gp.Model('VideoGameStoreListing')
x_vars = m.addVars(I, J, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in I for j in J)), gp.GRB.MAXIMIZE)
for i in I:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in J)) <= capacity_dict[i], name=f'capacity_{i}')
m.optimize()