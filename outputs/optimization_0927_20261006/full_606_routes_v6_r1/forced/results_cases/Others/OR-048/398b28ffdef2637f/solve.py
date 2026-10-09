import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'StorageID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'StorageID' and 'Capacity' columns.")
capacity_df['StorageID'] = capacity_df['StorageID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
storage_ids = capacity_df['StorageID'].tolist()
if len(set(storage_ids)) != len(storage_ids):
    raise ValueError('Duplicate StorageID found in capacity.csv.')
storage_ids_int = [int(sid) for sid in storage_ids]
storage_id_map = dict(zip(storage_ids_int, storage_ids))
capacity_dict = {}
for (idx, row) in capacity_df.iterrows():
    sid = row['StorageID']
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for StorageID {sid}: {row['Capacity']}")
    capacity_dict[sid] = cap
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv', sep=',', dtype=str, keep_default_na=False)
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_names = products_df['ProductName'].tolist()
if len(set(product_names)) != len(product_names):
    raise ValueError('Duplicate ProductName found in products.csv.')
value_dict = {}
weight_dict = {}
for (idx, row) in products_df.iterrows():
    pname = row['ProductName']
    try:
        val = int(row['Value'])
        wt = int(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Value or Weight for ProductName {pname}: Value={row['Value']}, Weight={row['Weight']}")
    value_dict[pname] = val
    weight_dict[pname] = wt
I = storage_ids
J = product_names
for sid in I:
    if sid not in capacity_dict:
        raise KeyError(f'Missing capacity for StorageID {sid}')
for pname in J:
    if pname not in value_dict or pname not in weight_dict:
        raise KeyError(f'Missing value or weight for ProductName {pname}')
m = gp.Model('Amazon_AirConditioner_Storage_Allocation')
x_vars = m.addVars(I, J, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in I for j in J)), gp.GRB.MAXIMIZE)
for i in I:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in J)) <= capacity_dict[i])
m.optimize()