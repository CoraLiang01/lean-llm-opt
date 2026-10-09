import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv', dtype=str, keep_default_na=False)
storage_ids = capacity_df['StorageID'].astype(int).tolist()
capacity_dict = {}
for (idx, row) in capacity_df.iterrows():
    sid = int(row['StorageID'])
    if sid in capacity_dict:
        raise ValueError(f'Duplicate StorageID found: {sid}')
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for StorageID {sid}: {row['Capacity']}")
    capacity_dict[sid] = cap
if set(storage_ids) != set(capacity_dict.keys()):
    raise ValueError('Mismatch in StorageID keys after parsing.')
product_names = products_df['ProductName'].tolist()
value_dict = {}
weight_dict = {}
for (idx, row) in products_df.iterrows():
    pname = row['ProductName']
    if pname in value_dict or pname in weight_dict:
        raise ValueError(f'Duplicate ProductName found: {pname}')
    try:
        val = int(row['Value'])
        wt = int(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Value or Weight for ProductName {pname}: Value={row['Value']}, Weight={row['Weight']}")
    value_dict[pname] = val
    weight_dict[pname] = wt
if set(product_names) != set(value_dict.keys()) or set(product_names) != set(weight_dict.keys()):
    raise ValueError('Mismatch in ProductName keys after parsing.')
decision_keys = [(i, j) for i in storage_ids for j in product_names]
m = gp.Model('Amazon_AC_Storage_Allocation')
quantity_vars = m.addVars(decision_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * quantity_vars[i, j] for (i, j) in decision_keys)), gp.GRB.MAXIMIZE)
for i in storage_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * quantity_vars[i, j] for j in product_names)) <= capacity_dict[i], name='cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for (i, j) in decision_keys:
        var = quantity_vars[i, j]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')