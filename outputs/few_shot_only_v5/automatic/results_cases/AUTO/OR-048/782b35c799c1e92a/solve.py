import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv', dtype=str, keep_default_na=False)
capacity_df['StorageID'] = capacity_df['StorageID'].astype(str).str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
storage_ids = list(capacity_df['StorageID'])
capacity = dict(zip(capacity_df['StorageID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv', dtype=str, keep_default_na=False)
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
product_names = list(products_df['ProductName'])
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(storage_ids) != set(capacity.keys()):
    raise ValueError('Mismatch in storage IDs and capacity keys')
if set(product_names) != set(value.keys()) or set(product_names) != set(weight.keys()):
    raise ValueError('Mismatch in product names and value/weight keys')
m = gp.Model('Amazon_AC_Storage_Allocation')
x_vars = m.addVars(storage_ids, product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in storage_ids for j in product_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x_vars[i, j] for j in product_names)) <= capacity[i] for i in storage_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')