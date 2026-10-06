import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv', sep=',')
capacity_df['StorageID'] = capacity_df['StorageID'].astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv', sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str)
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
storage_ids = capacity_df['StorageID'].tolist()
product_names = products_df['ProductName'].tolist()
value = products_df.set_index('ProductName')['Value'].to_dict()
weight = products_df.set_index('ProductName')['Weight'].to_dict()
capacity = capacity_df.set_index('StorageID')['Capacity'].to_dict()
if len(storage_ids) != len(capacity):
    raise ValueError('Mismatch in number of storage areas and capacity entries.')
if len(product_names) != len(value) or len(product_names) != len(weight):
    raise ValueError('Mismatch in number of products and value/weight entries.')
m = gp.Model('Amazon_AC_Storage_Allocation')
x = m.addVars(storage_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in storage_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in storage_ids:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in product_names)) <= capacity[i], name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('--- Allocation Plan (nonzero only) ---')
    for i in storage_ids:
        for j in product_names:
            v = x[i, j].X
            if v > 1e-06:
                print(f"Storage Area {i}, Product '{j}': {int(round(v))} units")
else:
    print(f'No optimal solution found. Status: {m.status}')