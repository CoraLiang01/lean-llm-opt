import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv', sep=',')
capacity_df['StorageID'] = capacity_df['StorageID'].astype(int)
storage_ids = capacity_df['StorageID'].tolist()
capacity_dict = dict(zip(capacity_df['StorageID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv', sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
product_names = products_df['ProductName'].tolist()
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(storage_ids) == 0:
    raise ValueError('No storage areas found in capacity.csv')
if len(product_names) == 0:
    raise ValueError('No products found in products.csv')
if set(capacity_dict.keys()) != set(storage_ids):
    raise ValueError('Mismatch in storage area identifiers')
if set(value_dict.keys()) != set(product_names) or set(weight_dict.keys()) != set(product_names):
    raise ValueError('Mismatch in product identifiers')
m = gp.Model('Amazon_AirConditioner_Allocation')
x = m.addVars(storage_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in storage_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in storage_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in product_names)) <= capacity_dict[i], name=f'capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('--- Allocation Plan ---')
    for i in storage_ids:
        area_output = []
        for j in product_names:
            units = x[i, j].X
            if units >= 1e-06:
                area_output.append((j, int(round(units))))
        if area_output:
            print(f'Storage Area {i}:')
            for (prod, qty) in area_output:
                print(f'  {prod}: {qty}')
else:
    print(f'No optimal solution found. Status: {m.status}')