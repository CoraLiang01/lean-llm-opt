import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv', sep=',')
capacity_df['StorageID'] = capacity_df['StorageID'].astype(int)
products_df['ProductName'] = products_df['ProductName'].astype(str)
storage_ids = capacity_df['StorageID'].tolist()
product_names = products_df['ProductName'].tolist()
value = products_df.set_index('ProductName')['Value'].to_dict()
weight = products_df.set_index('ProductName')['Weight'].to_dict()
capacity = capacity_df.set_index('StorageID')['Capacity'].to_dict()
if set(product_names) != set(value.keys()) or set(product_names) != set(weight.keys()):
    raise ValueError('Mismatch in product names between index set and parameter dictionaries.')
if set(storage_ids) != set(capacity.keys()):
    raise ValueError('Mismatch in storage IDs between index set and capacity dictionary.')
m = gp.Model('Amazon_AC_Storage_Allocation')
x = m.addVars(storage_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in storage_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in storage_ids:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in product_names)) <= capacity[i], name=f'capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('--- Allocation Plan ---')
    for i in storage_ids:
        area_total_value = 0
        area_total_weight = 0
        allocation = []
        for j in product_names:
            units = int(round(x[i, j].X))
            if units > 0:
                allocation.append((j, units, value[j], weight[j]))
                area_total_value += value[j] * units
                area_total_weight += weight[j] * units
        if allocation:
            print(f'Storage Area {i}:')
            for j, units, v, w in allocation:
                print(f'  {j}: {units} units (Value/unit: {v}, Weight/unit: {w})')
            print(f'  Total Value: {area_total_value}, Total Weight: {area_total_weight}, Capacity: {capacity[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')