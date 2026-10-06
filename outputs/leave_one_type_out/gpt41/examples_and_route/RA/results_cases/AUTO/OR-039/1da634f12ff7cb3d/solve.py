import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
product_ids = products_df['ProductName'].tolist()
value = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
weight = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
capacity_df = pd.read_csv(capacity_path, sep=',')
capacity_df['Warehouse ID'] = capacity_df['Warehouse ID'].astype(str).str.strip()
warehouse_ids = capacity_df['Warehouse ID'].tolist()
capacity = capacity_df.set_index('Warehouse ID')['Capacity'].astype(int).to_dict()
if set(product_ids) != set(value.keys()) or set(product_ids) != set(weight.keys()):
    raise ValueError('Mismatch in product identifiers between products.csv columns.')
if set(warehouse_ids) != set(capacity.keys()):
    raise ValueError('Mismatch in warehouse identifiers between capacity.csv columns.')
m = gp.Model('CarInventoryAllocation')
x = m.addVars(warehouse_ids, product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[i] * x[w, i] for w in warehouse_ids for i in product_ids)), gp.GRB.MAXIMIZE)
for w in warehouse_ids:
    m.addConstr(gp.quicksum((weight[i] * x[w, i] for i in product_ids)) <= capacity[w], name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('\n--- Allocation Plan (number of vehicles per warehouse and type) ---')
    for w in warehouse_ids:
        warehouse_total_value = 0
        warehouse_total_weight = 0
        print(f'\nWarehouse: {w} (Capacity: {capacity[w]})')
        for i in product_ids:
            xi = int(round(x[w, i].X))
            if xi > 0:
                print(f'  {i}: {xi} units (Value: {value[i]}, Weight: {weight[i]})')
                warehouse_total_value += value[i] * xi
                warehouse_total_weight += weight[i] * xi
        print(f'  >> Total value in warehouse: {warehouse_total_value}')
        print(f'  >> Total weight in warehouse: {warehouse_total_weight} / {capacity[w]}')
else:
    print(f'No optimal solution found. Status: {m.status}')