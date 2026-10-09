import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', sep=',')
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
capacity_df['Warehouse ID'] = capacity_df['Warehouse ID'].astype(str).str.strip()
products = list(products_df['ProductName'])
warehouses = list(capacity_df['Warehouse ID'])
value = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
weight = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
capacity = capacity_df.set_index('Warehouse ID')['Capacity'].astype(int).to_dict()
if set(products) != set(value.keys()) or set(products) != set(weight.keys()):
    raise ValueError('Mismatch in product identifiers between products and value/weight columns.')
if set(warehouses) != set(capacity.keys()):
    raise ValueError('Mismatch in warehouse identifiers between capacity file and warehouse list.')
m = gp.Model('CarInventoryMultiKnapsack')
x = m.addVars(products, warehouses, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[i] * x[i, w] for i in products for w in warehouses)), gp.GRB.MAXIMIZE)
for w in warehouses:
    m.addConstr(gp.quicksum((weight[i] * x[i, w] for i in products)) <= capacity[w], name=f'cap_{w}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('\n--- Inventory Plan (number of vehicles per type per warehouse) ---')
    for w in warehouses:
        print(f'\nWarehouse: {w} (Capacity: {capacity[w]})')
        total_weight = 0
        total_value = 0
        for i in products:
            qty = int(round(x[i, w].X))
            if qty > 0:
                print(f'  {i}: {qty} units (Value: {value[i]}, Weight: {weight[i]})')
                total_weight += qty * weight[i]
                total_value += qty * value[i]
        print(f'  >> Total weight used: {total_weight} / {capacity[w]}')
        print(f'  >> Total value in warehouse: {total_value}')
else:
    print(f'No optimal solution found. Status: {m.status}')