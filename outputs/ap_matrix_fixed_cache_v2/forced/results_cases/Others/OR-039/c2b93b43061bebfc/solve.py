import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_str(s):
    return re.sub('\\s+', ' ', str(s).strip()).casefold()
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', dtype=str, keep_default_na=False)
warehouses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', dtype=str, keep_default_na=False)
products_df['ProductName_norm'] = products_df['ProductName'].apply(norm_str)
warehouses_df['WarehouseID_norm'] = warehouses_df['Warehouse ID'].apply(norm_str)
products = list(products_df['ProductName'])
products_norm = list(products_df['ProductName_norm'])
warehouses = list(warehouses_df['Warehouse ID'])
warehouses_norm = list(warehouses_df['WarehouseID_norm'])
try:
    value_dict = dict(zip(products_norm, products_df['Value'].astype(int)))
    weight_dict = dict(zip(products_norm, products_df['Weight'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting Value or Weight columns to int: {e}')
try:
    capacity_dict = dict(zip(warehouses_norm, warehouses_df['Capacity'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting Capacity column to int: {e}')
if len(value_dict) != len(products):
    raise ValueError('Mismatch in number of products and value_dict entries.')
if len(weight_dict) != len(products):
    raise ValueError('Mismatch in number of products and weight_dict entries.')
if len(capacity_dict) != len(warehouses):
    raise ValueError('Mismatch in number of warehouses and capacity_dict entries.')
m = gp.Model('Car_Inventory_Replenishment')
x_vars = m.addVars(products_norm, warehouses_norm, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i, w] for i in products_norm for w in warehouses_norm)), gp.GRB.MAXIMIZE)
for w, w_norm in zip(warehouses, warehouses_norm):
    m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i, w_norm] for i in products_norm)) <= capacity_dict[w_norm], name=f'capacity_{w_norm}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('\n--- Inventory Plan (number of vehicles per warehouse) ---')
    for w, w_norm in zip(warehouses, warehouses_norm):
        print(f'\nWarehouse: {w} (Capacity: {capacity_dict[w_norm]})')
        total_weight = 0
        total_value = 0
        for i, i_norm in zip(products, products_norm):
            qty = int(round(x_vars[i_norm, w_norm].X))
            if qty > 0:
                weight = weight_dict[i_norm] * qty
                value = value_dict[i_norm] * qty
                print(f'  {i}: {qty} units (Weight: {weight}, Value: {value})')
                total_weight += weight
                total_value += value
        print(f'  >> Total weight used: {total_weight} / {capacity_dict[w_norm]}')
        print(f'  >> Total value in warehouse: {total_value}')
else:
    print(f'No optimal solution found. Status: {m.status}')