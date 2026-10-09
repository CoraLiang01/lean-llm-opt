import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv', dtype=str, keep_default_na=False)
if 'ShelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'ShelfID' and 'Capacity' columns.")
shelf_ids = capacity_df['ShelfID'].astype(int).tolist()
capacity_dict = dict(zip(capacity_df['ShelfID'].astype(int), capacity_df['Capacity'].astype(int)))
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
product_ids = products_df['ProductName'].astype(int).tolist()
value_dict = dict(zip(products_df['ProductName'].astype(int), products_df['Value'].astype(int)))
weight_dict = dict(zip(products_df['ProductName'].astype(int), products_df['Weight'].astype(int)))
if set(capacity_dict.keys()) != set(shelf_ids):
    raise ValueError('Mismatch in shelf IDs between index set and capacity_dict.')
if set(value_dict.keys()) != set(product_ids) or set(weight_dict.keys()) != set(product_ids):
    raise ValueError('Mismatch in product IDs between index set and value/weight dicts.')
m = gp.Model('BigMartShelfAllocation')
x_vars = m.addVars(shelf_ids, product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in shelf_ids for j in product_ids)), gp.GRB.MAXIMIZE)
for i in shelf_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_ids)) <= capacity_dict[i], name=f'shelf_capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('--- Shelf Allocations (x[i,j]) ---')
    for i in shelf_ids:
        shelf_total_value = 0
        shelf_total_weight = 0
        print(f'Shelf {i}:')
        for j in product_ids:
            x_val = x_vars[i, j].X
            if x_val > 1e-06:
                print(f'  Product {j}: {int(round(x_val))} units (Value: {value_dict[j]}, Weight: {weight_dict[j]})')
                shelf_total_value += value_dict[j] * x_val
                shelf_total_weight += weight_dict[j] * x_val
        print(f'  >> Total value: {shelf_total_value:.0f}, Total weight: {shelf_total_weight:.0f} / Capacity: {capacity_dict[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')