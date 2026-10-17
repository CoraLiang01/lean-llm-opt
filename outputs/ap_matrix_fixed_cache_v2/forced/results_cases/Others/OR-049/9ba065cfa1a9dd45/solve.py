import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if not {'ShelfID', 'Capacity'}.issubset(capacity_df.columns):
    raise KeyError("capacity.csv must contain columns 'ShelfID' and 'Capacity'.")
capacity_df['ShelfID'] = capacity_df['ShelfID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
try:
    capacity_df['ShelfID'] = capacity_df['ShelfID'].astype(int)
    capacity_df['Capacity'] = capacity_df['Capacity'].astype(float)
except Exception as e:
    raise ValueError(f'Error converting ShelfID or Capacity to numeric: {e}')
shelves = list(capacity_df['ShelfID'])
shelf_capacity = dict(zip(capacity_df['ShelfID'], capacity_df['Capacity']))
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError("products.csv must contain columns 'ProductName', 'Value', and 'Weight'.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
try:
    products_df['Value'] = products_df['Value'].astype(int)
    products_df['Weight'] = products_df['Weight'].astype(float)
except Exception as e:
    raise ValueError(f'Error converting Value or Weight to numeric: {e}')
products = list(products_df['ProductName'])
product_value = dict(zip(products_df['ProductName'], products_df['Value']))
product_weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(shelves) == 0:
    raise ValueError('No shelves found in capacity.csv.')
if len(products) == 0:
    raise ValueError('No products found in products.csv.')
if set(shelf_capacity.keys()) != set(shelves):
    raise ValueError('Mismatch in shelf IDs and capacity mapping.')
if set(product_value.keys()) != set(products) or set(product_weight.keys()) != set(products):
    raise ValueError('Mismatch in product keys and value/weight mapping.')
m = gp.Model('RetailShelfAllocation')
x_vars = m.addVars(shelves, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_value[j] * x_vars[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((product_weight[j] * x_vars[i, j] for j in products)) <= shelf_capacity[i], name=f'shelf_capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Shelf Allocations (units per product per shelf) ---')
    for i in shelves:
        shelf_total_value = 0
        shelf_total_weight = 0
        print(f'Shelf {i} (Capacity: {shelf_capacity[i]:.2f}):')
        for j in products:
            units = x_vars[i, j].X
            if units > 1e-06:
                value = product_value[j] * units
                weight = product_weight[j] * units
                shelf_total_value += value
                shelf_total_weight += weight
                print(f'  {j}: {units:.0f} units (Value: {value:.2f}, Weight: {weight:.2f})')
        print(f'  >> Shelf total value: {shelf_total_value:.2f}, total weight: {shelf_total_weight:.2f} / {shelf_capacity[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')