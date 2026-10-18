import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv', dtype=str, keep_default_na=False)
if 'ShelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'ShelfID' and 'Capacity' columns.")
shelf_ids = capacity_df['ShelfID'].apply(lambda x: int(x.strip())).tolist()
capacity_df['ShelfID'] = capacity_df['ShelfID'].apply(lambda x: int(x.strip()))
capacity_df['Capacity'] = capacity_df['Capacity'].apply(lambda x: float(x.strip()))
shelf_capacities = dict(zip(capacity_df['ShelfID'], capacity_df['Capacity']))
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
product_names = products_df['ProductName'].apply(lambda x: x.strip()).tolist()
products_df['ProductName'] = products_df['ProductName'].apply(lambda x: x.strip())
products_df['Value'] = products_df['Value'].apply(lambda x: int(x.strip()))
products_df['Weight'] = products_df['Weight'].apply(lambda x: float(x.strip()))
product_values = dict(zip(products_df['ProductName'], products_df['Value']))
product_weights = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(shelf_ids) != set(shelf_capacities.keys()):
    raise ValueError('Mismatch in shelf IDs between index set and capacity data.')
if set(product_names) != set(product_values.keys()) or set(product_names) != set(product_weights.keys()):
    raise ValueError('Mismatch in product names between index set and value/weight data.')
m = gp.Model('RetailShelfAllocation')
x_vars = m.addVars(shelf_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[j] * x_vars[i, j] for i in shelf_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in shelf_ids:
    m.addConstr(gp.quicksum((product_weights[j] * x_vars[i, j] for j in product_names)) <= shelf_capacities[i], name=f'shelf_capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Shelf Allocations (units per product per shelf) ---')
    for i in shelf_ids:
        shelf_total_value = 0
        shelf_total_weight = 0
        print(f'Shelf {i}:')
        for j in product_names:
            units = x_vars[i, j].X
            if units > 1e-06:
                value = product_values[j] * units
                weight = product_weights[j] * units
                print(f'  {j}: {units:.0f} units (Value: {value:.2f}, Weight: {weight:.2f})')
                shelf_total_value += value
                shelf_total_weight += weight
        print(f'  >> Total value: {shelf_total_value:.2f}, Total weight: {shelf_total_weight:.2f} / Capacity: {shelf_capacities[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')