import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv'
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if not {'ShelfID', 'Capacity'}.issubset(capacity_df.columns):
    raise KeyError("capacity.csv must contain columns 'ShelfID' and 'Capacity'.")
capacity_df['ShelfID'] = capacity_df['ShelfID'].str.strip().astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip().astype(float)
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError("products.csv must contain columns 'ProductName', 'Value', and 'Weight'.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip().astype(int)
products_df['Weight'] = products_df['Weight'].str.strip().astype(float)
shelf_ids = capacity_df['ShelfID'].tolist()
product_names = products_df['ProductName'].tolist()
capacity_dict = dict(zip(capacity_df['ShelfID'], capacity_df['Capacity']))
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(capacity_dict) != len(shelf_ids):
    raise ValueError('Mismatch in number of shelves and capacity entries.')
if len(value_dict) != len(product_names) or len(weight_dict) != len(product_names):
    raise ValueError('Mismatch in number of products and value/weight entries.')
m = gp.Model('RetailShelfAllocation')
x_vars = m.addVars(shelf_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in shelf_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in shelf_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_names)) <= capacity_dict[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Shelf Allocations (units per product per shelf) ---')
    for i in shelf_ids:
        shelf_total_value = 0
        shelf_total_weight = 0
        print(f'Shelf {i} (Capacity: {capacity_dict[i]}):')
        for j in product_names:
            units = x_vars[i, j].X
            if units > 1e-06:
                value = value_dict[j] * units
                weight = weight_dict[j] * units
                shelf_total_value += value
                shelf_total_weight += weight
                print(f'  {j}: {int(round(units))} units (Value: {value:.2f}, Weight: {weight:.2f})')
        print(f'  >> Shelf total value: {shelf_total_value:.2f}, total weight: {shelf_total_weight:.2f} / {capacity_dict[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')