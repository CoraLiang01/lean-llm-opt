import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv', dtype=str, keep_default_na=False)
if 'BookshelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError('Missing required columns in capacity.csv')
capacity_df['BookshelfID'] = capacity_df['BookshelfID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
if capacity_df['BookshelfID'].eq('').any() or capacity_df['Capacity'].eq('').any():
    raise ValueError('Blank BookshelfID or Capacity in capacity.csv')
capacity_df['BookshelfID'] = capacity_df['BookshelfID'].astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(float)
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError('Missing required columns in products.csv')
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
if products_df['ProductName'].eq('').any() or products_df['Value'].eq('').any() or products_df['Weight'].eq('').any():
    raise ValueError('Blank ProductName, Value, or Weight in products.csv')
products_df['Value'] = products_df['Value'].astype(float)
products_df['Weight'] = products_df['Weight'].astype(float)
bookshelf_ids = capacity_df['BookshelfID'].tolist()
product_names = products_df['ProductName'].tolist()
bookshelf_capacities = dict(zip(capacity_df['BookshelfID'], capacity_df['Capacity']))
product_values = dict(zip(products_df['ProductName'], products_df['Value']))
product_weights = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(bookshelf_capacities) != len(bookshelf_ids):
    raise ValueError('Duplicate or missing BookshelfID in capacity.csv')
if len(product_values) != len(product_names) or len(product_weights) != len(product_names):
    raise ValueError('Duplicate or missing ProductName in products.csv')
m = gp.Model('Bookstore_MultiKnapsack')
x_vars = m.addVars(bookshelf_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[j] * x_vars[i, j] for i in bookshelf_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in bookshelf_ids:
    m.addConstr(gp.quicksum((product_weights[j] * x_vars[i, j] for j in product_names)) <= bookshelf_capacities[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Book Allocation ---')
    for i in bookshelf_ids:
        shelf_total_value = 0.0
        shelf_total_weight = 0.0
        print(f'Bookshelf {i}:')
        for j in product_names:
            qty = x_vars[i, j].X
            if qty > 1e-06:
                value = product_values[j] * qty
                weight = product_weights[j] * qty
                shelf_total_value += value
                shelf_total_weight += weight
                print(f'  {j}: {int(round(qty))} units (Value: {value:.2f}, Weight: {weight:.2f})')
        print(f'  >> Total value: {shelf_total_value:.2f}, Total weight: {shelf_total_weight:.2f} / Capacity: {bookshelf_capacities[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')