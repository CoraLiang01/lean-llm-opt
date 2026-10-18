import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if 'BookshelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError('Missing required columns in capacity.csv')
capacity_df['BookshelfID'] = capacity_df['BookshelfID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
try:
    capacity_df['BookshelfID'] = capacity_df['BookshelfID'].astype(int)
    capacity_df['Capacity'] = capacity_df['Capacity'].astype(float)
except Exception as e:
    raise ValueError(f'Error converting BookshelfID or Capacity to numeric: {e}')
bookshelf_ids = capacity_df['BookshelfID'].tolist()
capacity_dict = dict(zip(capacity_df['BookshelfID'], capacity_df['Capacity']))
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError('Missing required columns in products.csv')
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
try:
    products_df['Value'] = products_df['Value'].astype(float)
    products_df['Weight'] = products_df['Weight'].astype(float)
except Exception as e:
    raise ValueError(f'Error converting Value or Weight to numeric: {e}')
book_names = products_df['ProductName'].tolist()
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(bookshelf_ids) == 0:
    raise ValueError('No bookshelves found in capacity.csv')
if len(book_names) == 0:
    raise ValueError('No products found in products.csv')
if set(capacity_df['BookshelfID']) != set(bookshelf_ids):
    raise ValueError('Mismatch in bookshelf IDs')
if set(products_df['ProductName']) != set(book_names):
    raise ValueError('Mismatch in product names')
m = gp.Model('Bookstore_MultiKnapsack')
x_vars = m.addVars(bookshelf_ids, book_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in bookshelf_ids for j in book_names)), gp.GRB.MAXIMIZE)
for i in bookshelf_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in book_names)) <= capacity_dict[i], name=f'capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Allocation per bookshelf ---')
    for i in bookshelf_ids:
        total_weight = 0.0
        total_value = 0.0
        print(f'Bookshelf {i}:')
        for j in book_names:
            qty = x_vars[i, j].X
            if qty > 1e-06:
                print(f'  {j}: {int(round(qty))} units (Value: {value_dict[j]}, Weight: {weight_dict[j]})')
                total_weight += qty * weight_dict[j]
                total_value += qty * value_dict[j]
        print(f'  >> Total weight: {total_weight:.2f} / {capacity_dict[i]}')
        print(f'  >> Total value: {total_value:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')