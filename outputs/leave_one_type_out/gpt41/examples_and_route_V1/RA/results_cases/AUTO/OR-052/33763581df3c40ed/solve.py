import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
bookshelf_ids = capacity_df['BookshelfID'].astype(int).tolist()
capacity_dict = dict(zip(capacity_df['BookshelfID'].astype(int), capacity_df['Capacity'].astype(int)))
book_names = products_df['ProductName'].astype(str).tolist()
value_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(int)))
weight_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(int)))
if len(bookshelf_ids) != len(capacity_dict):
    raise ValueError('Mismatch in bookshelf IDs and capacities.')
if len(book_names) != len(value_dict) or len(book_names) != len(weight_dict):
    raise ValueError('Mismatch in book names and value/weight data.')
m = gp.Model('Bookstore_MultiKnapsack')
x = m.addVars(bookshelf_ids, book_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in bookshelf_ids for j in book_names)), gp.GRB.MAXIMIZE)
for i in bookshelf_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in book_names)) <= capacity_dict[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Book Allocation per Shelf ---')
    for i in bookshelf_ids:
        shelf_total_value = 0
        shelf_total_weight = 0
        allocation = []
        for j in book_names:
            units = x[i, j].X
            if units > 1e-06:
                allocation.append((j, int(round(units)), value_dict[j], weight_dict[j]))
                shelf_total_value += value_dict[j] * units
                shelf_total_weight += weight_dict[j] * units
        if allocation:
            print(f'\nBookshelf {i} (Capacity: {capacity_dict[i]}):')
            for book, units, val, wt in allocation:
                print(f'  {book}: {units} units (Value/unit: {val}, Weight/unit: {wt})')
            print(f'  >> Shelf total value: {shelf_total_value:.2f}, total weight: {shelf_total_weight:.2f} / {capacity_dict[i]}')
        else:
            print(f'\nBookshelf {i} (Capacity: {capacity_dict[i]}): No books allocated.')
else:
    print(f'No optimal solution found. Status: {m.status}')