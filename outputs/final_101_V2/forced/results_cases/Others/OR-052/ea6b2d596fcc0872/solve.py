import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
bookshelf_ids = capacity_df['BookshelfID'].astype(int).tolist()
capacity = dict(zip(capacity_df['BookshelfID'].astype(int), capacity_df['Capacity'].astype(int)))
book_names = products_df['ProductName'].astype(str).tolist()
value = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(int)))
weight = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(int)))
if len(bookshelf_ids) != len(capacity):
    raise ValueError('Mismatch in bookshelf IDs and capacity mapping.')
if len(book_names) != len(value) or len(book_names) != len(weight):
    raise ValueError('Mismatch in book names and value/weight mapping.')
m = gp.Model('BookstoreMultiKnapsack')
x = m.addVars(bookshelf_ids, book_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in bookshelf_ids for j in book_names)), gp.GRB.MAXIMIZE)
for i in bookshelf_ids:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in book_names)) <= capacity[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Allocation (bookshelf, book, units) ---')
    for i in bookshelf_ids:
        shelf_total_value = 0
        shelf_total_weight = 0
        for j in book_names:
            units = x[i, j].X
            if units > 1e-06:
                print(f"Shelf {i} | Book: '{j}' | Units: {int(round(units))} | Value: {value[j]} | Weight: {weight[j]}")
                shelf_total_value += value[j] * units
                shelf_total_weight += weight[j] * units
        print(f'  >> Shelf {i} total value: {shelf_total_value:.2f}, total weight: {shelf_total_weight:.2f} / capacity {capacity[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')