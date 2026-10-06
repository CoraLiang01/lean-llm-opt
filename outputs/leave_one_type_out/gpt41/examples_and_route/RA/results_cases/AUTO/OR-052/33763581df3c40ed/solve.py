import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_products = pd.read_csv(products_path, sep=',')
bookshelf_ids = df_capacity['BookshelfID'].astype(int).tolist()
capacity_dict = dict(zip(df_capacity['BookshelfID'].astype(int), df_capacity['Capacity'].astype(int)))
book_names = df_products['ProductName'].astype(str).tolist()
value_dict = dict(zip(df_products['ProductName'].astype(str), df_products['Value'].astype(int)))
weight_dict = dict(zip(df_products['ProductName'].astype(str), df_products['Weight'].astype(int)))
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
        print(f'Bookshelf {i} (Capacity: {capacity_dict[i]})')
        for j in book_names:
            n = x[i, j].X
            if n > 1e-06:
                v = value_dict[j]
                w = weight_dict[j]
                shelf_total_value += v * n
                shelf_total_weight += w * n
                print(f'  {j}: {int(round(n))} units (Value/unit: {v}, Weight/unit: {w})')
        print(f'  >> Shelf total value: {shelf_total_value:.0f}, total weight: {shelf_total_weight:.0f} / {capacity_dict[i]}')
        print()
else:
    print(f'No optimal solution found. Status: {m.status}')