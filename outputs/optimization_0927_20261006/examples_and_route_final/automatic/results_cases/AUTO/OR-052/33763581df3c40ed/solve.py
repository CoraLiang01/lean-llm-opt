import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv', dtype=str, keep_default_na=False)
bookshelf_ids = capacity_df['BookshelfID'].astype(int).tolist()
bookshelf_capacity = dict(zip(capacity_df['BookshelfID'].astype(int), capacity_df['Capacity'].astype(int)))
book_names = products_df['ProductName'].tolist()
book_value = dict(zip(products_df['ProductName'], products_df['Value'].astype(int)))
book_weight = dict(zip(products_df['ProductName'], products_df['Weight'].astype(int)))
if len(bookshelf_ids) != len(bookshelf_capacity):
    raise ValueError('Mismatch in bookshelf IDs and capacities.')
if len(book_names) != len(book_value) or len(book_names) != len(book_weight):
    raise ValueError('Mismatch in book names and value/weight parameters.')
m = gp.Model('Bookstore_MultiKnapsack')
x_vars = m.addVars(bookshelf_ids, book_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((book_value[j] * x_vars[i, j] for i in bookshelf_ids for j in book_names)), gp.GRB.MAXIMIZE)
for i in bookshelf_ids:
    m.addConstr(gp.quicksum((book_weight[j] * x_vars[i, j] for j in book_names)) <= bookshelf_capacity[i], name=f'cap_{i}')
m.optimize()