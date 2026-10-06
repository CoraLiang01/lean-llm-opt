import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv', sep=',')
bookshelf_ids = capacity_df['BookshelfID'].astype(int).tolist()
product_names = products_df['ProductName'].astype(str).tolist()
if products_df['Value'].isnull().any() or products_df['Weight'].isnull().any():
    raise ValueError('Missing Value or Weight for some products in products.csv')
if capacity_df['Capacity'].isnull().any():
    raise ValueError('Missing Capacity for some bookshelves in capacity.csv')
value = {row['ProductName']: int(row['Value']) for (_, row) in products_df.iterrows()}
weight = {row['ProductName']: int(row['Weight']) for (_, row) in products_df.iterrows()}
capacity = {int(row['BookshelfID']): int(row['Capacity']) for (_, row) in capacity_df.iterrows()}
if set(bookshelf_ids) != set(capacity.keys()):
    raise ValueError('Mismatch between bookshelf IDs in capacity.csv and extracted bookshelf_ids')
if set(product_names) != set(value.keys()) or set(product_names) != set(weight.keys()):
    raise ValueError('Mismatch between product names in products.csv and extracted product_names')
var_keys = [(i, j) for i in bookshelf_ids for j in product_names]
m = gp.Model('Bookstore_MultiKnapsack')
x = m.addVars(var_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in bookshelf_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in bookshelf_ids:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in product_names)) <= capacity[i], name=f'cap_{i}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in bookshelf_ids:
        for j in product_names:
            var = x[i, j]
            print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')