import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
bookshelf_ids = capacity_df['BookshelfID'].astype(int).tolist()
capacity_dict = {}
for (idx, row) in capacity_df.iterrows():
    bookshelf_id = int(row['BookshelfID'])
    try:
        capacity = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for BookshelfID {bookshelf_id}: {row['Capacity']}")
    if bookshelf_id in capacity_dict:
        raise ValueError(f'Duplicate BookshelfID found: {bookshelf_id}')
    capacity_dict[bookshelf_id] = capacity
product_names = products_df['ProductName'].tolist()
value_dict = {}
weight_dict = {}
for (idx, row) in products_df.iterrows():
    product_name = row['ProductName']
    try:
        value = int(row['Value'])
        weight = int(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Value or Weight for ProductName '{product_name}': Value={row['Value']}, Weight={row['Weight']}")
    if product_name in value_dict or product_name in weight_dict:
        raise ValueError(f"Duplicate ProductName found: '{product_name}'")
    value_dict[product_name] = value
    weight_dict[product_name] = weight
if set(capacity_dict.keys()) != set(bookshelf_ids):
    raise ValueError('Mismatch in BookshelfID keys between capacity_dict and bookshelf_ids')
if set(value_dict.keys()) != set(product_names) or set(weight_dict.keys()) != set(product_names):
    raise ValueError('Mismatch in ProductName keys between value_dict/weight_dict and product_names')
decision_keys = [(i, j) for i in bookshelf_ids for j in product_names]
m = gp.Model('BookstoreMultiKnapsack')
x_vars = m.addVars(decision_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in bookshelf_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in bookshelf_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_names)) <= capacity_dict[i], name=f'cap_{i}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (i, j) in decision_keys:
        var = x_vars[i, j]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')