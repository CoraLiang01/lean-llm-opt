import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv'
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
if 'BookshelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'BookshelfID' and 'Capacity' columns.")
capacity_df['BookshelfID'] = capacity_df['BookshelfID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
bookshelf_ids = list(capacity_df['BookshelfID'])
if len(set(bookshelf_ids)) != len(bookshelf_ids):
    raise ValueError('BookshelfID values must be unique.')
capacity_dict = {}
for idx, row in capacity_df.iterrows():
    bid = row['BookshelfID']
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for BookshelfID {bid}: {row['Capacity']}")
    capacity_dict[bid] = cap
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_names = list(products_df['ProductName'])
if len(set(product_names)) != len(product_names):
    raise ValueError('ProductName values must be unique.')
value_dict = {}
weight_dict = {}
for idx, row in products_df.iterrows():
    pname = row['ProductName']
    try:
        val = int(row['Value'])
        wgt = int(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Value or Weight for ProductName {pname}: Value={row['Value']}, Weight={row['Weight']}")
    value_dict[pname] = val
    weight_dict[pname] = wgt
m = gp.Model('Bookstore_MultiKnapsack')
x_vars = m.addVars(bookshelf_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in bookshelf_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in bookshelf_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_names)) <= capacity_dict[i])
m.optimize()