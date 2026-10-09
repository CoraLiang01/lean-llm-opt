import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv', dtype=str, keep_default_na=False)
if 'BookshelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError('Missing required columns in capacity.csv')
capacity_df['BookshelfID'] = capacity_df['BookshelfID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
if capacity_df['BookshelfID'].duplicated().any():
    raise ValueError('Duplicate BookshelfID found in capacity.csv')
if (capacity_df['BookshelfID'] == '').any():
    raise ValueError('Blank BookshelfID found in capacity.csv')
bookshelf_ids = list(capacity_df['BookshelfID'])
capacity_dict = {}
for (idx, row) in capacity_df.iterrows():
    bid = row['BookshelfID']
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f'Invalid Capacity value for BookshelfID {bid}')
    capacity_dict[bid] = cap
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError('Missing required columns in products.csv')
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
if products_df['ProductName'].duplicated().any():
    raise ValueError('Duplicate ProductName found in products.csv')
if (products_df['ProductName'] == '').any():
    raise ValueError('Blank ProductName found in products.csv')
product_names = list(products_df['ProductName'])
value_dict = {}
weight_dict = {}
for (idx, row) in products_df.iterrows():
    pname = row['ProductName']
    try:
        val = int(row['Value'])
        wgt = int(row['Weight'])
    except Exception:
        raise ValueError(f'Invalid Value or Weight for ProductName {pname}')
    value_dict[pname] = val
    weight_dict[pname] = wgt
m = gp.Model('Bookstore_MultiKnapsack')
x_vars = m.addVars(bookshelf_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in bookshelf_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in bookshelf_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_names)) <= capacity_dict[i])
m.optimize()