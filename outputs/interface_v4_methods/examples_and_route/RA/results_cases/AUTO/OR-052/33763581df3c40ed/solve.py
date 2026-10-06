import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv', sep=',')
capacity_df['BookshelfID'] = capacity_df['BookshelfID'].astype(int)
bookshelf_ids = list(capacity_df['BookshelfID'])
capacity_dict = dict(zip(capacity_df['BookshelfID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv', sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str)
product_names = list(products_df['ProductName'])
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(bookshelf_ids) == 0:
    raise ValueError('No bookshelf IDs found in capacity.csv')
if len(product_names) == 0:
    raise ValueError('No product names found in products.csv')
if set(capacity_dict.keys()) != set(bookshelf_ids):
    raise ValueError('Mismatch in bookshelf IDs between index and capacity_dict')
if set(value_dict.keys()) != set(product_names) or set(weight_dict.keys()) != set(product_names):
    raise ValueError('Mismatch in product names between index and value/weight dicts')
m = gp.Model('Bookstore_MultiKnapsack')
x = m.addVars(bookshelf_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in bookshelf_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in bookshelf_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in product_names)) <= capacity_dict[i], name='')
m.optimize()