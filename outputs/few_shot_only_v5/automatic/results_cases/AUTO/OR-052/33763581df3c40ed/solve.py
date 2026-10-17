import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv', dtype=str, keep_default_na=False)
capacity_df['BookshelfID'] = capacity_df['BookshelfID'].astype(str).str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
bookshelves = list(capacity_df['BookshelfID'])
capacity = dict(zip(capacity_df['BookshelfID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv', dtype=str, keep_default_na=False)
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
products = list(products_df['ProductName'])
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(bookshelves) == 0 or len(products) == 0:
    raise ValueError('No bookshelf or product data found.')
for b in bookshelves:
    if b not in capacity:
        raise ValueError(f'Missing capacity for bookshelf {b}')
for p in products:
    if p not in value or p not in weight:
        raise ValueError(f'Missing value or weight for product {p}')
m = gp.Model('Bookshelf_Allocation')
x_vars = m.addVars(bookshelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x_vars[b, p] for b in bookshelves for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[p] * x_vars[b, p] for p in products)) <= capacity[b] for b in bookshelves), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')