import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'BookshelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise ValueError('capacity.csv must have columns BookshelfID and Capacity')
bookshelves = list(capacity_df['BookshelfID'])
capacity = {}
for idx, row in capacity_df.iterrows():
    try:
        capacity[row['BookshelfID']] = float(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid capacity value for bookshelf {row['BookshelfID']}")
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA18/BooksSalesAndRatings1/products.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError('products.csv must have columns ProductName, Value, Weight')
products = list(products_df['ProductName'])
value = {}
weight = {}
for idx, row in products_df.iterrows():
    try:
        value[row['ProductName']] = float(row['Value'])
        weight[row['ProductName']] = float(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid value or weight for product {row['ProductName']}")
if len(products) == 0 or len(bookshelves) == 0:
    raise ValueError('No products or bookshelves found in the input files.')
for b in bookshelves:
    if b not in capacity:
        raise ValueError(f'Missing capacity for bookshelf {b}')
for p in products:
    if p not in value or p not in weight:
        raise ValueError(f'Missing value or weight for product {p}')
m = gp.Model('Bookshelf_Allocation')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(bookshelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x_vars[b, p] for b in bookshelves for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[p] * x_vars[b, p] for p in products)) <= capacity[b] for b in bookshelves), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')