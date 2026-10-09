import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if not {'ShelfID', 'Capacity'}.issubset(capacity_df.columns):
    raise KeyError('capacity.csv must contain columns: ShelfID, Capacity')
capacity_df['ShelfID'] = capacity_df['ShelfID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
shelf_ids = capacity_df['ShelfID'].astype(int).tolist()
shelf_capacities = dict(zip(capacity_df['ShelfID'].astype(int), capacity_df['Capacity'].astype(int)))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv', sep=',', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError('products.csv must contain columns: ProductName, Value, Weight')
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_ids = products_df['ProductName'].astype(int).tolist()
product_values = dict(zip(products_df['ProductName'].astype(int), products_df['Value'].astype(int)))
product_weights = dict(zip(products_df['ProductName'].astype(int), products_df['Weight'].astype(int)))
if len(shelf_ids) != len(shelf_capacities):
    raise ValueError('Mismatch in shelf IDs and capacities.')
if len(product_ids) != len(product_values) or len(product_ids) != len(product_weights):
    raise ValueError('Mismatch in product IDs, values, or weights.')
m = gp.Model('BigMart_MultiShelf_Knapsack')
x_vars = m.addVars(shelf_ids, product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[j] * x_vars[i, j] for i in shelf_ids for j in product_ids)), gp.GRB.MAXIMIZE)
for i in shelf_ids:
    m.addConstr(gp.quicksum((product_weights[j] * x_vars[i, j] for j in product_ids)) <= shelf_capacities[i], name=f'shelf_capacity_{i}')
m.optimize()