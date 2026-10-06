import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv', sep=',')
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] != 1:
    raise ValueError("capacity.csv must have exactly one row with a 'Capacity' column.")
capacity = int(capacity_df['Capacity'].iloc[0])
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    missing = required_cols - set(products_df.columns)
    raise ValueError(f'products.csv is missing required columns: {missing}')
products_df['ProductName'] = products_df['ProductName'].astype(str)
product_ids = products_df['ProductName'].tolist()
values = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
weights = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
if any((pd.isnull(values[p]) or pd.isnull(weights[p]) for p in product_ids)):
    raise ValueError('Missing Value or Weight for some products.')
m = gp.Model('CarSalesInventoryReplenishment')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[i] * x[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x[i] for i in product_ids)) <= capacity, name='inventory_capacity')
m.optimize()