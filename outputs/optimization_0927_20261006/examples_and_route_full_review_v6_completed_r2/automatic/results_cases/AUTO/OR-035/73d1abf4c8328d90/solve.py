import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
required_product_cols = {'ProductName', 'Value', 'Weight'}
if not required_product_cols.issubset(products_df.columns):
    missing = required_product_cols - set(products_df.columns)
    raise KeyError(f'Missing columns in products.csv: {missing}')
products_df['ProductName'] = products_df['ProductName'].astype(str)
product_ids = products_df['ProductName'].tolist()
try:
    value_dict = dict(zip(products_df['ProductName'], products_df['Value'].astype(int)))
    weight_dict = dict(zip(products_df['ProductName'], products_df['Weight'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting Value or Weight to int in products.csv: {e}')
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Missing 'Capacity' column in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row for total capacity')
try:
    total_capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to int in capacity.csv: {e}')
m = gp.Model('BakeryOrderKnapsack')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in product_ids)) <= total_capacity, name='storage_capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Bread Order ---')
    for i in product_ids:
        qty = x_vars[i].X
        if qty > 1e-06:
            print(f'{i}: {qty:.0f} units (Profit/unit: {value_dict[i]}, Weight/unit: {weight_dict[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')