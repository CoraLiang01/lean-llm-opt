import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['ProductName', 'Value', 'Weight']
for col in required_columns:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
products_df['ProductName'] = products_df['ProductName'].astype(str)
products_df = products_df.set_index('ProductName', drop=False)
try:
    products_df['Value'] = products_df['Value'].astype(int)
    products_df['Weight'] = products_df['Weight'].astype(int)
except Exception as e:
    raise ValueError(f'Error converting Value or Weight to int in products.csv: {e}')
product_ids = list(products_df.index)
value_dict = products_df['Value'].to_dict()
weight_dict = products_df['Weight'].to_dict()
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row for total capacity.')
try:
    total_capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to int in capacity.csv: {e}')
m = gp.Model('BakeryBreadOrder')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in product_ids)) <= total_capacity, name='storage_capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Bread Order Plan ---')
    for i in product_ids:
        qty = x_vars[i].X
        if qty > 0.5:
            print(f'{i}: {int(round(qty))} units (Profit/unit: {value_dict[i]}, Weight/unit: {weight_dict[i]})')
    total_weight = sum((weight_dict[i] * x_vars[i].X for i in product_ids))
    print(f'Total storage used: {int(round(total_weight))} / {total_capacity}')
else:
    print(f'No optimal solution found. Status: {m.status}')