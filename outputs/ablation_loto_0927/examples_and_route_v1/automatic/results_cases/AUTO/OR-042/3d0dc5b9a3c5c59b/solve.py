import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv', sep=',')
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain columns: 'ProductName', 'Value', 'Weight'.")
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
product_ids = list(products_df['ProductName'])
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(product_ids) != set(value_dict.keys()) or set(product_ids) != set(weight_dict.keys()):
    raise ValueError('Mismatch in product identifiers between Value and Weight columns.')
if 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain column: 'Capacity'.")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row.')
capacity = int(capacity_df['Capacity'].iloc[0])
m = gp.Model('PharmacyRestock')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in product_ids)) <= capacity, name='capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f}')
    print('--- Order Plan ---')
    for i in product_ids:
        xi = int(round(x[i].X))
        if xi > 0:
            print(f'{i}: {xi} units (Value/unit: {value_dict[i]}, Weight/unit: {weight_dict[i]})')
    total_weight = sum((weight_dict[i] * int(round(x[i].X)) for i in product_ids))
    print(f'Total weight used: {total_weight} / {capacity}')
else:
    print(f'No optimal solution found. Status: {m.status}')