import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
required_product_cols = ['ProductName', 'Weight', 'Value']
for col in required_product_cols:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
products_df['ProductName'] = products_df['ProductName'].astype(str)
product_ids = products_df['ProductName'].tolist()
try:
    products_df['Weight'] = products_df['Weight'].astype(int)
    products_df['Value'] = products_df['Value'].astype(int)
except Exception as e:
    raise ValueError(f'Failed to convert Weight or Value to int: {e}')
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
value = dict(zip(products_df['ProductName'], products_df['Value']))
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row')
try:
    capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Failed to convert Capacity to int: {e}')
if set(weight.keys()) != set(product_ids) or set(value.keys()) != set(product_ids):
    raise ValueError('Mismatch in product identifiers between columns')
m = gp.Model('SupermarketRestock')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x_vars[i] for i in product_ids)) <= capacity, name='TotalWeight')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Daily Order Quantities ---')
    for i in product_ids:
        print(f'{i}: {int(round(x_vars[i].X))} units')
else:
    print(f'No optimal solution found. Status: {m.status}')