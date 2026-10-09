import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
required_product_cols = ['ProductName', 'Weight', 'Value']
for col in required_product_cols:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
product_ids = products_df['ProductName'].tolist()
try:
    weight_dict = dict(zip(products_df['ProductName'], products_df['Weight'].apply(lambda x: int(x.strip()))))
    value_dict = dict(zip(products_df['ProductName'], products_df['Value'].apply(lambda x: int(x.strip()))))
except Exception as e:
    raise ValueError(f'Error converting Weight/Value columns to int: {e}')
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row')
try:
    capacity = int(capacity_df.iloc[0]['Capacity'].strip())
except Exception as e:
    raise ValueError(f'Error converting Capacity to int: {e}')
for pid in product_ids:
    if pid not in weight_dict or pid not in value_dict:
        raise KeyError(f"Missing weight or value for product '{pid}'")
m = gp.Model('SupermarketStockReplenishment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[pid] * x_vars[pid] for pid in product_ids)) <= capacity, name='StockCapacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Daily Order Quantities ---')
    for pid in product_ids:
        qty = x_vars[pid].X
        print(f'{pid}: {int(round(qty))} units')
else:
    print(f'No optimal solution found. Status: {m.status}')