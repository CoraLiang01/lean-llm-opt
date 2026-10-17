import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
required_product_cols = ['ProductName', 'Weight', 'Value']
for col in required_product_cols:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
products_df['ProductName'] = products_df['ProductName'].astype(str)
products_df['Weight'] = products_df['Weight'].apply(lambda x: int(x.strip()))
products_df['Value'] = products_df['Value'].apply(lambda x: int(x.strip()))
product_ids = list(products_df['ProductName'])
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row')
try:
    capacity = int(capacity_df.iloc[0]['Capacity'].strip())
except Exception as e:
    raise ValueError(f'Could not parse Capacity value: {e}')
m = gp.Model('SupermarketStockReplenishment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[pid] * x_vars[pid] for pid in product_ids)) <= capacity, name='TotalStockCapacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Daily Order Quantities ---')
    for pid in product_ids:
        qty = x_vars[pid].X
        if qty > 1e-06:
            print(f'{pid}: {qty:.0f} units (Weight per unit: {weight_dict[pid]}, Value per unit: {value_dict[pid]})')
        else:
            print(f'{pid}: 0 units')
else:
    print(f'No optimal solution found. Status: {m.status}')