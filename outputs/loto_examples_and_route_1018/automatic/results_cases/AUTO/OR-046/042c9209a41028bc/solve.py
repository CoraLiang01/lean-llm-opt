import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
for col in ['ProductName', 'Weight', 'Value']:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
products_df['ProductName'] = products_df['ProductName'].astype(str)
product_ids = products_df['ProductName'].tolist()

def safe_int(x, col, pid):
    try:
        return int(x)
    except Exception:
        raise ValueError(f"Invalid integer in column '{col}' for product '{pid}': {x}")
weight_dict = {}
value_dict = {}
for idx, row in products_df.iterrows():
    pid = row['ProductName']
    weight_dict[pid] = safe_int(row['Weight'], 'Weight', pid)
    value_dict[pid] = safe_int(row['Value'], 'Value', pid)
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row')
try:
    capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception:
    raise ValueError(f"Invalid integer in 'Capacity': {capacity_df.iloc[0]['Capacity']}")
m = gp.Model('SupermarketStockReplenishment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[pid] * x_vars[pid] for pid in product_ids)) <= capacity, name='StockCapacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Order Quantities ---')
    for pid in product_ids:
        qty = x_vars[pid].X
        print(f'{pid}: {qty:.0f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')