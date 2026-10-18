import gurobipy as gp
import pandas as pd
import numpy as np
import re

def find_col(df, pattern):
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'ProductName' not in products_df.columns:
    raise KeyError("products.csv missing required column 'ProductName'")
product_ids = products_df['ProductName'].tolist()

def norm_str(x):
    return x.strip()
if 'Value' not in products_df.columns:
    raise KeyError("products.csv missing required column 'Value'")
profit_dict = {}
for idx, row in products_df.iterrows():
    pid = norm_str(row['ProductName'])
    try:
        profit = int(row['Value'])
    except Exception:
        raise ValueError(f"Invalid Value for product '{pid}': {row['Value']}")
    profit_dict[pid] = profit
if 'Weight' not in products_df.columns:
    raise KeyError("products.csv missing required column 'Weight'")
weight_dict = {}
for idx, row in products_df.iterrows():
    pid = norm_str(row['ProductName'])
    try:
        weight = int(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Weight for product '{pid}': {row['Weight']}")
    weight_dict[pid] = weight
if 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv missing required column 'Capacity'")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row')
try:
    capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception:
    raise ValueError(f"Invalid Capacity value: {capacity_df.iloc[0]['Capacity']}")
m = gp.Model('CarSalesInventoryReplenishment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((profit_dict[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[pid] * x_vars[pid] for pid in product_ids)) <= capacity, name='TotalCapacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Daily Ordering Plan ---')
    for pid in product_ids:
        qty = x_vars[pid].X
        if qty > 1e-06:
            print(f'{pid}: {qty:.0f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')