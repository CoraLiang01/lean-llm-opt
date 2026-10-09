import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math

def find_col(df, pattern):
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if 'ShelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain columns 'ShelfID' and 'Capacity'")
shelf_ids = capacity_df['ShelfID'].astype(int).tolist()
shelf_capacity = {}
for (idx, row) in capacity_df.iterrows():
    sid = int(row['ShelfID'])
    try:
        cap = float(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for ShelfID {sid}: {row['Capacity']}")
    shelf_capacity[sid] = cap
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain columns 'ProductName', 'Value', and 'Weight'")
product_names = products_df['ProductName'].tolist()
product_value = {}
product_weight = {}
for (idx, row) in products_df.iterrows():
    pname = str(row['ProductName'])
    try:
        val = int(row['Value'])
    except Exception:
        raise ValueError(f"Invalid Value for Product '{pname}': {row['Value']}")
    try:
        wgt = float(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Weight for Product '{pname}': {row['Weight']}")
    product_value[pname] = val
    product_weight[pname] = wgt
if set(shelf_capacity.keys()) != set(shelf_ids):
    raise ValueError('Mismatch in shelf IDs and capacity mapping.')
if set(product_value.keys()) != set(product_names) or set(product_weight.keys()) != set(product_names):
    raise ValueError('Mismatch in product names and value/weight mapping.')
m = gp.Model('RetailShelfAllocation')
x_vars = m.addVars(shelf_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_value[j] * x_vars[i, j] for i in shelf_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in shelf_ids:
    m.addConstr(gp.quicksum((product_weight[j] * x_vars[i, j] for j in product_names)) <= shelf_capacity[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Shelf Allocation ---')
    for i in shelf_ids:
        print(f'Shelf {i} (Capacity {shelf_capacity[i]}):')
        for j in product_names:
            qty = x_vars[i, j].X
            if qty > 1e-06:
                print(f'  {j}: {qty:.0f} units (Value per unit: {product_value[j]}, Weight per unit: {product_weight[j]})')
        total_weight = sum((product_weight[j] * x_vars[i, j].X for j in product_names))
        print(f'  >> Total weight used: {total_weight:.2f} / {shelf_capacity[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')