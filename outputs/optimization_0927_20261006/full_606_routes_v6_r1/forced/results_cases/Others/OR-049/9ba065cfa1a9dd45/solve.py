import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv'
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
if 'ShelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'ShelfID' and 'Capacity' columns.")
shelf_ids = capacity_df['ShelfID'].apply(lambda x: int(x.strip())).tolist()
shelf_capacities = capacity_df.set_index(capacity_df['ShelfID'].apply(lambda x: int(x.strip())))['Capacity'].apply(lambda x: float(x.strip())).to_dict()
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
product_names = products_df['ProductName'].apply(lambda x: x.strip()).tolist()
product_values = products_df.set_index(products_df['ProductName'].apply(lambda x: x.strip()))['Value'].apply(lambda x: int(x.strip())).to_dict()
product_weights = products_df.set_index(products_df['ProductName'].apply(lambda x: x.strip()))['Weight'].apply(lambda x: float(x.strip())).to_dict()
for sid in shelf_ids:
    if sid not in shelf_capacities:
        raise ValueError(f'ShelfID {sid} missing capacity.')
for pname in product_names:
    if pname not in product_values or pname not in product_weights:
        raise ValueError(f"Product '{pname}' missing value or weight.")
m = gp.Model('RetailShelfAllocation')
x_vars = m.addVars(shelf_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[j] * x_vars[i, j] for i in shelf_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in shelf_ids:
    m.addConstr(gp.quicksum((product_weights[j] * x_vars[i, j] for j in product_names)) <= shelf_capacities[i], name=f'shelf_capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Allocation per shelf ---')
    for i in shelf_ids:
        shelf_total_value = 0
        shelf_total_weight = 0
        print(f'Shelf {i}:')
        for j in product_names:
            qty = x_vars[i, j].X
            if qty > 1e-06:
                value = product_values[j] * qty
                weight = product_weights[j] * qty
                print(f'  {j}: {qty:.0f} units (Value: {value:.2f}, Weight: {weight:.2f})')
                shelf_total_value += value
                shelf_total_weight += weight
        print(f'  >> Total value: {shelf_total_value:.2f}, Total weight: {shelf_total_weight:.2f} / Capacity: {shelf_capacities[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')