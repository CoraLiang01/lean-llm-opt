import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
required_product_cols = ['ProductName', 'Value', 'Weight']
for col in required_product_cols:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
products_df['ProductName'] = products_df['ProductName'].astype(str)
product_ids = products_df['ProductName'].tolist()
try:
    value_dict = dict(zip(products_df['ProductName'], products_df['Value'].astype(int)))
    weight_dict = dict(zip(products_df['ProductName'], products_df['Weight'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting Value or Weight columns to int: {e}')
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row')
try:
    total_capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to int: {e}')
m = gp.Model('CarSales_Inventory_Replenishment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[pid] * x_vars[pid] for pid in product_ids)) <= total_capacity, name='InventoryCapacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Daily Ordering Plan ---')
    for pid in product_ids:
        qty = int(round(x_vars[pid].X))
        print(f'{pid}: {qty} units')
else:
    print(f'No optimal solution found. Status: {m.status}')