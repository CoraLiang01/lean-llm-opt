import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
required_cols = ['ProductName', 'Value', 'Weight']
for col in required_cols:
    if col not in products_df.columns:
        raise KeyError(f"Missing required column '{col}' in products.csv")
product_ids = products_df['ProductName'].tolist()
try:
    value_dict = dict(zip(products_df['ProductName'], products_df['Value'].astype(np.int64)))
    weight_dict = dict(zip(products_df['ProductName'], products_df['Weight'].astype(np.int64)))
except Exception as e:
    raise ValueError(f'Error converting Value or Weight columns to integer: {e}')
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Missing required column 'Capacity' in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row')
try:
    capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to integer: {e}')
for pid in product_ids:
    if pid not in value_dict or pid not in weight_dict:
        raise KeyError(f"Missing Value or Weight for product '{pid}'")
m = gp.Model('CarSalesInventoryReplenishment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[pid] * x_vars[pid] for pid in product_ids)) <= capacity, name='InventoryCapacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Daily Ordering Plan ---')
    for pid in product_ids:
        qty = x_vars[pid].X
        if qty > 1e-06:
            print(f'{pid}: {int(round(qty))} units')
else:
    print(f'No optimal solution found. Status: {m.status}')