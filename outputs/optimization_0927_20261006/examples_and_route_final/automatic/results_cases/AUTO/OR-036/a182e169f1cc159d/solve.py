import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv', sep=',', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
product_ids = products_df['ProductName'].tolist()
try:
    value_param = {}
    weight_param = {}
    for (idx, row) in products_df.iterrows():
        pid = str(row['ProductName'])
        try:
            value_param[pid] = int(row['Value'])
        except Exception:
            raise ValueError(f"Invalid Value for product '{pid}': {row['Value']}")
        try:
            weight_param[pid] = int(row['Weight'])
        except Exception:
            raise ValueError(f"Invalid Weight for product '{pid}': {row['Weight']}")
    if set(value_param.keys()) != set(product_ids) or set(weight_param.keys()) != set(product_ids):
        raise ValueError('Mismatch in product identifiers between Value and Weight columns.')
except KeyError as e:
    raise KeyError(f'Missing required column in products.csv: {e}')
try:
    if 'Capacity' not in capacity_df.columns:
        raise KeyError("Missing 'Capacity' column in capacity.csv")
    if len(capacity_df) != 1:
        raise ValueError('capacity.csv must contain exactly one row.')
    capacity_val = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error reading capacity.csv: {e}')
m = gp.Model('CarSalesInventoryReplenishment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_param[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[pid] * x_vars[pid] for pid in product_ids)) <= capacity_val, name='inventory_capacity')
m.optimize()