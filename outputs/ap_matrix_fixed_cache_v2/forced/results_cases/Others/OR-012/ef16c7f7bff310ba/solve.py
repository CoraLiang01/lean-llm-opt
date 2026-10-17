import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM3/OnlineSalesDataset.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
df['Product Name'] = df['Product Name'].str.strip()
product_ids = df['Product Name'].tolist()
if len(set(product_ids)) != len(product_ids):
    raise ValueError("Duplicate product names found in 'Product Name' column.")

def safe_int(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to int: {e}")

def safe_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to float: {e}")
revenue_dict = dict(zip(product_ids, safe_float(df['Revenue'], 'Revenue')))
demand_dict = dict(zip(product_ids, safe_int(df['Demand'], 'Demand')))
inventory_dict = dict(zip(product_ids, safe_int(df['Initial Inventory'], 'Initial Inventory')))
for pid in product_ids:
    if not (isinstance(revenue_dict[pid], float) or isinstance(revenue_dict[pid], int)):
        raise ValueError(f"Revenue for product '{pid}' is not numeric.")
    if not (isinstance(demand_dict[pid], int) and demand_dict[pid] >= 0):
        raise ValueError(f"Demand for product '{pid}' is not a non-negative integer.")
    if not (isinstance(inventory_dict[pid], int) and inventory_dict[pid] >= 0):
        raise ValueError(f"Initial Inventory for product '{pid}' is not a non-negative integer.")
m = gp.Model('OnlineRetailerRevenueMaximization')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
demand_constrs = m.addConstrs((x_vars[pid] <= demand_dict[pid] for pid in product_ids), name='')
inventory_constrs = m.addConstrs((x_vars[pid] <= inventory_dict[pid] for pid in product_ids), name='')
m.setObjective(gp.quicksum((revenue_dict[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Fulfillment Plan ---')
    for pid in product_ids:
        fulfilled = int(round(x_vars[pid].X))
        print(f'{pid}: Fulfilled {fulfilled} units (Demand: {demand_dict[pid]}, Inventory: {inventory_dict[pid]}, Revenue/unit: {revenue_dict[pid]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')