import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
product_name_col = 'Product Name'
s700_mask = df[product_name_col].str.strip().str.casefold().str.startswith('s700_')
df_s700 = df[s700_mask].copy()
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df_s700.columns:
        raise KeyError(f"Required column '{col}' not found in filtered data.")
df_s700['Product Name'] = df_s700['Product Name'].astype(str)
df_s700 = df_s700.set_index('Product Name', drop=False)
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    try:
        df_s700[col] = pd.to_numeric(df_s700[col], errors='raise')
    except Exception as e:
        raise ValueError(f"Column '{col}' contains non-numeric values: {e}")
product_ids = list(df_s700.index)
revenue = df_s700['Revenue'].to_dict()
demand = df_s700['Demand'].to_dict()
inventory = df_s700['Initial Inventory'].to_dict()
for pid in product_ids:
    if pid not in revenue or pid not in demand or pid not in inventory:
        raise KeyError(f"Missing parameter for product '{pid}'.")
m = gp.Model('S700_Fulfillment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[pid] <= demand[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= inventory[pid] for pid in product_ids), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Fulfillment Plan for S700_ Products ---')
    for pid in product_ids:
        print(f'Product {pid}: Fulfill {int(round(x_vars[pid].X))} units (Demand: {demand[pid]}, Inventory: {inventory[pid]}, Revenue/unit: {revenue[pid]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')