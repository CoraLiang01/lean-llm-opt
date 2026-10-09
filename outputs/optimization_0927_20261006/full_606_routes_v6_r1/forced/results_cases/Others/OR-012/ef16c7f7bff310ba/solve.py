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
df['Product Name'] = df['Product Name'].astype(str).str.strip()
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    if col == 'Revenue':
        df[col] = df[col].astype(float)
    else:
        df[col] = df[col].astype(int)
product_ids = df['Product Name'].tolist()
revenue = dict(zip(product_ids, df['Revenue']))
demand = dict(zip(product_ids, df['Demand']))
initial_inventory = dict(zip(product_ids, df['Initial Inventory']))
if not set(revenue.keys()) == set(demand.keys()) == set(initial_inventory.keys()) == set(product_ids):
    raise ValueError('Mismatch in product keys among parameters.')
m = gp.Model('OnlineRetailerRevenueMaximization')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
demand_constrs = m.addConstrs((x_vars[i] <= demand[i] for i in product_ids), name='')
inventory_constrs = m.addConstrs((x_vars[i] <= initial_inventory[i] for i in product_ids), name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Fulfillment Plan ---')
    for i in product_ids:
        print(f'{i}: Fulfilled {int(round(x_vars[i].X))} units (Demand: {demand[i]}, Inventory: {initial_inventory[i]}, Revenue/unit: {revenue[i]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')