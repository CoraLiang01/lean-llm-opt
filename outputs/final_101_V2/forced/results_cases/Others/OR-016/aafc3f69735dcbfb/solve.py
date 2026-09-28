import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv'
df = pd.read_csv(csv_path, sep=',')
if df['Product Name'].isnull().any():
    raise ValueError('Missing Product Name(s) in the data.')
if df['Revenue'].isnull().any():
    raise ValueError('Missing Revenue value(s) in the data.')
if df['Demand'].isnull().any():
    raise ValueError('Missing Demand value(s) in the data.')
if df['Initial Inventory'].isnull().any():
    raise ValueError('Missing Initial Inventory value(s) in the data.')
products = df['Product Name'].astype(str).tolist()
revenue = dict(zip(df['Product Name'].astype(str), df['Revenue'].astype(float)))
demand = dict(zip(df['Product Name'].astype(str), df['Demand'].astype(float)))
inventory = dict(zip(df['Product Name'].astype(str), df['Initial Inventory'].astype(float)))
for p in products:
    if p not in revenue or p not in demand or p not in inventory:
        raise ValueError(f"Missing parameter(s) for product '{p}'.")
m = gp.Model('RetailMerchandiseAllocation')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
m.addConstrs((x[p] <= demand[p] for p in products), name='')
m.addConstrs((x[p] <= inventory[p] for p in products), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Optimal Fulfillment Plan ---')
    for p in products:
        print(f'{p}: Fulfill {x[p].X:.2f} units (Demand: {demand[p]}, Inventory: {inventory[p]}, Revenue/unit: {revenue[p]})')
else:
    print(f'No optimal solution found. Status: {m.status}')