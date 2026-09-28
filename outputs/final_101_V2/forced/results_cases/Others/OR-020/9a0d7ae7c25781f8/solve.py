import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv'
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
revenue = df.set_index('Product Name')['Revenue'].to_dict()
demand = df.set_index('Product Name')['Demand'].astype(int).to_dict()
init_inventory = df.set_index('Product Name')['Initial Inventory'].astype(int).to_dict()
for p in products:
    if p not in revenue or p not in demand or p not in init_inventory:
        raise ValueError(f'Missing parameter(s) for product {p}')
m = gp.Model('SupermarketRevenueMaximization')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.addConstrs((x[p] <= demand[p] for p in products), name='')
m.addConstrs((x[p] <= init_inventory[p] for p in products), name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Fulfillment Plan ---')
    for p in products:
        print(f'{p}: Fulfill {int(round(x[p].X))} units (Demand: {demand[p]}, Inventory: {init_inventory[p]}, Revenue/unit: {revenue[p]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')