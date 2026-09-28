import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM3/OnlineSalesDataset.csv'
df = pd.read_csv(csv_path, sep=',')
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
products = df['Product Name'].astype(str).tolist()
revenue = df.set_index('Product Name')['Revenue'].astype(float).to_dict()
demand = df.set_index('Product Name')['Demand'].astype(int).to_dict()
init_inventory = df.set_index('Product Name')['Initial Inventory'].astype(int).to_dict()
for pname in products:
    if pname not in revenue or pname not in demand or pname not in init_inventory:
        raise ValueError(f"Missing parameter for product '{pname}'.")
m = gp.Model('OnlineRetailerRevenueMaximization')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
m.addConstrs((x[p] <= demand[p] for p in products), name='')
m.addConstrs((x[p] <= init_inventory[p] for p in products), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Fulfillment Plan ---')
    for p in products:
        fulfilled = int(round(x[p].X))
        print(f'{p}: {fulfilled} units fulfilled (Demand: {demand[p]}, Inventory: {init_inventory[p]}, Revenue/unit: {revenue[p]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')