import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv'
df = pd.read_csv(csv_path, sep=',')
mask_s700 = df['Product Name'].astype(str).str.startswith('S700_')
df_s700 = df[mask_s700].copy()
if df_s700.empty:
    raise ValueError("No products found with 'Product Name' starting with 'S700_'.")
products = df_s700['Product Name'].astype(str).tolist()
revenue = df_s700.set_index('Product Name')['Revenue'].to_dict()
demand = df_s700.set_index('Product Name')['Demand'].to_dict()
inventory = df_s700.set_index('Product Name')['Initial Inventory'].to_dict()
for i in products:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing data for product {i}.')
m = gp.Model('S700_Fulfillment')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.addConstrs((x[i] <= float(demand[i]) for i in products), name='')
m.addConstrs((x[i] <= float(inventory[i]) for i in products), name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Fulfillment Plan for S700_ Products ---')
    for i in products:
        print(f'Product {i}: Fulfill {x[i].X:.2f} units (Demand: {demand[i]}, Inventory: {inventory[i]}, Revenue/unit: {revenue[i]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')