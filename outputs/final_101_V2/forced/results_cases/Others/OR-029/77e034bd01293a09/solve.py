import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv'
df = pd.read_csv(csv_path, sep=',')
faux_mask = df['Product Name'].astype(str).str.casefold().str.contains('faux')
df_faux = df[faux_mask].copy()
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df_faux.columns:
        raise KeyError(f"Required column '{col}' not found in filtered DataFrame.")
    if df_faux[col].isnull().any():
        raise ValueError(f"Missing values found in column '{col}' for FAUX products.")
product_names = df_faux['Product Name'].astype(str).tolist()
revenue = dict(zip(product_names, df_faux['Revenue'].astype(float)))
demand = dict(zip(product_names, df_faux['Demand'].astype(float)))
inventory = dict(zip(product_names, df_faux['Initial Inventory'].astype(float)))
m = gp.Model('FauxRevenueMaximization')
x = m.addVars(product_names, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.addConstrs((x[i] <= demand[i] for i in product_names), name='')
m.addConstrs((x[i] <= inventory[i] for i in product_names), name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in product_names)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Fulfillment Plan for FAUX Products ---')
    for i in product_names:
        fulfilled = x[i].X
        print(f'Product: {i}')
        print(f'  Units fulfilled: {fulfilled:.2f} (Demand: {demand[i]:.0f}, Inventory: {inventory[i]:.0f}, Revenue/unit: {revenue[i]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')