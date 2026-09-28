import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM4/OnlineSalesinUSA.csv'
df = pd.read_csv(csv_path, sep=',')
mask_4u = df['Product Name'].astype(str).str.contains('4U', case=False, na=False)
df_4u = df[mask_4u].copy()
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df_4u.columns:
        raise KeyError(f"Required column '{col}' not found in the data.")
    if df_4u[col].isnull().any():
        raise ValueError(f"Missing values found in column '{col}' for selected 4U products.")
products = df_4u['Product Name'].astype(str).tolist()
revenue = dict(zip(products, df_4u['Revenue'].astype(float)))
demand = dict(zip(products, df_4u['Demand'].astype(int)))
inventory = dict(zip(products, df_4u['Initial Inventory'].astype(int)))
for p in products:
    if p not in revenue or p not in demand or p not in inventory:
        raise ValueError(f"Missing parameter for product '{p}'.")
m = gp.Model('4U_Product_Fulfillment')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
m.addConstrs((x[p] <= demand[p] for p in products), name='')
m.addConstrs((x[p] <= inventory[p] for p in products), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_revenue = m.objVal
    print(f'Optimal total revenue: {total_revenue:.2f}')
    print('--- Fulfillment Plan for 4U Products ---')
    for p in products:
        fulfilled = int(round(x[p].X))
        print(f'Product: {p} | Fulfilled: {fulfilled} | Demand: {demand[p]} | Inventory: {inventory[p]} | Revenue/unit: {revenue[p]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')