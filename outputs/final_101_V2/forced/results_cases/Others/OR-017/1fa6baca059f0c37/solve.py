import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv'
df = pd.read_csv(csv_path, sep=',')
zz_mask = df['SKU'].astype(str).str.contains('ZZ', case=True, na=False)
df_zz = df.loc[zz_mask].copy()
sku_list = df_zz['SKU'].astype(str).tolist()
required_cols = ['Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if df_zz[col].isnull().any():
        raise ValueError(f"Missing values detected in column '{col}' for selected 'ZZ' SKUs.")
revenue = df_zz.set_index('SKU')['Revenue'].astype(float).to_dict()
demand = df_zz.set_index('SKU')['Demand'].astype(float).to_dict()
init_inventory = df_zz.set_index('SKU')['Initial Inventory'].astype(float).to_dict()
m = gp.Model('Maximize_ZZ_Revenue')
x = m.addVars(sku_list, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.addConstrs((x[i] <= demand[i] for i in sku_list), name='')
m.addConstrs((x[i] <= init_inventory[i] for i in sku_list), name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in sku_list)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print("--- Fulfillment Plan for 'ZZ' SKUs ---")
    for i in sku_list:
        fulfilled = x[i].X
        print(f'SKU: {i} | Fulfilled: {fulfilled:.2f} | Demand: {demand[i]:.2f} | Inventory: {init_inventory[i]:.2f} | Revenue/unit: {revenue[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')