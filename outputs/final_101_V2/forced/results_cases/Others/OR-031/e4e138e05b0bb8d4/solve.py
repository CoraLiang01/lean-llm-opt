import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv'
df = pd.read_csv(csv_path, sep=',')
required_cols = ['Full_Product_Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
products = df['Full_Product_Name'].astype(str).tolist()
revenue = df.set_index('Full_Product_Name')['Revenue'].astype(float).to_dict()
demand = df.set_index('Full_Product_Name')['Demand'].astype(int).to_dict()
inventory = df.set_index('Full_Product_Name')['Initial Inventory'].astype(int).to_dict()
for p in products:
    if p not in revenue or p not in demand or p not in inventory:
        raise ValueError(f"Missing parameter(s) for product '{p}'.")
m = gp.Model('DairyGoodsOrderFulfillment')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
m.addConstrs((x[p] <= demand[p] for p in products), name='')
m.addConstrs((x[p] <= inventory[p] for p in products), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Fulfillment Plan ---')
    for p in products:
        print(f'{p}: Fulfilled {int(round(x[p].X))} units (Demand: {demand[p]}, Inventory: {inventory[p]}, Revenue/unit: {revenue[p]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')