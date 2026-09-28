import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv'
df = pd.read_csv(csv_path, sep=',')
if df['Product Name'].isnull().any():
    raise ValueError('Missing product names in the data.')
products = df['Product Name'].astype(str).tolist()
revenue = df.set_index('Product Name')['Revenue'].astype(float).to_dict()
demand = df.set_index('Product Name')['Demand'].astype(float).to_dict()
init_inventory = df.set_index('Product Name')['Initial Inventory'].astype(float).to_dict()
for p in products:
    if p not in revenue or p not in demand or p not in init_inventory:
        raise KeyError(f"Missing data for product '{p}'.")
m = gp.Model('FrenchBakeryRevenueMax')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.addConstrs((x[p] <= demand[p] for p in products), name='')
m.addConstrs((x[p] <= init_inventory[p] for p in products), name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Fulfillment Plan ---')
    for p in products:
        fulfilled = x[p].X
        if fulfilled > 1e-06:
            print(f'{p}: Fulfill {fulfilled:.2f} units (Demand: {demand[p]}, Inventory: {init_inventory[p]}, Revenue/unit: {revenue[p]})')
else:
    print(f'No optimal solution found. Status: {m.status}')