import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM12/Salesofsummerclothes.csv'
df = pd.read_csv(csv_path, sep=',')
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
df['Product Name'] = df['Product Name'].astype(str)
df['Revenue'] = pd.to_numeric(df['Revenue'], errors='raise')
df['Demand'] = pd.to_numeric(df['Demand'], errors='raise', downcast='integer')
df['Initial Inventory'] = pd.to_numeric(df['Initial Inventory'], errors='raise', downcast='integer')
products = df['Product Name'].tolist()
revenue = dict(zip(df['Product Name'], df['Revenue']))
demand = dict(zip(df['Product Name'], df['Demand']))
inventory = dict(zip(df['Product Name'], df['Initial Inventory']))
for pname in products:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise ValueError(f"Missing parameter for product '{pname}'.")
m = gp.Model('MaximizeRevenue')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for pname in products:
    m.addConstr(x[pname] <= demand[pname], name=f'demand_{pname}')
    m.addConstr(x[pname] <= inventory[pname], name=f'inventory_{pname}')
m.setObjective(gp.quicksum((revenue[pname] * x[pname] for pname in products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Fulfillment Plan ---')
    for pname in products:
        fulfilled = x[pname].X
        print(f'Product: {pname}')
        print(f'  Units fulfilled: {fulfilled:.2f} (Demand: {demand[pname]}, Inventory: {inventory[pname]}, Revenue/unit: {revenue[pname]})')
else:
    print(f'No optimal solution found. Status: {m.status}')