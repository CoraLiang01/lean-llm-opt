import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv'
df = pd.read_csv(csv_path, sep=',')
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in the dataset.")
pizza_types = df['Product Name'].astype(str).tolist()
revenue = dict(zip(pizza_types, df['Revenue'].astype(float)))
demand = dict(zip(pizza_types, df['Demand'].astype(int)))
inventory = dict(zip(pizza_types, df['Initial Inventory'].astype(int)))
if not set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()) == set(pizza_types):
    raise ValueError('Mismatch in pizza type keys among parameters.')
m = gp.Model('PizzaFulfillment')
x = m.addVars(pizza_types, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in pizza_types)), gp.GRB.MAXIMIZE)
m.addConstrs((x[i] <= demand[i] for i in pizza_types), name='')
m.addConstrs((x[i] <= inventory[i] for i in pizza_types), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('\n--- Optimal Fulfillment Plan ---')
    for i in pizza_types:
        fulfilled = int(round(x[i].X))
        print(f'Pizza: {i:20s} | Fulfilled: {fulfilled:5d} | Demand: {demand[i]:5d} | Inventory: {inventory[i]:5d} | Revenue/unit: {revenue[i]:6.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')