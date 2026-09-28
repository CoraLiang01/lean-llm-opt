import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv'
df = pd.read_csv(csv_path, sep=',')
if df['Product Name'].isnull().any():
    raise ValueError('Missing Product Name(s) in input data.')
if df['Revenue'].isnull().any():
    raise ValueError('Missing Revenue value(s) in input data.')
if df['Demand'].isnull().any():
    raise ValueError('Missing Demand value(s) in input data.')
if df['Initial Inventory'].isnull().any():
    raise ValueError('Missing Initial Inventory value(s) in input data.')
products = df['Product Name'].astype(str).tolist()
revenue = df.set_index('Product Name')['Revenue'].astype(float).to_dict()
demand = df.set_index('Product Name')['Demand'].astype(int).to_dict()
inventory = df.set_index('Product Name')['Initial Inventory'].astype(int).to_dict()
if not set(products) == set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()):
    raise ValueError('Mismatch in product identifiers across columns.')
m = gp.Model('MobileDeviceOrderFulfillment')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), gp.GRB.MAXIMIZE)
m.addConstrs((x[i] <= demand[i] for i in products), name='')
m.addConstrs((x[i] <= inventory[i] for i in products), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Fulfillment Plan ---')
    for i in products:
        fulfilled = int(round(x[i].X))
        print(f'Product: {i} | Fulfilled: {fulfilled} | Demand: {demand[i]} | Inventory: {inventory[i]} | Revenue/unit: {revenue[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')