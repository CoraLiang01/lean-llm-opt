import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv'
df = pd.read_csv(csv_path, sep=',')

def is_27in(name):
    return bool(re.search('\\b27in\\b', str(name).replace('\xa0', ' ').strip(), flags=re.IGNORECASE))
df_27in = df[df['Product Name'].apply(is_27in)].copy()
if df_27in.empty:
    raise ValueError("No products with '27in' in the 'Product Name' found in the data.")
products = df_27in['Product Name'].astype(str).tolist()
revenue = df_27in.set_index('Product Name')['Revenue'].astype(float).to_dict()
demand = df_27in.set_index('Product Name')['Demand'].astype(float).to_dict()
inventory = df_27in.set_index('Product Name')['Initial Inventory'].astype(float).to_dict()
for pname in products:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise ValueError(f"Missing data for product '{pname}'.")
m = gp.Model('27in_Product_Revenue_Maximization')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), gp.GRB.MAXIMIZE)
m.addConstrs((x[i] <= demand[i] for i in products), name='')
m.addConstrs((x[i] <= inventory[i] for i in products), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print("--- Fulfillment Plan for '27in' Products ---")
    for i in products:
        print(f'Product: {i}')
        print(f'  Units fulfilled: {x[i].X:.2f}')
        print(f'  Demand: {demand[i]:.0f}')
        print(f'  Initial Inventory: {inventory[i]:.0f}')
        print(f'  Per-unit Revenue: {revenue[i]:.2f}')
        print(f'  Revenue from this product: {revenue[i] * x[i].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')