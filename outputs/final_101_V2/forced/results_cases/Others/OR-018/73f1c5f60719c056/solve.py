import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv'
df = pd.read_csv(csv_path, sep=',')

def is_baby(row):
    return str(row['Product Name']).strip().casefold().startswith('baby')
baby_mask = df.apply(is_baby, axis=1)
baby_df = df[baby_mask].copy()
if baby_df.empty:
    raise ValueError("No 'Baby' products found in the data (no 'Product Name' starts with 'Baby').")
baby_products = list(baby_df['Product Name'])
baby_products = [str(p) for p in baby_products]
revenue = dict(zip(baby_df['Product Name'].astype(str), baby_df['Revenue'].astype(float)))
demand = dict(zip(baby_df['Product Name'].astype(str), baby_df['Demand'].astype(float)))
init_inventory = dict(zip(baby_df['Product Name'].astype(str), baby_df['Initial Inventory'].astype(float)))
for p in baby_products:
    if p not in revenue or p not in demand or p not in init_inventory:
        raise ValueError(f"Missing parameter(s) for product '{p}'.")
m = gp.Model('BabyProductRevenueMax')
x = m.addVars(baby_products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in baby_products)), gp.GRB.MAXIMIZE)
m.addConstrs((x[i] <= demand[i] for i in baby_products), name='')
m.addConstrs((x[i] <= init_inventory[i] for i in baby_products), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print("--- Fulfillment Plan for 'Baby' Products ---")
    for i in baby_products:
        print(f'Product: {i}')
        print(f'  Units fulfilled (x): {x[i].X:.0f}')
        print(f'  Demand: {demand[i]:.0f}')
        print(f'  Initial Inventory: {init_inventory[i]:.0f}')
        print(f'  Revenue per unit: {revenue[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')