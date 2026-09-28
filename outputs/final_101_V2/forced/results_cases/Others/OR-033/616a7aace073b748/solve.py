import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv'
df = pd.read_csv(csv_path, sep=',')
baby_mask = df['Product Name'].astype(str).str.casefold().str.contains('baby')
baby_df = df[baby_mask].copy()
if baby_df.empty:
    raise ValueError("No products with 'Baby' in the 'Product Name' column were found.")
products = list(baby_df['Product Name'])
revenue = {}
demand = {}
inventory = {}
for idx, row in baby_df.iterrows():
    pname = str(row['Product Name'])
    try:
        revenue[pname] = float(row['Revenue'])
        demand[pname] = int(row['Demand'])
        inventory[pname] = int(row['Initial Inventory'])
    except Exception as e:
        raise ValueError(f"Invalid data for product '{pname}': {e}")
m = gp.Model('MaximizeBabyRevenue')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.addConstrs((x[i] <= demand[i] for i in products), name='')
m.addConstrs((x[i] <= inventory[i] for i in products), name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print("--- Fulfillment Plan for 'Baby' Products ---")
    for i in products:
        print(f'Product: {i}')
        print(f'  Units fulfilled: {x[i].X:.2f}')
        print(f'  Demand: {demand[i]}')
        print(f'  Initial Inventory: {inventory[i]}')
        print(f'  Revenue per unit: {revenue[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')