import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv'
df = pd.read_csv(csv_path, sep=',')
books_mask = df['Product_Name'].astype(str).str.strip().str.casefold().str.startswith('books_')
books_df = df[books_mask].copy()
if books_df.empty:
    raise ValueError("No products found with Product_Name starting with 'Books_'.")
books_df['Product_Name'] = books_df['Product_Name'].astype(str).str.strip()
product_ids = books_df['Product_Name'].tolist()
revenue = dict(zip(books_df['Product_Name'], books_df['Revenue']))
demand = dict(zip(books_df['Product_Name'], books_df['Demand']))
inventory = dict(zip(books_df['Product_Name'], books_df['Initial Inventory']))
for pid in product_ids:
    if pid not in revenue or pid not in demand or pid not in inventory:
        raise ValueError(f"Missing parameter(s) for product '{pid}'.")
m = gp.Model('Books_Revenue_Maximization')
x = m.addVars(product_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstrs((x[i] <= demand[i] for i in product_ids), name='')
m.addConstrs((x[i] <= inventory[i] for i in product_ids), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print("--- Fulfillment Plan for 'Books' Products ---")
    for i in product_ids:
        print(f'Product: {i}')
        print(f'  Units fulfilled: {x[i].X:.2f}')
        print(f'  Revenue per unit: {revenue[i]:.2f}')
        print(f'  Demand: {demand[i]}')
        print(f'  Initial Inventory: {inventory[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')