import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def is_eles(val):
    return 'ele-s' in val.strip().casefold()
eles_mask = df['Product_Reference'].apply(is_eles)
eles_df = df[eles_mask].copy()
if eles_df.empty:
    raise ValueError("No products found with 'ELE-S' in 'Product_Reference'.")
product_ids = eles_df['Product_Reference'].tolist()
eles_df['Revenue'] = eles_df['Revenue'].astype(float)
eles_df['Demand'] = eles_df['Demand'].astype(float)
eles_df['Initial Inventory'] = eles_df['Initial Inventory'].astype(float)
revenue = dict(zip(eles_df['Product_Reference'], eles_df['Revenue']))
demand = dict(zip(eles_df['Product_Reference'], eles_df['Demand']))
init_inventory = dict(zip(eles_df['Product_Reference'], eles_df['Initial Inventory']))
m = gp.Model('ELE-S_Fulfillment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
for i in product_ids:
    m.addConstr(x_vars[i] <= init_inventory[i], name=f'inv_{i}')
    m.addConstr(x_vars[i] <= demand[i], name=f'dem_{i}')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print("--- Fulfillment Plan for 'ELE-S' Products ---")
    for i in product_ids:
        print(f'Product {i}: Fulfill {int(round(x_vars[i].X))} units (Demand: {int(demand[i])}, Inventory: {int(init_inventory[i])}, Revenue/unit: {revenue[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')