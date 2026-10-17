import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM19/WomenClothingEcommerceSalesData.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
df['Product Name'] = df['Product Name'].str.strip()
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    try:
        df[col] = df[col].astype(int)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to integer: {e}")
products = df['Product Name'].tolist()
revenue = dict(zip(df['Product Name'], df['Revenue']))
demand = dict(zip(df['Product Name'], df['Demand']))
init_inventory = dict(zip(df['Product Name'], df['Initial Inventory']))
for p in products:
    if p not in revenue or p not in demand or p not in init_inventory:
        raise KeyError(f"Missing parameter for product '{p}'.")
m = gp.Model('DemandFulfillment')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
demand_constrs = m.addConstrs((x_vars[p] <= demand[p] for p in products), name='')
inventory_constrs = m.addConstrs((x_vars[p] <= init_inventory[p] for p in products), name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Demand Fulfillment Plan ---')
    for p in products:
        print(f'Product {p}: Fulfilled {int(round(x_vars[p].X))} units (Demand: {demand[p]}, Inventory: {init_inventory[p]}, Revenue/unit: {revenue[p]})')
else:
    print(f'No optimal solution found. Status: {m.status}')