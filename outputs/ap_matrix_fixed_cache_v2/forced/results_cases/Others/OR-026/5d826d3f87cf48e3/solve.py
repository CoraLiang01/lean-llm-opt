import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

def is_fashion(name):
    return 'fashion' in name.strip().casefold()
fashion_mask = df['Product Name'].apply(is_fashion)
fashion_df = df.loc[fashion_mask].copy()
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in fashion_df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
fashion_df.set_index('Product Name', inplace=True, drop=False)
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    try:
        fashion_df[col] = fashion_df[col].astype(float)
    except Exception as e:
        raise ValueError(f"Column '{col}' contains non-numeric values or missing data: {e}")
fashion_products = list(fashion_df.index)
revenue = fashion_df['Revenue'].to_dict()
demand = fashion_df['Demand'].to_dict()
inventory = fashion_df['Initial Inventory'].to_dict()
fulfillment_ub = {i: min(demand[i], inventory[i]) for i in fashion_products}
m = gp.Model('FashionRevenueMaximization')
x_vars = m.addVars(fashion_products, lb=0.0, ub=[fulfillment_ub[i] for i in fashion_products], vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in fashion_products)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in fashion_products), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in fashion_products), name='')
m.addConstrs((x_vars[i] >= 0 for i in fashion_products), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print("--- Fulfillment Plan for 'Fashion' Products ---")
    for i in fashion_products:
        print(f'Product: {i}')
        print(f'  Revenue per unit: {revenue[i]:.2f}')
        print(f'  Demand: {demand[i]:.0f}')
        print(f'  Initial Inventory: {inventory[i]:.0f}')
        print(f'  Fulfilled units (x): {x_vars[i].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')