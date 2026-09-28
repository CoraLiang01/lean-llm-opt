import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv'
df = pd.read_csv(csv_path, sep=',')

def is_fashion(name):
    if pd.isnull(name):
        return False
    name_str = str(name).casefold().strip()
    return name_str.startswith('fashion') or 'fashion' in name_str
fashion_mask = df['Product Name'].apply(is_fashion)
fashion_df = df[fashion_mask].copy()
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in fashion_df.columns:
        raise KeyError(f"Required column '{col}' not found in data.")
    if fashion_df[col].isnull().any():
        missing = fashion_df[fashion_df[col].isnull()]
        raise ValueError(f"Missing values in column '{col}' for Fashion products: {missing[['Product Name']].values.flatten()}")
fashion_products = list(fashion_df['Product Name'])
revenue = dict(zip(fashion_df['Product Name'], fashion_df['Revenue']))
demand = dict(zip(fashion_df['Product Name'], fashion_df['Demand']))
inventory = dict(zip(fashion_df['Product Name'], fashion_df['Initial Inventory']))
for pname in fashion_products:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise KeyError(f"Missing coefficients for Fashion product '{pname}'.")
m = gp.Model('FashionRevenueMaximization')
x = m.addVars(fashion_products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in fashion_products)), gp.GRB.MAXIMIZE)
m.addConstrs((x[i] <= demand[i] for i in fashion_products), name='')
m.addConstrs((x[i] <= inventory[i] for i in fashion_products), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Fashion Product Fulfillment Plan ---')
    for i in fashion_products:
        fulfilled = x[i].X
        if fulfilled > 1e-06:
            print(f'Product: {i} | Fulfilled: {fulfilled:.2f} units | Revenue/unit: {revenue[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')