import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
product_name_norm = df['Product Name'].str.strip().str.casefold()
mask_27in = product_name_norm.str.contains('27in')
df_27in = df[mask_27in].copy()
if df_27in.empty:
    raise ValueError("No products found with '27in' in the Product Name.")
df_27in.set_index('Product Name', inplace=True)
try:
    revenue = df_27in['Revenue'].astype(float)
    demand = df_27in['Demand'].astype(int)
    initial_inventory = df_27in['Initial Inventory'].astype(int)
except Exception as e:
    raise ValueError(f'Error converting numeric columns: {e}')
products_27in = list(df_27in.index)
fulfill_upper = pd.DataFrame({'Demand': demand, 'Initial Inventory': initial_inventory}).min(axis=1)
m = gp.Model('27in_Product_Revenue_Maximization')
x_vars = m.addVars(products_27in, vtype=gp.GRB.INTEGER, lb=0, name='')
for i in products_27in:
    m.addConstr(x_vars[i] <= fulfill_upper[i], name=f'ub_{i}')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products_27in)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print("--- Fulfillment Plan for '27in' Products ---")
    for i in products_27in:
        print(f'{i}: Fulfilled {int(round(x_vars[i].X))} units (Demand: {demand[i]}, Inventory: {initial_inventory[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')