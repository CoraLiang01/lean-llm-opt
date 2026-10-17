import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

def normalize_str(s):
    return re.sub('\\s+', ' ', s.strip()).casefold()
product_name_col = 'Product Name'
product_mask = df[product_name_col].apply(lambda x: '27in' in normalize_str(x))
df_27in = df[product_mask].copy()
if df_27in.empty:
    raise ValueError("No products found with '27in' in the Product Name column.")
df_27in.set_index(product_name_col, inplace=True)
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    if col not in df_27in.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
    df_27in[col] = pd.to_numeric(df_27in[col], errors='raise')
products = list(df_27in.index)
revenue = df_27in['Revenue'].to_dict()
demand = df_27in['Demand'].to_dict()
init_inventory = df_27in['Initial Inventory'].to_dict()
fulfill_limit = {p: min(demand[p], init_inventory[p]) for p in products}
m = gp.Model('27in_Product_Revenue_Maximization')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
for p in products:
    m.addConstr(x_vars[p] <= demand[p], name=f'demand_{p}')
    m.addConstr(x_vars[p] <= init_inventory[p], name=f'inventory_{p}')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print("--- Fulfillment Plan for '27in' Products ---")
    for p in products:
        print(f'Product: {p}')
        print(f'  Units fulfilled: {int(round(x_vars[p].X))} (Demand: {demand[p]}, Initial Inventory: {init_inventory[p]})')
        print(f'  Per-unit revenue: {revenue[p]:.4f}')
else:
    print(f'No optimal solution found. Status: {m.status}')