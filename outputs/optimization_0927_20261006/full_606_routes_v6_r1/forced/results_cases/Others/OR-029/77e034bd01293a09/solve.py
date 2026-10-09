import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
product_name_norm = df['Product Name'].str.strip().str.casefold()
faux_mask = product_name_norm.str.contains('faux')
df_faux = df.loc[faux_mask].copy()
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df_faux.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
product_keys = df_faux['Product Name'].tolist()

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Could not convert column '{colname}' to float: {e}")

def to_int_series(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Could not convert column '{colname}' to int: {e}")
revenue = to_float_series(df_faux['Revenue'], 'Revenue')
demand = to_int_series(df_faux['Demand'], 'Demand')
initial_inventory = to_int_series(df_faux['Initial Inventory'], 'Initial Inventory')
revenue_dict = dict(zip(product_keys, revenue))
demand_dict = dict(zip(product_keys, demand))
inventory_dict = dict(zip(product_keys, initial_inventory))
m = gp.Model('FauxRevenueMaximization')
x_vars = m.addVars(product_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for i in product_keys:
    m.addConstr(x_vars[i] <= inventory_dict[i], name=f'inventory_{i}')
    m.addConstr(x_vars[i] <= demand_dict[i], name=f'demand_{i}')
m.setObjective(gp.quicksum((revenue_dict[i] * x_vars[i] for i in product_keys)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print("--- Fulfillment Plan for 'FAUX' Products ---")
    for i in product_keys:
        fulfilled = x_vars[i].X
        print(f'Product: {i}')
        print(f'  Units fulfilled: {fulfilled:.2f} (Demand: {demand_dict[i]}, Inventory: {inventory_dict[i]}, Revenue/unit: {revenue_dict[i]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')