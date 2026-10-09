import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM4/OnlineSalesinUSA.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def contains_4u(val):
    return '4u' in val.strip().casefold()
is_4u = df['Product Name'].apply(contains_4u)
df_4u = df[is_4u].copy()
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df_4u.columns:
        raise KeyError(f"Required column '{col}' not found in filtered DataFrame.")
df_4u.set_index('Product Name', inplace=True, drop=False)
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    try:
        df_4u[col] = pd.to_numeric(df_4u[col], errors='raise')
    except Exception as e:
        raise ValueError(f"Column '{col}' contains non-numeric values: {e}")
products = list(df_4u.index)
revenue = df_4u['Revenue'].to_dict()
demand = df_4u['Demand'].to_dict()
inventory = df_4u['Initial Inventory'].to_dict()
m = gp.Model('4U_Product_Fulfillment')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
for i in products:
    m.addConstr(x_vars[i] <= inventory[i], name=f'inventory_{i}')
    m.addConstr(x_vars[i] <= demand[i], name=f'demand_{i}')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), gp.GRB.MAXIMIZE)
m.optimize()