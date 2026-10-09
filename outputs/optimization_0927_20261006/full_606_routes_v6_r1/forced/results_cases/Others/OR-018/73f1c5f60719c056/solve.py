import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def is_baby_product(name):
    return name.strip().casefold().startswith('baby')
baby_mask = df['Product Name'].apply(is_baby_product)
baby_df = df[baby_mask].copy()
if baby_df.shape[0] == 0:
    raise ValueError("No 'Baby' products found in the data (Product Name starting with 'Baby').")
baby_df.set_index('Product Name', inplace=True, drop=False)
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    if col not in baby_df.columns:
        raise KeyError(f"Required column '{col}' not found in the CSV.")
    try:
        baby_df[col] = pd.to_numeric(baby_df[col], errors='raise')
    except Exception as e:
        raise ValueError(f"Column '{col}' contains non-numeric values: {e}")
baby_products = list(baby_df.index)
revenue = baby_df['Revenue'].to_dict()
demand = baby_df['Demand'].to_dict()
init_inventory = baby_df['Initial Inventory'].to_dict()
m = gp.Model('BabyProductRevenueMax')
x_vars = m.addVars(baby_products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in baby_products)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in baby_products), name='')
m.addConstrs((x_vars[i] <= init_inventory[i] for i in baby_products), name='')
m.optimize()