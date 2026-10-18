import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

def normalize_str(s):
    return s.strip().casefold()
baby_mask = df['Product Name'].apply(lambda x: 'baby' in normalize_str(x))
baby_df = df[baby_mask].copy()
if baby_df.shape[0] == 0:
    raise ValueError("No products found with 'Baby' in the 'Product Name' column.")
baby_df.set_index('Product Name', inplace=True)
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    if col not in baby_df.columns:
        raise KeyError(f"Required column '{col}' not found in the CSV.")
    baby_df[col] = pd.to_numeric(baby_df[col], errors='raise')
product_keys = list(baby_df.index)
revenue = baby_df['Revenue'].to_dict()
demand = baby_df['Demand'].to_dict()
init_inventory = baby_df['Initial Inventory'].to_dict()
m = gp.Model('MaximizeBabyRevenue')
x_vars = m.addVars(product_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in product_keys)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= init_inventory[i] for i in product_keys), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in product_keys), name='')
m.optimize()