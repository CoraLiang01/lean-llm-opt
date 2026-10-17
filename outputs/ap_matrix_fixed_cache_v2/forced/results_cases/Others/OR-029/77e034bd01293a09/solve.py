import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM20/ZARASales.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

def faux_predicate(name):
    return 'faux' in name.strip().casefold()
faux_mask = df['Product Name'].apply(faux_predicate)
df_faux = df[faux_mask].copy()
if df_faux.empty:
    raise ValueError("No products with 'FAUX' in 'Product Name' found in the data.")
product_ids = df_faux['Product Name'].tolist()

def to_float_or_raise(val, col, pid):
    try:
        return float(val)
    except Exception:
        raise ValueError(f"Non-numeric value '{val}' in column '{col}' for product '{pid}'.")

def to_int_or_raise(val, col, pid):
    try:
        return int(float(val))
    except Exception:
        raise ValueError(f"Non-integer value '{val}' in column '{col}' for product '{pid}'.")
revenue = {}
demand = {}
inventory = {}
fulfill_limit = {}
for idx, row in df_faux.iterrows():
    pid = row['Product Name']
    revenue[pid] = to_float_or_raise(row['Revenue'], 'Revenue', pid)
    demand[pid] = to_int_or_raise(row['Demand'], 'Demand', pid)
    inventory[pid] = to_int_or_raise(row['Initial Inventory'], 'Initial Inventory', pid)
    fulfill_limit[pid] = min(demand[pid], inventory[pid])
m = gp.Model('ZARA_FAUX_Revenue_Maximization')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
for pid in product_ids:
    m.addConstr(x_vars[pid] <= demand[pid], name=f'demand_{pid}')
    m.addConstr(x_vars[pid] <= inventory[pid], name=f'inventory_{pid}')
m.setObjective(gp.quicksum((revenue[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.optimize()