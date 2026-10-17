import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM12/Salesofsummerclothes.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
df['Product Name'] = df['Product Name'].str.strip()
product_ids = df['Product Name'].tolist()

def safe_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be converted to float: {e}")

def safe_int(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be converted to int: {e}")
revenue = dict(zip(product_ids, safe_float(df['Revenue'], 'Revenue')))
demand = dict(zip(product_ids, safe_int(df['Demand'], 'Demand')))
initial_inventory = dict(zip(product_ids, safe_int(df['Initial Inventory'], 'Initial Inventory')))
for pid in product_ids:
    if pid not in revenue or pid not in demand or pid not in initial_inventory:
        raise KeyError(f"Missing parameter for product '{pid}'.")
m = gp.Model('MaximizeRevenue')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
for pid in product_ids:
    m.addConstr(x_vars[pid] <= demand[pid], name=f'demand_{pid}')
    m.addConstr(x_vars[pid] <= initial_inventory[pid], name=f'inventory_{pid}')
m.setObjective(gp.quicksum((revenue[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.optimize()