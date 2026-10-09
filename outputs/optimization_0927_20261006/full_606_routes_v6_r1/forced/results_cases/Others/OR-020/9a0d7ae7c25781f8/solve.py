import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
df['Product Name'] = df['Product Name'].str.strip()
product_ids = df['Product Name'].tolist()

def to_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be converted to float: {e}")

def to_int(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be converted to int: {e}")
revenue = to_float(df['Revenue'], 'Revenue')
demand = to_int(df['Demand'], 'Demand')
init_inventory = to_int(df['Initial Inventory'], 'Initial Inventory')
revenue_dict = dict(zip(product_ids, revenue))
demand_dict = dict(zip(product_ids, demand))
inventory_dict = dict(zip(product_ids, init_inventory))
for pid in product_ids:
    if pid not in revenue_dict or pid not in demand_dict or pid not in inventory_dict:
        raise ValueError(f"Missing parameter for product '{pid}'.")
m = gp.Model('SupermarketRevenueMaximization')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
demand_constrs = m.addConstrs((x_vars[pid] <= demand_dict[pid] for pid in product_ids), name='')
inventory_constrs = m.addConstrs((x_vars[pid] <= inventory_dict[pid] for pid in product_ids), name='')
m.setObjective(gp.quicksum((revenue_dict[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.optimize()