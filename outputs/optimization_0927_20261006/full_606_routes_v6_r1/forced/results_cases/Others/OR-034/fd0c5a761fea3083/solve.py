import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
product_ids = df['Product Name'].tolist()

def to_float_or_raise(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be converted to float: {e}")

def to_int_or_raise(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be converted to int: {e}")
revenue = to_float_or_raise(df['Revenue'], 'Revenue')
demand = to_int_or_raise(df['Demand'], 'Demand')
init_inventory = to_int_or_raise(df['Initial Inventory'], 'Initial Inventory')
revenue_dict = dict(zip(product_ids, revenue))
demand_dict = dict(zip(product_ids, demand))
inventory_dict = dict(zip(product_ids, init_inventory))
m = gp.Model('FrenchBakeryRevenueMax')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.addConstrs((x_vars[pid] <= demand_dict[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= inventory_dict[pid] for pid in product_ids), name='')
m.setObjective(gp.quicksum((revenue_dict[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.optimize()