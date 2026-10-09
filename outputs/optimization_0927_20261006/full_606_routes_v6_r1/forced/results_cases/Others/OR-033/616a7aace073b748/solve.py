import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

def is_baby_product(name):
    return 'baby' in name.strip().casefold()
baby_mask = df['Product Name'].apply(is_baby_product)
baby_df = df[baby_mask].copy()
if baby_df.shape[0] == 0:
    raise ValueError("No 'Baby' products found in the data (no 'Product Name' contains 'Baby').")
product_ids = baby_df['Product Name'].tolist()

def to_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to float: {e}")

def to_int(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to int: {e}")
revenue_dict = dict(zip(baby_df['Product Name'], to_float(baby_df['Revenue'], 'Revenue')))
demand_dict = dict(zip(baby_df['Product Name'], to_int(baby_df['Demand'], 'Demand')))
inventory_dict = dict(zip(baby_df['Product Name'], to_int(baby_df['Initial Inventory'], 'Initial Inventory')))
m = gp.Model('Maximize_Baby_Revenue')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue_dict[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand_dict[i] for i in product_ids), name='')
m.addConstrs((x_vars[i] <= inventory_dict[i] for i in product_ids), name='')
m.optimize()