import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
subcat_norm = df['Sub Category'].str.strip().str.casefold()
organ_mask = subcat_norm.str.contains('organic')
organ_df = df[organ_mask].copy()
if organ_df.empty:
    raise ValueError("No products found in 'Sub Category' containing 'Organic'.")
organ_ids = organ_df['Sub Category'].tolist()

def safe_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to float: {e}")

def safe_int(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to int: {e}")
revenue_dict = dict(zip(organ_df['Sub Category'], safe_float(organ_df['Revenue'], 'Revenue')))
demand_dict = dict(zip(organ_df['Sub Category'], safe_int(organ_df['Demand'], 'Demand')))
inventory_dict = dict(zip(organ_df['Sub Category'], safe_float(organ_df['Initial Inventory'], 'Initial Inventory')))
for i in organ_ids:
    if i not in revenue_dict or i not in demand_dict or i not in inventory_dict:
        raise KeyError(f"Missing parameter(s) for product '{i}'.")
m = gp.Model('OrganProductRevenueMaximization')
x_vars = m.addVars(organ_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue_dict[i] * x_vars[i] for i in organ_ids)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand_dict[i] for i in organ_ids), name='')
m.addConstrs((x_vars[i] <= inventory_dict[i] for i in organ_ids), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print("--- Fulfillment Plan for 'Organ' Products ---")
    for i in organ_ids:
        print(f'Product: {i}, Fulfilled: {x_vars[i].X:.2f} units, Demand: {demand_dict[i]}, Inventory: {inventory_dict[i]:.2f}, Revenue/unit: {revenue_dict[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')