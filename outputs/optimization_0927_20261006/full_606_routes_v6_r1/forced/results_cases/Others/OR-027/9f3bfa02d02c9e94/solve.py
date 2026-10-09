import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

def is_organ(row):
    return 'organic' in row['Sub Category'].strip().casefold()
organ_mask = df.apply(is_organ, axis=1)
organ_df = df[organ_mask].copy()
if organ_df.shape[0] == 0:
    raise ValueError("No products found in 'Sub Category' containing 'Organic'.")
organ_df.set_index('Sub Category', inplace=True)
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    try:
        organ_df[col] = pd.to_numeric(organ_df[col], errors='raise')
    except Exception as e:
        raise ValueError(f"Column '{col}' contains non-numeric values for selected 'Organ' products.") from e
organ_products = organ_df.index.tolist()
revenue = organ_df['Revenue'].to_dict()
demand = organ_df['Demand'].to_dict()
inventory = organ_df['Initial Inventory'].to_dict()
m = gp.Model('OrganProductRevenueMaximization')
x_vars = m.addVars(organ_products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for i in organ_products:
    m.addConstr(x_vars[i] <= demand[i], name=f'demand_{i}')
    m.addConstr(x_vars[i] <= inventory[i], name=f'inventory_{i}')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in organ_products)), gp.GRB.MAXIMIZE)
m.optimize()