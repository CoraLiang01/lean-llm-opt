import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
for col in ['ProductName', 'Value', 'Weight']:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
area_ids = products_df['ProductName'].tolist()
try:
    value_dict = dict(zip(area_ids, products_df['Value'].astype(float)))
    weight_dict = dict(zip(area_ids, products_df['Weight'].astype(float)))
except Exception as e:
    raise ValueError(f'Error converting Value or Weight columns to float: {e}')
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row')
try:
    total_capacity = float(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to float: {e}')
m = gp.Model('NYC_RealEstate_Development')
x_vars = m.addVars(area_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((value_dict[area] * x_vars[area] for area in area_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[area] * x_vars[area] for area in area_ids)) <= total_capacity, name='TotalCapacity')
m.optimize()