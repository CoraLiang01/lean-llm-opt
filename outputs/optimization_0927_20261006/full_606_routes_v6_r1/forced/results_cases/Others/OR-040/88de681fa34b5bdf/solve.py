import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
areas = products_df['ProductName'].tolist()
try:
    value_param = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
    weight_param = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f'Error converting Value or Weight columns to int: {e}')
if 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'Capacity' column.")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row.')
try:
    total_capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to int: {e}')
m = gp.Model('NYC_Property_Development')
x_vars = m.addVars(areas, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_param[area] * x_vars[area] for area in areas)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[area] * x_vars[area] for area in areas)) <= total_capacity, name='TotalCapacity')
m.optimize()