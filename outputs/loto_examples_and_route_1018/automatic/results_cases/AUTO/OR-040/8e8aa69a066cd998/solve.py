import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain columns: 'ProductName', 'Value', 'Weight'.")
areas = products_df['ProductName'].tolist()
try:
    value_dict = dict(zip(products_df['ProductName'], products_df['Value'].astype(int)))
    weight_dict = dict(zip(products_df['ProductName'], products_df['Weight'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting Value or Weight columns to int: {e}')
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain column: 'Capacity'.")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row.')
try:
    total_capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to int: {e}')
for area in areas:
    if area not in value_dict or area not in weight_dict:
        raise KeyError(f"Missing Value or Weight for area '{area}'.")
m = gp.Model('NYC_Property_Development')
x_vars = m.addVars(areas, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[area] * x_vars[area] for area in areas)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[area] * x_vars[area] for area in areas)) <= total_capacity, name='total_capacity')
m.optimize()