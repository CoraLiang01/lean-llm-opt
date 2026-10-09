import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError('products.csv must contain columns: ProductName, Value, Weight')
products_df['ProductName'] = products_df['ProductName'].str.strip()
try:
    products_df['Value'] = products_df['Value'].astype(int)
    products_df['Weight'] = products_df['Weight'].astype(int)
except Exception as e:
    raise ValueError(f'Failed to convert Value or Weight to integer: {e}')
areas = products_df['ProductName'].tolist()
value_param = dict(zip(products_df['ProductName'], products_df['Value']))
weight_param = dict(zip(products_df['ProductName'], products_df['Weight']))
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError('capacity.csv must contain column: Capacity')
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row')
try:
    total_capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Failed to convert Capacity to integer: {e}')
if set(value_param.keys()) != set(weight_param.keys()) or set(value_param.keys()) != set(areas):
    raise ValueError('Mismatch in area identifiers between Value, Weight, and ProductName columns.')
m = gp.Model('NYC_Property_Development')
x_vars = m.addVars(areas, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_param[area] * x_vars[area] for area in areas)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[area] * x_vars[area] for area in areas)) <= total_capacity, name='capacity')
m.optimize()