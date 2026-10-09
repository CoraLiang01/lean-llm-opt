import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row')
try:
    total_capacity = float(capacity_df.loc[0, 'Capacity'])
except Exception as e:
    raise ValueError(f"Could not convert capacity value to float: {capacity_df.loc[0, 'Capacity']}") from e
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv', sep=',', dtype=str, keep_default_na=False)
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    missing = required_cols - set(products_df.columns)
    raise KeyError(f'Missing columns in products.csv: {missing}')
products_df['ProductName'] = products_df['ProductName'].astype(str)
product_ids = products_df['ProductName'].tolist()
try:
    value_dict = dict(zip(products_df['ProductName'], products_df['Value'].astype(float)))
    weight_dict = dict(zip(products_df['ProductName'], products_df['Weight'].astype(float)))
except Exception as e:
    raise ValueError("Could not convert 'Value' or 'Weight' columns to float in products.csv") from e
m = gp.Model('PharmacyInventoryKnapsack')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in product_ids)) <= total_capacity, name='capacity')
m.optimize()