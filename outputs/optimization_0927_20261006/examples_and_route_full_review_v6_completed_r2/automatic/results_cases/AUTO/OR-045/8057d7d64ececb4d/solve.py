import gurobipy as gp
import pandas as pd
import numpy as np
import re
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
required_product_cols = ['ProductName', 'Weight', 'Value']
for col in required_product_cols:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df = products_df.set_index('ProductName', drop=False)
try:
    products_df['Weight'] = products_df['Weight'].astype(int)
    products_df['Value'] = products_df['Value'].astype(int)
except Exception as e:
    raise ValueError(f'Error converting Weight or Value to int: {e}')
products = list(products_df['ProductName'])
weight = products_df['Weight'].to_dict()
value = products_df['Value'].to_dict()
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row')
try:
    capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to int: {e}')
for p in products:
    if p not in weight or p not in value:
        raise KeyError(f"Missing weight or value for product '{p}'")
m = gp.Model('SupermarketRestock')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[p] * x_vars[p] for p in products)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x_vars[p] for p in products)) <= capacity, name='capacity')
m.optimize()