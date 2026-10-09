import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
for col in ['ProductName', 'Value', 'Weight']:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
products_df['Value'] = products_df['Value'].apply(lambda x: int(x.strip()))
products_df['Weight'] = products_df['Weight'].apply(lambda x: int(x.strip()))
products_df['ProductName'] = products_df['ProductName'].apply(lambda x: x.strip())
product_ids = products_df['ProductName'].tolist()
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row')
capacity_val = int(capacity_df.iloc[0]['Capacity'].strip())
for pid in product_ids:
    if pid not in value_dict or pid not in weight_dict:
        raise KeyError(f"Missing Value or Weight for product '{pid}'")
m = gp.Model('PharmacyDrugOrder')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[pid] * x_vars[pid] for pid in product_ids)) <= capacity_val, name='stock_capacity')
m.optimize()