import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
area_ids = products_df['ProductName'].tolist()
try:
    value_col = 'Value'
    weight_col = 'Weight'
    id_col = 'ProductName'
    for col in [id_col, value_col, weight_col]:
        if col not in products_df.columns:
            raise KeyError(f"Column '{col}' not found in products.csv")
    value_param = {}
    weight_param = {}
    for (idx, row) in products_df.iterrows():
        area = str(row[id_col])
        try:
            value = int(row[value_col])
            weight = int(row[weight_col])
        except Exception as e:
            raise ValueError(f"Non-integer value in products.csv at row {idx + 2} for area '{area}': {e}")
        value_param[area] = value
        weight_param[area] = weight
except Exception as e:
    raise RuntimeError(f'Error processing products.csv: {e}')
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row')
try:
    total_capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Non-integer value in capacity.csv: {e}')
if set(area_ids) != set(value_param.keys()) or set(area_ids) != set(weight_param.keys()):
    raise ValueError('Mismatch between area identifiers and parameter keys in products.csv')
m = gp.Model('NYC_Property_Development')
x_vars = m.addVars(area_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_param[area] * x_vars[area] for area in area_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[area] * x_vars[area] for area in area_ids)) <= total_capacity, name='capacity')
m.optimize()