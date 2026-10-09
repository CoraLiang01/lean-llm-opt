import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
for col in ['ProductName', 'Value', 'Weight']:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
products_df['ProductName'] = products_df['ProductName'].astype(str)
products_df = products_df.set_index('ProductName', drop=False)
try:
    products_df['Value'] = products_df['Value'].astype(float)
    products_df['Weight'] = products_df['Weight'].astype(float)
except Exception as e:
    raise ValueError(f'Failed to convert Value or Weight to float: {e}')
areas = list(products_df.index)
value = products_df['Value'].to_dict()
weight = products_df['Weight'].to_dict()
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row')
try:
    capacity = float(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Failed to convert Capacity to float: {e}')
for i in areas:
    if i not in value or i not in weight:
        raise KeyError(f"Missing value or weight for area '{i}'")
m = gp.Model('NYC_RealEstate_Development')
x_vars = m.addVars(areas, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((value[i] * x_vars[i] for i in areas)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x_vars[i] for i in areas)) <= capacity, name='capacity')
m.optimize()