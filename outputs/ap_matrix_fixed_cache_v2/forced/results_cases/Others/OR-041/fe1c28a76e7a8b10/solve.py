import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
required_product_cols = ['ProductName', 'Value', 'Weight']
for col in required_product_cols:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
area_ids = products_df['ProductName'].tolist()
try:
    value_series = products_df['Value'].astype(float)
    weight_series = products_df['Weight'].astype(float)
except Exception as e:
    raise ValueError(f'Error converting Value or Weight to float: {e}')
value_dict = dict(zip(area_ids, value_series))
weight_dict = dict(zip(area_ids, weight_series))
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row')
try:
    total_capacity = float(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to float: {e}')
m = gp.Model('NYC_RealEstate_Development')
x_vars = m.addVars(area_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in area_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in area_ids)) <= total_capacity, name='TotalCapacity')
m.optimize()