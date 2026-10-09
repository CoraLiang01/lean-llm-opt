import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', sep=',', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv', sep=',', dtype=str, keep_default_na=False)
areas = products_df['ProductName'].tolist()
try:
    value_param = products_df.set_index('ProductName')['Value'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'Value' column to float: {e}")
try:
    weight_param = products_df.set_index('ProductName')['Weight'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'Weight' column to float: {e}")
try:
    if capacity_df.shape[0] != 1 or 'Capacity' not in capacity_df.columns:
        raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
    capacity = float(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error reading capacity from capacity.csv: {e}')
m = gp.Model('NYC_RealEstate_Development')
x_vars = m.addVars(areas, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((value_param[area] * x_vars[area] for area in areas)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[area] * x_vars[area] for area in areas)) <= capacity, name='TotalCapacity')
m.optimize()