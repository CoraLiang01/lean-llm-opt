import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv', sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] != 1:
    raise ValueError("capacity.csv must have exactly one row with a 'Capacity' column.")
try:
    total_capacity = int(capacity_df.loc[0, 'Capacity'])
except Exception as e:
    raise ValueError(f"Could not parse 'Capacity' value: {e}")
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError(f'products.csv must contain columns: {required_cols}')
areas = products_df['ProductName'].tolist()
if len(set(areas)) != len(areas):
    raise ValueError('Duplicate ProductName entries found in products.csv.')
try:
    value_param = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
    weight_param = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Could not parse 'Value' or 'Weight' columns as integers: {e}")
if set(value_param.keys()) != set(areas) or set(weight_param.keys()) != set(areas):
    raise ValueError('Mismatch in parameter keys and area identifiers.')
m = gp.Model('NYC_RealEstate_Development')
m.setParam('MIPGap', 0.0001)
x_vars = m.addVars(areas, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((value_param[area] * x_vars[area] for area in areas)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[area] * x_vars[area] for area in areas)) <= total_capacity, name='capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for area in areas:
        print(f'{x_vars[area].VarName} {x_vars[area].X}')
else:
    print(f'Solver status: {m.status}')