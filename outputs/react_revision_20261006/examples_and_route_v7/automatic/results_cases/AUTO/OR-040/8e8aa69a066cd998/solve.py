import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
areas = products_df['ProductName'].tolist()
if not set(['Value', 'Weight']).issubset(products_df.columns):
    raise ValueError("products.csv must contain 'Value' and 'Weight' columns.")
try:
    value_dict = {row['ProductName']: int(row['Value']) for (_, row) in products_df.iterrows()}
    weight_dict = {row['ProductName']: int(row['Weight']) for (_, row) in products_df.iterrows()}
except Exception as e:
    raise ValueError(f'Error converting Value or Weight columns to int: {e}')
if 'Capacity' not in capacity_df.columns or len(capacity_df) != 1:
    raise ValueError("capacity.csv must contain exactly one row with a 'Capacity' column.")
try:
    capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to int: {e}')
for area in areas:
    if area not in value_dict or area not in weight_dict:
        raise ValueError(f"Missing Value or Weight for area '{area}'.")

def solve_nyc_development(areas, value_dict, weight_dict, capacity):
    m = gp.Model('NYC_Development')
    x_vars = m.addVars(areas, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in areas)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in areas)) <= capacity, name='cap')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_nyc_development(areas, value_dict, weight_dict, capacity)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')