import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
capacity_df = pd.read_csv(capacity_path, sep=',')
areas = products_df['ProductName'].astype(str).tolist()
if products_df['ProductName'].duplicated().any():
    raise ValueError('Duplicate area names found in products.csv; area names must be unique.')
value_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Weight']))
if set(areas) != set(value_dict.keys()) or set(areas) != set(weight_dict.keys()):
    raise ValueError('Mismatch in area keys between Value and Weight columns.')
if capacity_df.shape[0] != 1 or 'Capacity' not in capacity_df.columns:
    raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
total_capacity = int(capacity_df['Capacity'].iloc[0])

def solve_problem(areas, value_dict, weight_dict, total_capacity):
    m = gp.Model('NYC_Development')
    x = m.addVars(areas, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value_dict[a] * x[a] for a in areas)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[a] * x[a] for a in areas)) <= total_capacity, name='cap')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem(areas, value_dict, weight_dict, total_capacity)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')