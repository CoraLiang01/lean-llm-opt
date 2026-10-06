import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
capacity_df = pd.read_csv(capacity_path, sep=',')
areas = products_df['ProductName'].astype(str).tolist()
value = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
weight = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
if capacity_df.shape[0] != 1 or 'Capacity' not in capacity_df.columns:
    raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
capacity = int(capacity_df['Capacity'].iloc[0])
for area in areas:
    if area not in value or area not in weight:
        raise ValueError(f"Missing value or weight for area '{area}'.")
m = gp.Model('NYCPropertyDevelopment')
x = m.addVars(areas, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[area] * x[area] for area in areas)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[area] * x[area] for area in areas)) <= capacity, name='capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Daily Development Plan ---')
    for area in areas:
        xi = x[area].X
        print(f'{area}: {int(round(xi))} units per day')
else:
    print(f'No optimal solution found. Status: {m.status}')