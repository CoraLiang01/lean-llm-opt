import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
capacity_df = pd.read_csv(capacity_path, sep=',')
areas = products_df['ProductName'].astype(str).tolist()
value_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(float)))
weight_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(float)))
if capacity_df.shape[0] != 1 or 'Capacity' not in capacity_df.columns:
    raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
capacity = float(capacity_df['Capacity'].iloc[0])
if set(value_dict.keys()) != set(areas) or set(weight_dict.keys()) != set(areas):
    raise ValueError('Mismatch in area keys between value_dict, weight_dict, and areas.')
m = gp.Model('NYC_RealEstate_Development')
x = m.addVars(areas, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((value_dict[a] * x[a] for a in areas)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[a] * x[a] for a in areas)) <= capacity, name='capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Development Plan ---')
    for a in areas:
        print(f'{a}: x = {x[a].X:.6f}')
else:
    print(f'No optimal solution found. Status: {m.status}')