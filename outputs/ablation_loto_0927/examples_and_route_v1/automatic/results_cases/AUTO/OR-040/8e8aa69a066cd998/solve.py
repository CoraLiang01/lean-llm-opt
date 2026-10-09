import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
capacity_df = pd.read_csv(capacity_path, sep=',')
areas = products_df['ProductName'].astype(str).tolist()
value_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(int)))
weight_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(int)))
if capacity_df.shape[0] != 1 or 'Capacity' not in capacity_df.columns:
    raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
capacity = int(capacity_df['Capacity'].iloc[0])
if set(value_dict.keys()) != set(areas) or set(weight_dict.keys()) != set(areas):
    raise ValueError('Mismatch in area keys between value and weight dictionaries.')
m = gp.Model('NYC_Property_Development')
x = m.addVars(areas, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[a] * x[a] for a in areas)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[a] * x[a] for a in areas)) <= capacity, name='capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f}')
    print('--- Optimal Daily Development Plan ---')
    for a in areas:
        print(f'{a}: {int(round(x[a].X))} units/day')
else:
    print(f'No optimal solution found. Status: {m.status}')