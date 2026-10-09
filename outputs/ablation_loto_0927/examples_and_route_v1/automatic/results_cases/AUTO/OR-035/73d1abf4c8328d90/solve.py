import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
if products_df.isnull().any().any():
    raise ValueError('products.csv contains missing values.')
capacity_df = pd.read_csv(capacity_path, sep=',')
if capacity_df.isnull().any().any():
    raise ValueError('capacity.csv contains missing values.')
if capacity_df.shape[0] != 1:
    raise ValueError('capacity.csv must have exactly one row.')
product_ids = products_df['ProductName'].astype(str).tolist()
if len(set(product_ids)) != len(product_ids):
    raise ValueError('Duplicate ProductName entries found in products.csv.')
value_dict = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
weight_dict = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
capacity = int(capacity_df['Capacity'].iloc[0])
m = gp.Model('BakeryBreadOrder')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in product_ids)) <= capacity, name='storage')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Bread Order Plan ---')
    for i in product_ids:
        xi = int(round(x[i].X))
        print(f'{i}: {xi} units (Profit per unit: {value_dict[i]}, Weight per unit: {weight_dict[i]})')
    total_weight = sum((weight_dict[i] * int(round(x[i].X)) for i in product_ids))
    print(f'Total storage used: {total_weight} / {capacity}')
else:
    print(f'No optimal solution found. Status: {m.status}')