import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
if products_df.isnull().any().any():
    raise ValueError('products.csv contains missing values.')
capacity_df = pd.read_csv(capacity_path, sep=',')
if capacity_df.shape[0] != 1 or capacity_df['Capacity'].isnull().any():
    raise ValueError('capacity.csv must have exactly one row with a valid Capacity value.')
product_ids = products_df['ProductName'].astype(str).tolist()
value_dict = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
weight_dict = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
capacity = int(capacity_df['Capacity'].iloc[0])
if set(product_ids) != set(value_dict.keys()) or set(product_ids) != set(weight_dict.keys()):
    raise ValueError('Mismatch in product identifiers between index set and parameter dictionaries.')
m = gp.Model('BakeryBreadOrder')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in product_ids)) <= capacity, name='storage_capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Bread Order Plan ---')
    for i in product_ids:
        xi = x[i].X
        if xi > 1e-06:
            print(f'{i}: {int(round(xi))} units (Profit/unit: {value_dict[i]}, Weight/unit: {weight_dict[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')