import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
if products_df.isnull().any().any():
    raise ValueError('Missing data in products.csv')
capacity_df = pd.read_csv(capacity_path, sep=',')
if capacity_df.shape[0] != 1 or capacity_df['Capacity'].isnull().any():
    raise ValueError('capacity.csv must have exactly one row with a valid Capacity value.')
product_ids = products_df['ProductName'].astype(str).tolist()
value_dict = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
weight_dict = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
capacity = int(capacity_df['Capacity'].iloc[0])
for pid in product_ids:
    if pid not in value_dict or pid not in weight_dict:
        raise ValueError(f"Missing Value or Weight for product '{pid}'.")
m = gp.Model('PharmacyDrugOrder')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in product_ids)) <= capacity, name='stock_capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Drug Order Plan ---')
    for i in product_ids:
        xi = x[i].X
        if xi > 1e-06:
            print(f'  {i}: {int(round(xi))} units (Value/unit: {value_dict[i]}, Weight/unit: {weight_dict[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')