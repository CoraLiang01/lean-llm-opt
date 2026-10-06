import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    missing = required_cols - set(products_df.columns)
    raise KeyError(f'Missing columns in products.csv: {missing}')
products_df['ProductName'] = products_df['ProductName'].astype(str)
products_df['Value'] = pd.to_numeric(products_df['Value'])
products_df['Weight'] = pd.to_numeric(products_df['Weight'])
product_ids = products_df['ProductName'].tolist()
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
capacity_df = pd.read_csv(capacity_path, sep=',')
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Missing 'Capacity' column in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row')
capacity = int(capacity_df['Capacity'].iloc[0])
m = gp.Model('PharmacyDrugOrder')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in product_ids)) <= capacity, name='stock_capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Drug Order Plan ---')
    for i in product_ids:
        xi = x[i].X
        if xi > 1e-06:
            print(f'{i}: {int(round(xi))} units (Value/unit: {value[i]}, Weight/unit: {weight[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')