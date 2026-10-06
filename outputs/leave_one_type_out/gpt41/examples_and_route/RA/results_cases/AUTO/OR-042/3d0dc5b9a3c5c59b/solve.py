import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv', sep=',')
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] != 1:
    raise ValueError("capacity.csv must have exactly one row with a 'Capacity' column.")
capacity = int(capacity_df['Capacity'].iloc[0])
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv', sep=',')
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError(f'products.csv must contain columns: {required_cols}')
products_df['ProductName'] = products_df['ProductName'].astype(str)
products_df['Value'] = pd.to_numeric(products_df['Value'], errors='raise')
products_df['Weight'] = pd.to_numeric(products_df['Weight'], errors='raise')
products = list(products_df['ProductName'])
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if any((pd.isnull([value[p], weight[p]]) for p in products)):
    raise ValueError('Missing Value or Weight for some products.')
m = gp.Model('PharmacyRestock')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x[p] for p in products)) <= capacity, name='capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f}')
    print('--- Order Plan ---')
    for p in products:
        qty = int(round(x[p].X))
        if qty > 0:
            print(f'{p}: {qty} units (Value/unit: {value[p]}, Weight/unit: {weight[p]})')
    total_weight = sum((weight[p] * int(round(x[p].X)) for p in products))
    print(f'Total weight used: {total_weight} / {capacity}')
else:
    print(f'No optimal solution found. Status: {m.status}')