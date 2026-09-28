import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv', sep=',')
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] != 1:
    raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
total_capacity = int(capacity_df.loc[0, 'Capacity'])
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
m = gp.Model('PharmacyRestock')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in products)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in products)) <= total_capacity, name='capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Order Plan ---')
    for i in products:
        xi = x[i].X
        if xi > 0.5:
            print(f'  {i}: {int(round(xi))} units (Value/unit: {value[i]}, Weight/unit: {weight[i]})')
    total_weight = sum((weight[i] * x[i].X for i in products))
    print(f'Total weight used: {int(round(total_weight))} / {total_capacity}')
else:
    print(f'No optimal solution found. Status: {m.status}')