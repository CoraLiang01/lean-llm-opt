import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv', sep=',')
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] != 1:
    raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
total_capacity = int(capacity_df['Capacity'].iloc[0])
required_cols = {'ProductName', 'Weight', 'Value'}
if not required_cols.issubset(products_df.columns):
    raise ValueError(f'products.csv must contain columns: {required_cols}')
products_df['ProductName'] = products_df['ProductName'].astype(str)
products = list(products_df['ProductName'])
weights = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
values = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
if any((pd.isnull(weights[p]) or pd.isnull(values[p]) for p in products)):
    raise ValueError('Missing weight or value data for some products.')
m = gp.Model('SupermarketRestock')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x[p] for p in products)) <= total_capacity, name='capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Daily Order Quantities ---')
    for p in products:
        qty = int(round(x[p].X))
        print(f'{p}: {qty} units')
    total_weight = sum((weights[p] * int(round(x[p].X)) for p in products))
    print(f'Total weight used: {total_weight} / {total_capacity}')
else:
    print(f'No optimal solution found. Status: {m.status}')