import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str)
capacity_df = pd.read_csv(capacity_path, sep=',')
if capacity_df.shape[0] != 1 or 'Capacity' not in capacity_df.columns:
    raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
capacity = int(capacity_df['Capacity'].iloc[0])
product_names = products_df['ProductName'].tolist()
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(product_names) != set(value_dict.keys()) or set(product_names) != set(weight_dict.keys()):
    raise ValueError('Mismatch in product identifiers between Value and Weight columns.')
m = gp.Model('CarSalesInventoryReplenishment')
x = m.addVars(product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in product_names)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in product_names)) <= capacity, name='inventory_capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f}')
    print('--- Optimal Daily Ordering Plan ---')
    for i in product_names:
        xi = x[i].X
        if xi > 0.5:
            print(f'  {i}: {int(round(xi))} units')
    total_weight = sum((weight_dict[i] * x[i].X for i in product_names))
    print(f'Total inventory used: {int(round(total_weight))} / {capacity}')
else:
    print(f'No optimal solution found. Status: {m.status}')