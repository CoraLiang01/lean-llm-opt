import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
capacity_df = pd.read_csv(capacity_path, sep=',')
if 'ProductName' not in products_df.columns or 'Weight' not in products_df.columns or 'Value' not in products_df.columns:
    raise KeyError("products.csv must contain columns: 'ProductName', 'Weight', 'Value'.")
product_names = products_df['ProductName'].astype(str).tolist()
weight = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
value = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
if set(product_names) != set(weight.keys()) or set(product_names) != set(value.keys()):
    raise ValueError('Mismatch in product identifiers between index and parameter columns.')
if 'Capacity' not in capacity_df.columns or len(capacity_df) != 1:
    raise KeyError("capacity.csv must contain exactly one row with column 'Capacity'.")
capacity = int(capacity_df.iloc[0]['Capacity'])
m = gp.Model('SupermarketStockReplenishment')
x = m.addVars(product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in product_names)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in product_names)) <= capacity, name='StockCapacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Daily Order Quantities ---')
    for i in product_names:
        qty = int(round(x[i].X))
        print(f'{i}: {qty} units')
    used_capacity = sum((weight[i] * int(round(x[i].X)) for i in product_names))
    print(f'Total stock space used: {used_capacity} / {capacity}')
else:
    print(f'No optimal solution found. Status: {m.status}')