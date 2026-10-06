import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
if products_df.isnull().any().any():
    raise ValueError('products.csv contains missing values.')
capacity_df = pd.read_csv(capacity_path, sep=',')
if capacity_df.shape[0] != 1 or 'Capacity' not in capacity_df.columns:
    raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
capacity = int(capacity_df['Capacity'].iloc[0])
product_ids = products_df['ProductName'].astype(str).tolist()
weights = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
values = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
if set(weights.keys()) != set(product_ids) or set(values.keys()) != set(product_ids):
    raise ValueError('Mismatch in product identifiers between weights/values and product list.')
m = gp.Model('SupermarketStockReplenishment')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[i] * x[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x[i] for i in product_ids)) <= capacity, name='stock_capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Daily Order Quantities ---')
    for i in product_ids:
        qty = int(round(x[i].X))
        print(f'{i}: {qty} units')
    total_weight = sum((weights[i] * int(round(x[i].X)) for i in product_ids))
    print(f'Total stock weight used: {total_weight} / {capacity}')
else:
    print(f'No optimal solution found. Status: {m.status}')