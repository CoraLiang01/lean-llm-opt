import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
capacity_df = pd.read_csv(capacity_path, sep=',')
if capacity_df.shape[0] != 1 or 'Capacity' not in capacity_df.columns:
    raise ValueError("capacity.csv must have exactly one row with a 'Capacity' column.")
capacity = int(capacity_df['Capacity'].iloc[0])
product_names = products_df['ProductName'].tolist()
value = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
weight = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
if set(product_names) != set(value.keys()) or set(product_names) != set(weight.keys()):
    raise ValueError('Mismatch in product identifiers between value and weight columns.')
m = gp.Model('CarSalesInventory')
x = m.addVars(product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in product_names)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in product_names)) <= capacity, name='capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Daily Ordering Plan ---')
    for i in product_names:
        qty = int(round(x[i].X))
        if qty > 0:
            print(f'{i}: {qty} units (Profit per unit: {value[i]}, Weight per unit: {weight[i]})')
    total_weight = sum((weight[i] * int(round(x[i].X)) for i in product_names))
    print(f'Total vehicles ordered: {sum((int(round(x[i].X)) for i in product_names))}')
    print(f'Total inventory weight used: {total_weight} / {capacity}')
else:
    print(f'No optimal solution found. Status: {m.status}')