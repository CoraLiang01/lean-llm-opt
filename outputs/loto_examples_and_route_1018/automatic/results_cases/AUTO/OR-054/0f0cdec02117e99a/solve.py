import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv', dtype=str, keep_default_na=False)
if 'ShelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'ShelfID' and 'Capacity' columns.")
shelf_ids = capacity_df['ShelfID'].astype(int).tolist()
shelf_capacities = capacity_df.set_index(capacity_df['ShelfID'].astype(int))['Capacity'].astype(int).to_dict()
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
product_ids = products_df['ProductName'].astype(int).tolist()
product_values = products_df.set_index(products_df['ProductName'].astype(int))['Value'].astype(int).to_dict()
product_weights = products_df.set_index(products_df['ProductName'].astype(int))['Weight'].astype(int).to_dict()
if set(shelf_ids) != set(shelf_capacities.keys()):
    raise ValueError('Mismatch between shelf IDs and capacities.')
if set(product_ids) != set(product_values.keys()) or set(product_ids) != set(product_weights.keys()):
    raise ValueError('Mismatch between product IDs and value/weight parameters.')
m = gp.Model('BigMart_MultiShelf_Knapsack')
x_vars = m.addVars(shelf_ids, product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[j] * x_vars[i, j] for i in shelf_ids for j in product_ids)), gp.GRB.MAXIMIZE)
for i in shelf_ids:
    m.addConstr(gp.quicksum((product_weights[j] * x_vars[i, j] for j in product_ids)) <= shelf_capacities[i], name=f'shelf_capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Allocation Plan (x_{ij}) ---')
    for i in shelf_ids:
        shelf_total_weight = 0
        shelf_total_value = 0
        print(f'Shelf {i}:')
        for j in product_ids:
            qty = x_vars[i, j].X
            if qty > 1e-06:
                print(f'  Product {j}: {qty:.0f} units (Value: {product_values[j]}, Weight: {product_weights[j]})')
                shelf_total_weight += qty * product_weights[j]
                shelf_total_value += qty * product_values[j]
        print(f'  >> Total weight: {shelf_total_weight:.0f} / {shelf_capacities[i]}')
        print(f'  >> Total value: {shelf_total_value:.0f}')
else:
    print(f'No optimal solution found. Status: {m.status}')