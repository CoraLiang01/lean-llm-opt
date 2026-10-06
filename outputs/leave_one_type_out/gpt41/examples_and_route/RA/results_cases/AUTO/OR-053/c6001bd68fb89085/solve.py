import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
if not {'ShelfID', 'Capacity'}.issubset(df_capacity.columns):
    raise KeyError('capacity.csv must contain columns: ShelfID, Capacity')
df_capacity['ShelfID'] = df_capacity['ShelfID'].astype(int)
df_capacity['Capacity'] = df_capacity['Capacity'].astype(float)
shelves = df_capacity['ShelfID'].tolist()
capacity_dict = dict(zip(df_capacity['ShelfID'], df_capacity['Capacity']))
df_products = pd.read_csv(products_path, sep=',')
if not {'ProductName', 'Value', 'Weight'}.issubset(df_products.columns):
    raise KeyError('products.csv must contain columns: ProductName, Value, Weight')
df_products['ProductName'] = df_products['ProductName'].astype(int)
df_products['Value'] = df_products['Value'].astype(float)
df_products['Weight'] = df_products['Weight'].astype(float)
products = df_products['ProductName'].tolist()
value_dict = dict(zip(df_products['ProductName'], df_products['Value']))
weight_dict = dict(zip(df_products['ProductName'], df_products['Weight']))
if len(shelves) == 0 or len(products) == 0:
    raise ValueError('No shelves or products found in the input data.')
for sid in shelves:
    if sid not in capacity_dict:
        raise KeyError(f'ShelfID {sid} missing capacity.')
for pid in products:
    if pid not in value_dict or pid not in weight_dict:
        raise KeyError(f'Product {pid} missing value or weight.')
m = gp.Model('BigMartShelfAllocation')
x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in products)) <= capacity_dict[i], name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Shelf Allocation ---')
    for i in shelves:
        shelf_total_value = 0
        shelf_total_weight = 0
        allocations = []
        for j in products:
            units = x[i, j].X
            if units > 1e-06:
                allocations.append((j, int(round(units)), value_dict[j], weight_dict[j]))
                shelf_total_value += value_dict[j] * units
                shelf_total_weight += weight_dict[j] * units
        print(f'Shelf {i}:')
        if allocations:
            for j, units, val, wt in allocations:
                print(f'  Product {j}: {units} units (Value/unit={val}, Weight/unit={wt})')
            print(f'  >> Total value: {shelf_total_value:.2f}, Total weight: {shelf_total_weight:.2f} / Capacity: {capacity_dict[i]:.2f}')
        else:
            print('  No products allocated.')
else:
    print(f'No optimal solution found. Status: {m.status}')