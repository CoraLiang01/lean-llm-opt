import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_capacity['ShelfID'] = df_capacity['ShelfID'].astype(int)
df_capacity['Capacity'] = df_capacity['Capacity'].astype(float)
shelves = df_capacity['ShelfID'].tolist()
capacity_dict = dict(zip(df_capacity['ShelfID'], df_capacity['Capacity']))
df_products = pd.read_csv(products_path, sep=',')
df_products['ProductName'] = df_products['ProductName'].astype(int)
df_products['Value'] = df_products['Value'].astype(float)
df_products['Weight'] = df_products['Weight'].astype(float)
products = df_products['ProductName'].tolist()
value_dict = dict(zip(df_products['ProductName'], df_products['Value']))
weight_dict = dict(zip(df_products['ProductName'], df_products['Weight']))
if set(shelves) != set(df_capacity['ShelfID']):
    raise ValueError('Mismatch in shelf IDs between index set and capacity data.')
if set(products) != set(df_products['ProductName']):
    raise ValueError('Mismatch in product IDs between index set and product data.')
if not all((j in value_dict and j in weight_dict for j in products)):
    raise ValueError('Missing value or weight data for some products.')
m = gp.Model('BigMartShelfAllocation')
x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in products)) <= capacity_dict[i], name=f'cap_{i}')
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
        if allocations:
            print(f'Shelf {i}:')
            for (j, units, val, wt) in allocations:
                print(f'  Product {j}: {units} units (Value/unit={val}, Weight/unit={wt})')
            print(f'  >> Total value: {shelf_total_value:.2f}, Total weight: {shelf_total_weight:.2f} / Capacity: {capacity_dict[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')