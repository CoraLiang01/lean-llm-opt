import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
if df_capacity.isnull().any().any():
    raise ValueError('Missing values detected in capacity.csv')
df_capacity['ShelfID'] = df_capacity['ShelfID'].astype(int)
df_capacity['Capacity'] = df_capacity['Capacity'].astype(int)
shelf_ids = df_capacity['ShelfID'].tolist()
capacity_dict = dict(zip(df_capacity['ShelfID'], df_capacity['Capacity']))
df_products = pd.read_csv(products_path, sep=',')
if df_products.isnull().any().any():
    raise ValueError('Missing values detected in products.csv')
df_products['ProductName'] = df_products['ProductName'].astype(int)
df_products['Value'] = df_products['Value'].astype(int)
df_products['Weight'] = df_products['Weight'].astype(int)
product_ids = df_products['ProductName'].tolist()
value_dict = dict(zip(df_products['ProductName'], df_products['Value']))
weight_dict = dict(zip(df_products['ProductName'], df_products['Weight']))
if len(set(shelf_ids)) != len(shelf_ids):
    raise ValueError('Duplicate ShelfID detected in capacity.csv')
if len(set(product_ids)) != len(product_ids):
    raise ValueError('Duplicate ProductName detected in products.csv')
if not all((pid in value_dict and pid in weight_dict for pid in product_ids)):
    raise ValueError('Missing value or weight for some products.')
m = gp.Model('BigMart_MultiKnapsack')
x = m.addVars(shelf_ids, product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in shelf_ids for j in product_ids)), gp.GRB.MAXIMIZE)
for i in shelf_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in product_ids)) <= capacity_dict[i], name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Shelf Allocations (x[i,j]) ---')
    for i in shelf_ids:
        shelf_total_value = 0
        shelf_total_weight = 0
        allocations = []
        for j in product_ids:
            units = int(round(x[i, j].X))
            if units > 0:
                allocations.append((j, units, value_dict[j], weight_dict[j]))
                shelf_total_value += value_dict[j] * units
                shelf_total_weight += weight_dict[j] * units
        if allocations:
            print(f'Shelf {i}:')
            for j, units, val, wt in allocations:
                print(f'  Product {j}: {units} units (Value: {val}, Weight: {wt})')
            print(f'  >> Total value: {shelf_total_value}, Total weight: {shelf_total_weight} / Capacity: {capacity_dict[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')