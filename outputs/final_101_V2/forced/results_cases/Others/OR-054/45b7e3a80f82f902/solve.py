import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
if df_capacity['ShelfID'].isnull().any() or df_capacity['Capacity'].isnull().any():
    raise ValueError('Missing ShelfID or Capacity in capacity.csv')
shelves = df_capacity['ShelfID'].astype(int).tolist()
capacity_dict = dict(zip(df_capacity['ShelfID'].astype(int), df_capacity['Capacity'].astype(int)))
df_products = pd.read_csv(products_path, sep=',')
if df_products['ProductName'].isnull().any() or df_products['Value'].isnull().any() or df_products['Weight'].isnull().any():
    raise ValueError('Missing ProductName, Value, or Weight in products.csv')
products = df_products['ProductName'].astype(int).tolist()
value_dict = dict(zip(df_products['ProductName'].astype(int), df_products['Value'].astype(int)))
weight_dict = dict(zip(df_products['ProductName'].astype(int), df_products['Weight'].astype(int)))
if len(shelves) != len(set(shelves)):
    raise ValueError('Duplicate ShelfID found in capacity.csv')
if len(products) != len(set(products)):
    raise ValueError('Duplicate ProductName found in products.csv')
if set(capacity_dict.keys()) != set(shelves):
    raise ValueError('Mismatch in ShelfID keys in capacity.csv')
if set(value_dict.keys()) != set(products) or set(weight_dict.keys()) != set(products):
    raise ValueError('Mismatch in ProductName keys in products.csv')
m = gp.Model('BigMart_Shelf_Allocation')
x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in products)) <= capacity_dict[i], name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('--- Shelf Allocation ---')
    for i in shelves:
        shelf_total_value = 0
        shelf_total_weight = 0
        allocation = []
        for j in products:
            units = int(round(x[i, j].X))
            if units > 0:
                allocation.append((j, units, value_dict[j], weight_dict[j]))
                shelf_total_value += value_dict[j] * units
                shelf_total_weight += weight_dict[j] * units
        print(f'Shelf {i}:')
        print(f'  Total value: {shelf_total_value}, Total weight: {shelf_total_weight} / {capacity_dict[i]}')
        if allocation:
            print('  Products placed:')
            for j, units, val, wt in allocation:
                print(f'    Product {j}: {units} units (Value/unit={val}, Weight/unit={wt})')
        else:
            print('  No products placed.')
else:
    print(f'No optimal solution found. Status: {m.status}')