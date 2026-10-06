import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_capacity['ShelfID'] = df_capacity['ShelfID'].astype(int)
shelves = df_capacity['ShelfID'].tolist()
capacity_dict = dict(zip(df_capacity['ShelfID'], df_capacity['Capacity']))
df_products = pd.read_csv(products_path, sep=',')
df_products['ProductName'] = df_products['ProductName'].astype(str).str.strip()
products = df_products['ProductName'].tolist()
value_dict = dict(zip(df_products['ProductName'], df_products['Value']))
weight_dict = dict(zip(df_products['ProductName'], df_products['Weight']))
if len(shelves) != len(set(shelves)):
    raise ValueError('Duplicate ShelfID found in capacity.csv')
if len(products) != len(set(products)):
    raise ValueError('Duplicate ProductName found in products.csv')
if not all((isinstance(capacity_dict[s], (int, float, np.integer, np.floating)) for s in shelves)):
    raise ValueError('Non-numeric capacity found in capacity.csv')
if not all((isinstance(value_dict[p], (int, float, np.integer, np.floating)) for p in products)):
    raise ValueError('Non-numeric value found in products.csv')
if not all((isinstance(weight_dict[p], (int, float, np.integer, np.floating)) for p in products)):
    raise ValueError('Non-numeric weight found in products.csv')
m = gp.Model('RetailShelfAllocation')
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
        allocation = []
        for j in products:
            qty = x[i, j].X
            if qty >= 1e-06:
                allocation.append((j, int(round(qty)), value_dict[j], weight_dict[j]))
                shelf_total_value += value_dict[j] * qty
                shelf_total_weight += weight_dict[j] * qty
        print(f'Shelf {i}:')
        if allocation:
            for j, qty, val, wt in allocation:
                print(f'  {j}: {qty} units (Value/unit={val}, Weight/unit={wt})')
            print(f'  >> Total value: {shelf_total_value:.2f}, Total weight: {shelf_total_weight:.2f} / Capacity: {capacity_dict[i]}')
        else:
            print('  (No products allocated)')
else:
    print(f'No optimal solution found. Status: {m.status}')