import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
if not {'ShelfID', 'Capacity'}.issubset(df_capacity.columns):
    raise KeyError('capacity.csv must contain columns: ShelfID, Capacity')
df_capacity['ShelfID'] = df_capacity['ShelfID'].astype(int)
shelves = df_capacity['ShelfID'].tolist()
capacity_dict = dict(zip(df_capacity['ShelfID'], df_capacity['Capacity']))
df_products = pd.read_csv(products_path, sep=',')
if not {'ProductName', 'Value', 'Weight'}.issubset(df_products.columns):
    raise KeyError('products.csv must contain columns: ProductName, Value, Weight')
df_products['ProductName'] = df_products['ProductName'].astype(str).str.strip()
products = df_products['ProductName'].tolist()
value_dict = dict(zip(df_products['ProductName'], df_products['Value']))
weight_dict = dict(zip(df_products['ProductName'], df_products['Weight']))
if len(shelves) != len(capacity_dict):
    raise ValueError('Mismatch in number of shelves and capacity entries.')
if len(products) != len(value_dict) or len(products) != len(weight_dict):
    raise ValueError('Mismatch in number of products and value/weight entries.')
m = gp.Model('ShelfProductAllocation')
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
        print(f'Shelf {i}:')
        for j in products:
            units = x[i, j].X
            if units > 1e-06:
                val = value_dict[j] * units
                wt = weight_dict[j] * units
                shelf_total_value += val
                shelf_total_weight += wt
                print(f'  {j}: {int(round(units))} units (Value: {val:.2f}, Weight: {wt:.2f})')
        print(f'  >> Total Value: {shelf_total_value:.2f}, Total Weight: {shelf_total_weight:.2f} / Capacity: {capacity_dict[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')