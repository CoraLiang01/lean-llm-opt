import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
if df_capacity.isnull().any().any():
    raise ValueError('Missing values detected in capacity.csv')
df_capacity['ShelfID'] = df_capacity['ShelfID'].astype(int)
shelves = df_capacity['ShelfID'].tolist()
capacity_dict = dict(zip(df_capacity['ShelfID'], df_capacity['Capacity']))
df_products = pd.read_csv(products_path, sep=',')
if df_products.isnull().any().any():
    raise ValueError('Missing values detected in products.csv')
df_products['ProductName'] = df_products['ProductName'].astype(str).str.strip()
products = df_products['ProductName'].tolist()
value_dict = dict(zip(df_products['ProductName'], df_products['Value']))
weight_dict = dict(zip(df_products['ProductName'], df_products['Weight']))
if len(shelves) != len(set(shelves)):
    raise ValueError('Duplicate ShelfID detected in capacity.csv')
if len(products) != len(set(products)):
    raise ValueError('Duplicate ProductName detected in products.csv')
if not all((s in capacity_dict for s in shelves)):
    raise ValueError('Some ShelfIDs missing from capacity_dict')
if not all((p in value_dict and p in weight_dict for p in products)):
    raise ValueError('Some ProductNames missing from value_dict or weight_dict')
m = gp.Model('RetailShelfAllocation')
x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in products)) <= capacity_dict[i], name=f'cap_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Shelf Allocations (x[i,j]) ---')
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