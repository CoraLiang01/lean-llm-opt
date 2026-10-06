import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
if not {'ShelfID', 'Capacity'}.issubset(df_capacity.columns):
    raise KeyError('capacity.csv must contain columns: ShelfID, Capacity')
shelves = df_capacity['ShelfID'].astype(int).tolist()
capacity_dict = dict(zip(df_capacity['ShelfID'].astype(int), df_capacity['Capacity'].astype(float)))
df_products = pd.read_csv(products_path, sep=',')
if not {'ProductName', 'Value', 'Weight'}.issubset(df_products.columns):
    raise KeyError('products.csv must contain columns: ProductName, Value, Weight')
products = df_products['ProductName'].astype(int).tolist()
value_dict = dict(zip(df_products['ProductName'].astype(int), df_products['Value'].astype(float)))
weight_dict = dict(zip(df_products['ProductName'].astype(int), df_products['Weight'].astype(float)))
if set(shelves) != set(df_capacity['ShelfID'].astype(int)):
    raise ValueError('Mismatch in shelf identifiers between index set and capacity.csv')
if set(products) != set(df_products['ProductName'].astype(int)):
    raise ValueError('Mismatch in product identifiers between index set and products.csv')
for j in products:
    if j not in value_dict or j not in weight_dict:
        raise ValueError(f'Missing value or weight for product {j}')
for i in shelves:
    if i not in capacity_dict:
        raise ValueError(f'Missing capacity for shelf {i}')
keys = [(i, j) for i in shelves for j in products]
m = gp.Model('BigMart_MultiShelf_Knapsack')
x = m.addVars(keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in products)) <= capacity_dict[i], name=f'cap_{i}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in shelves:
        for j in products:
            var = x[i, j]
            print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')