import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv'
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_products = pd.read_csv(products_path, sep=',')
df_capacity['ShelfID'] = df_capacity['ShelfID'].astype(int)
df_capacity['Capacity'] = df_capacity['Capacity'].astype(int)
df_products['ProductName'] = df_products['ProductName'].astype(int)
df_products['Value'] = df_products['Value'].astype(int)
df_products['Weight'] = df_products['Weight'].astype(int)
capacity_dict = dict(zip(df_capacity['ShelfID'], df_capacity['Capacity']))
value_dict = dict(zip(df_products['ProductName'], df_products['Value']))
weight_dict = dict(zip(df_products['ProductName'], df_products['Weight']))
shelves = list(capacity_dict.keys())
products = list(value_dict.keys())
m = gp.Model('BigMartSalesAllocation')
x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((value_dict[p] * x[s, p] for s in shelves for p in products)), gp.GRB.MAXIMIZE)
for s in shelves:
    m.addConstr(gp.quicksum((weight_dict[p] * x[s, p] for p in products)) <= capacity_dict[s])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('\n--- Allocation per Shelf ---')
    for s in shelves:
        total_weight = 0
        total_value = 0
        print(f'Shelf {s} (Capacity: {capacity_dict[s]}):')
        for p in products:
            qty = int(x[s, p].X)
            if qty > 0:
                print(f'  Product {p}: {qty} units (Value: {value_dict[p]}, Weight: {weight_dict[p]})')
                total_weight += weight_dict[p] * qty
                total_value += value_dict[p] * qty
        print(f'  Total weight used: {total_weight}')
        print(f'  Total value contributed: {total_value}\n')
else:
    print(f'No optimal solution found. Status: {m.status}')