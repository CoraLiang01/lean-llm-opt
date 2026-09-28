import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_products = pd.read_csv(products_path, sep=',')
sections = df_capacity['SectionID'].astype(int).tolist()
products = df_products['ProductName'].astype(int).tolist()
capacity = dict(zip(df_capacity['SectionID'].astype(int), df_capacity['Capacity'].astype(int)))
value = dict(zip(df_products['ProductName'].astype(int), df_products['Value'].astype(int)))
weight = dict(zip(df_products['ProductName'].astype(int), df_products['Weight'].astype(int)))
if set(sections) != set(capacity.keys()):
    raise ValueError('Mismatch between section index set and capacity keys.')
if set(products) != set(value.keys()) or set(products) != set(weight.keys()):
    raise ValueError('Mismatch between product index set and value/weight keys.')
m = gp.Model('SupermarketProductSelection')
x = m.addVars(sections, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in sections for j in products)), gp.GRB.MAXIMIZE)
for i in sections:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i], name=f'cap_sec_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Stocking Plan (units of each product in each section) ---')
    for i in sections:
        print(f'Section {i} (Capacity {capacity[i]}):')
        used = 0
        for j in products:
            units = int(round(x[i, j].X))
            if units > 0:
                print(f'  Product {j}: {units} units (Value {value[j]}, Weight {weight[j]})')
                used += weight[j] * units
        print(f'  Total space used: {used} / {capacity[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')