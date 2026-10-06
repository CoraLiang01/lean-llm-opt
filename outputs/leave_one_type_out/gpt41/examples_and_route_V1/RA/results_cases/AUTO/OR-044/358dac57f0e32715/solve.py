import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
sections = capacity_df['SectionID'].astype(int).tolist()
products = products_df['ProductName'].astype(int).tolist()
capacity_dict = dict(zip(capacity_df['SectionID'].astype(int), capacity_df['Capacity'].astype(int)))
value_dict = dict(zip(products_df['ProductName'].astype(int), products_df['Value'].astype(int)))
weight_dict = dict(zip(products_df['ProductName'].astype(int), products_df['Weight'].astype(int)))
if set(sections) != set(capacity_dict.keys()):
    raise ValueError('Mismatch between section index set and capacity dictionary keys.')
if set(products) != set(value_dict.keys()) or set(products) != set(weight_dict.keys()):
    raise ValueError('Mismatch between product index set and value/weight dictionary keys.')
m = gp.Model('SupermarketProductSelection')
x = m.addVars(sections, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in sections for j in products)), gp.GRB.MAXIMIZE)
for i in sections:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in products)) <= capacity_dict[i], name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Stocking Plan (units of each product in each section) ---')
    for i in sections:
        print(f'Section {i} (Capacity {capacity_dict[i]}):')
        for j in products:
            units = int(round(x[i, j].X))
            if units > 0:
                print(f'  Product {j}: {units} units (Value per unit: {value_dict[j]}, Weight per unit: {weight_dict[j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')