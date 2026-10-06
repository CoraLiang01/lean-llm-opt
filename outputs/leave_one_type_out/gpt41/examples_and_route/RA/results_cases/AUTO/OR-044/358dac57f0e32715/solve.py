import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv', sep=',')
sections = capacity_df['SectionID'].astype(int).tolist()
section_cap = dict(zip(capacity_df['SectionID'].astype(int), capacity_df['Capacity']))
products = products_df['ProductName'].astype(int).tolist()
product_value = dict(zip(products_df['ProductName'].astype(int), products_df['Value']))
product_weight = dict(zip(products_df['ProductName'].astype(int), products_df['Weight']))
if len(sections) != len(set(sections)):
    raise ValueError('Duplicate SectionID found in capacity.csv')
if len(products) != len(set(products)):
    raise ValueError('Duplicate ProductName found in products.csv')
if set(section_cap.keys()) != set(sections):
    raise ValueError('Mismatch in SectionID keys')
if set(product_value.keys()) != set(products) or set(product_weight.keys()) != set(products):
    raise ValueError('Mismatch in ProductName keys')
m = gp.Model('SupermarketProductSelection')
x = m.addVars(sections, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_value[j] * x[i, j] for i in sections for j in products)), gp.GRB.MAXIMIZE)
for i in sections:
    m.addConstr(gp.quicksum((product_weight[j] * x[i, j] for j in products)) <= section_cap[i], name=f'cap_sec_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Stocking Plan (units of each product in each section) ---')
    for i in sections:
        print(f'Section {i} (Capacity {section_cap[i]}):')
        for j in products:
            units = int(round(x[i, j].X))
            if units > 0:
                print(f'  Product {j}: {units} units (Value {product_value[j]}, Weight {product_weight[j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')