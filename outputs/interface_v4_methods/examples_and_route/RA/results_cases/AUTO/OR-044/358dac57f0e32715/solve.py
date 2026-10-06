import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv', sep=',')
capacity_df['SectionID'] = capacity_df['SectionID'].astype(int)
products_df['ProductName'] = products_df['ProductName'].astype(int)
sections = capacity_df['SectionID'].tolist()
products = products_df['ProductName'].tolist()
capacity = dict(zip(capacity_df['SectionID'], capacity_df['Capacity']))
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(sections) != set(capacity.keys()):
    raise ValueError('SectionID mismatch between index set and capacity parameter.')
if set(products) != set(value.keys()) or set(products) != set(weight.keys()):
    raise ValueError('ProductName mismatch between index set and value/weight parameters.')
m = gp.Model('SupermarketProductSelection')
x = m.addVars(sections, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in sections for j in products)), gp.GRB.MAXIMIZE)
for i in sections:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i], name=f'cap_sec_{i}')
m.optimize()