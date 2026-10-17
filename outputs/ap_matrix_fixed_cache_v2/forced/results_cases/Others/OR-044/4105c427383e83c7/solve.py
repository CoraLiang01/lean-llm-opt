import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
capacity_df['SectionID'] = capacity_df['SectionID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
capacity_df['SectionID'] = capacity_df['SectionID'].astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv', sep=',', dtype=str, keep_default_na=False)
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
products_df['ProductName'] = products_df['ProductName'].astype(int)
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
sections = capacity_df['SectionID'].tolist()
products = products_df['ProductName'].tolist()
capacity_dict = dict(zip(capacity_df['SectionID'], capacity_df['Capacity']))
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(sections) != set(capacity_dict.keys()):
    raise ValueError('Mismatch between section index set and capacity_dict keys.')
if set(products) != set(value_dict.keys()) or set(products) != set(weight_dict.keys()):
    raise ValueError('Mismatch between product index set and value/weight dict keys.')
m = gp.Model('SupermarketProductSelection')
x_vars = m.addVars(sections, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in sections for j in products)), gp.GRB.MAXIMIZE)
for i in sections:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in products)) <= capacity_dict[i], name=f'capacity_sec_{i}')
m.optimize()