import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'SectionID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'SectionID' and 'Capacity' columns.")
capacity_df['SectionID'] = capacity_df['SectionID'].str.strip().astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip().astype(int)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv', sep=',', dtype=str, keep_default_na=False)
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
products_df['ProductName'] = products_df['ProductName'].str.strip().astype(int)
products_df['Value'] = products_df['Value'].str.strip().astype(int)
products_df['Weight'] = products_df['Weight'].str.strip().astype(int)
sections = capacity_df['SectionID'].tolist()
products = products_df['ProductName'].tolist()
section_capacity = dict(zip(capacity_df['SectionID'], capacity_df['Capacity']))
product_value = dict(zip(products_df['ProductName'], products_df['Value']))
product_weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(sections) != set(section_capacity.keys()):
    raise ValueError('Mismatch between section index set and section_capacity keys.')
if set(products) != set(product_value.keys()) or set(products) != set(product_weight.keys()):
    raise ValueError('Mismatch between product index set and product_value/weight keys.')
m = gp.Model('SupermarketProductAllocation')
x_vars = m.addVars(sections, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_value[j] * x_vars[i, j] for i in sections for j in products)), gp.GRB.MAXIMIZE)
for i in sections:
    m.addConstr(gp.quicksum((product_weight[j] * x_vars[i, j] for j in products)) <= section_capacity[i], name=f'capacity_section_{i}')
m.optimize()