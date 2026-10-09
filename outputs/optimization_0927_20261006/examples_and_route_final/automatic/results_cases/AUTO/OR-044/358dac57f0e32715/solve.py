import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv', dtype=str, keep_default_na=False)
section_ids = capacity_df['SectionID'].astype(int).tolist()
product_ids = products_df['ProductName'].astype(int).tolist()
capacity_dict = dict(zip(capacity_df['SectionID'].astype(int), capacity_df['Capacity'].astype(int)))
value_dict = dict(zip(products_df['ProductName'].astype(int), products_df['Value'].astype(int)))
weight_dict = dict(zip(products_df['ProductName'].astype(int), products_df['Weight'].astype(int)))
if set(section_ids) != set(capacity_dict.keys()):
    raise ValueError('Mismatch between section IDs and capacity dictionary keys.')
if set(product_ids) != set(value_dict.keys()) or set(product_ids) != set(weight_dict.keys()):
    raise ValueError('Mismatch between product IDs and value/weight dictionary keys.')
m = gp.Model('SupermarketProductSelection')
x_vars = m.addVars(section_ids, product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in section_ids for j in product_ids)), gp.GRB.MAXIMIZE)
for i in section_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_ids)) <= capacity_dict[i], name=f'capacity_sec_{i}')
m.optimize()