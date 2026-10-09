import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv', dtype=str, keep_default_na=False)
capacity_df['SectionID'] = capacity_df['SectionID'].astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
sections = list(capacity_df['SectionID'])
products_df['ProductName'] = products_df['ProductName'].astype(int)
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
products = list(products_df['ProductName'])
capacity_dict = dict(zip(capacity_df['SectionID'], capacity_df['Capacity']))
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(sections) != set(capacity_dict.keys()):
    raise ValueError('Mismatch between section index set and capacity dictionary keys.')
if set(products) != set(value_dict.keys()) or set(products) != set(weight_dict.keys()):
    raise ValueError('Mismatch between product index set and value/weight dictionary keys.')
section_product_keys = [(i, j) for i in sections for j in products]

def solve_problem():
    m = gp.Model('SupermarketProductSelection')
    x_vars = m.addVars(section_product_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in sections for j in products)), gp.GRB.MAXIMIZE)
    for i in sections:
        m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in products)) <= capacity_dict[i], name='cap')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()