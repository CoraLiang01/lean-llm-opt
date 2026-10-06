import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv', sep=',')
capacity_df['ShelfID'] = capacity_df['ShelfID'].astype(int)
shelves = list(capacity_df['ShelfID'])
capacity_dict = dict(zip(capacity_df['ShelfID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv', sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str)
products = list(products_df['ProductName'])
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(shelves) != set(capacity_dict.keys()):
    raise ValueError('Mismatch between shelves and capacity_dict keys.')
if set(products) != set(value_dict.keys()) or set(products) != set(weight_dict.keys()):
    raise ValueError('Mismatch between products and value/weight dict keys.')
m = gp.Model('RetailShelfAllocation')
x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in products)) <= capacity_dict[i], name=f'shelf_cap_{i}')
m.optimize()