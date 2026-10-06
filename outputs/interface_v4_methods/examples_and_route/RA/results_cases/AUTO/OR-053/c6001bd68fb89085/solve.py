import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv', sep=',')
capacity_df['ShelfID'] = capacity_df['ShelfID'].astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(float)
shelves = capacity_df['ShelfID'].tolist()
capacity = dict(zip(capacity_df['ShelfID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv', sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(int)
products_df['Value'] = products_df['Value'].astype(float)
products_df['Weight'] = products_df['Weight'].astype(float)
products = products_df['ProductName'].tolist()
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(shelves) == 0 or len(products) == 0:
    raise ValueError('No shelves or products found in the input data.')
m = gp.Model('BigMartShelfAllocation')
x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i], name='')
m.optimize()