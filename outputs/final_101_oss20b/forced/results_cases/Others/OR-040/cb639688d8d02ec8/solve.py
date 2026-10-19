import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
total_capacity = int(capacity_df.iloc[0]['Capacity'])
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv'
products_df = pd.read_csv(products_path, sep=',')
product_names = products_df['ProductName'].tolist()
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
m = gp.Model('NYC_Property_Development')
x = m.addVars(product_names, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in product_names)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[i] for i in product_names)) <= total_capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.0f}')
    print('\nDevelopment plan per area:')
    for i in product_names:
        if x[i].X > 0:
            print(f'  {i}: {int(x[i].X)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')