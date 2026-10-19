import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
capacity = int(capacity_df['Capacity'].iloc[0])
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv'
products_df = pd.read_csv(products_path, sep=',')
products = products_df['ProductName'].tolist()
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
m = gp.Model('CarSalesInventory')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0)
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in products)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in products)) <= capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.2f}')
    print('--- Ordering Plan ---')
    for i in products:
        qty = int(x[i].X)
        if qty > 0:
            print(f'{i}: {qty} units')
else:
    print(f'No optimal solution found. Status: {m.status}')