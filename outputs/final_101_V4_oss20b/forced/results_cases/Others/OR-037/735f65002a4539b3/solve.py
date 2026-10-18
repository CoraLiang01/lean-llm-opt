import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv')
capacity = int(capacity_df['Capacity'].iloc[0])
products_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv')
product_names = products_df['ProductName'].tolist()
profit = dict(zip(product_names, products_df['Value']))
weight = dict(zip(product_names, products_df['Weight']))
m = gp.Model('CarSalesReplenishment')
x = m.addVars(product_names, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((profit[i] * x[i] for i in product_names)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in product_names)) <= capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total profit: {m.objVal:.2f}')
    print('--- Ordering Plan ---')
    for i in product_names:
        qty = int(x[i].X)
        if qty > 0:
            print(f'{i}: {qty} units')
else:
    print(f'No optimal solution found. Status: {m.status}')