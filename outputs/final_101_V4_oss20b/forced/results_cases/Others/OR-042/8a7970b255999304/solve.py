import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv', sep=',')
capacity = int(capacity_df['Capacity'].iloc[0])
products_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv', sep=',')
product_names = products_df['ProductName'].tolist()
values = products_df['Value'].tolist()
weights = products_df['Weight'].tolist()
m = gp.Model('PharmaRestock')
x = m.addVars(product_names, vtype=gp.GRB.INTEGER)
obj = gp.quicksum((values[i] * x[product_names[i]] for i in range(len(product_names))))
m.setObjective(obj, gp.GRB.MAXIMIZE)
capacity_constraint = gp.quicksum((weights[i] * x[product_names[i]] for i in range(len(product_names))))
m.addConstr(capacity_constraint <= capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal}')
    print('Order quantities:')
    for p in product_names:
        qty = int(x[p].X)
        if qty > 0:
            print(f'  {p}: {qty}')
else:
    print(f'No optimal solution found. Status: {m.status}')