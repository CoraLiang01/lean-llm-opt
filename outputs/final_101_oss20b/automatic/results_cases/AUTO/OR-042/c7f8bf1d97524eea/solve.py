import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
capacity_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv', sep=',')
capacity = int(capacity_df['Capacity'].iloc[0])
products_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv', sep=',')
product_names = products_df['ProductName'].tolist()
values = products_df['Value'].tolist()
weights = products_df['Weight'].tolist()
m = gp.Model('PharmaRestock')
x = m.addVars(len(product_names), vtype=gp.GRB.INTEGER, lb=0)
m.setObjective(gp.quicksum((values[i] * x[i] for i in range(len(product_names)))), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x[i] for i in range(len(product_names)))) <= capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total benefit: {m.objVal:.2f}')
    print('--- Order quantities per drug type ---')
    for i in range(len(product_names)):
        qty = int(round(x[i].X))
        print(f'{product_names[i]}: {qty} units')
else:
    print(f'No optimal solution found. Status: {m.status}')