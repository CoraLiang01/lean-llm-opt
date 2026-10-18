import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
csv_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv'
df = pd.read_csv(csv_path)
aalop_mask = df['Product Name'].str.contains('Aalop', case=False, na=False)
df_aalop = df[aalop_mask]
products = df_aalop['Product Name'].tolist()
revenue = dict(zip(products, df_aalop['Revenue']))
demand = dict(zip(products, df_aalop['Demand']))
initial_inventory = dict(zip(products, df_aalop['Initial Inventory']))
m = gp.Model('AalopRevenueMax')
x = m.addVars(products, lb=0)
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
for p in products:
    m.addConstr(x[p] <= demand[p])
    m.addConstr(x[p] <= initial_inventory[p])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Sales Plan for Aalop Products ---')
    for p in products:
        print(f'{p}: {x[p].X:.2f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')