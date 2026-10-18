import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
data_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv'
df = pd.read_csv(data_path, sep=',')
df_books = df[df['Product_Name'].str.strip().str.startswith('Books')].copy()
products = df_books['Product_Name'].tolist()
revenue = df_books['Revenue'].to_dict()
demand = df_books['Demand'].to_dict()
inventory = df_books['Initial Inventory'].to_dict()
m = gp.Model('BookRevenueMaximization')
x_vars = {}
for idx, prod in enumerate(products):
    var = m.addVar(vtype=gp.GRB.INTEGER)
    x_vars[prod] = var
obj = gp.quicksum((revenue[prod] * x_vars[prod] for prod in products))
m.setObjective(obj, gp.GRB.MAXIMIZE)
for idx, prod in enumerate(products):
    m.addConstr(x_vars[prod] <= demand[prod])
    m.addConstr(x_vars[prod] <= inventory[prod])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('\n--- Book Sales Plan ---')
    for prod in products:
        qty = x_vars[prod].X
        rev = revenue[prod] * qty
        print(f'{prod}: {int(qty)} units sold, revenue {rev:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')