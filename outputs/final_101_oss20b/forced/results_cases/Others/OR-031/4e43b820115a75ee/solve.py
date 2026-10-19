import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
data_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv'
df = pd.read_csv(data_path, sep=',')
products = df['Full_Product_Name'].tolist()
revenue_dict = dict(zip(products, df['Revenue']))
demand_dict = dict(zip(products, df['Demand']))
inventory_dict = dict(zip(products, df['Initial Inventory']))
m = gp.Model('DairyRevenueMaximization')
x_vars = {}
for idx, prod in enumerate(products):
    var = m.addVar(vtype=gp.GRB.CONTINUOUS)
    x_vars[prod] = var
obj = gp.quicksum((revenue_dict[prod] * x_vars[prod] for prod in products))
m.setObjective(obj, gp.GRB.MAXIMIZE)
for idx, prod in enumerate(products):
    m.addConstr(x_vars[prod] <= demand_dict[prod])
    m.addConstr(x_vars[prod] <= inventory_dict[prod])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('\nFulfillment plan:')
    for prod in products:
        fulfilled = x_vars[prod].X
        print(f'{prod}: {fulfilled:.2f} units')
else:
    print(f'Optimization did not find an optimal solution. Status: {m.status}')