import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
csv_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv'
df = pd.read_csv(csv_path, sep=',')
products = df['Product Name'].tolist()
revenues = df['Revenue'].tolist()
demands = df['Demand'].tolist()
inventories = df['Initial Inventory'].tolist()
m = gp.Model('FrenchBakeryRevenueMaximization')
var_dict = {}

def sanitize(name):
    return re.sub('\\W+', '_', name)
for prod, rev in zip(products, revenues):
    var_name = f'x_{sanitize(prod)}'
    var = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0)
    var_dict[prod] = var
objective = gp.quicksum((rev * var_dict[prod] for prod, rev in zip(products, revenues)))
m.setObjective(objective, gp.GRB.MAXIMIZE)
for prod, dem, inv in zip(products, demands, inventories):
    var = var_dict[prod]
    m.addConstr(var <= dem)
    m.addConstr(var <= inv)
m.optimize()
print(f'Optimal total revenue: {m.objVal:.2f}')
print('\n--- Sales Plan ---')
for prod, var in var_dict.items():
    print(f'{prod}: {var.X:.2f}')