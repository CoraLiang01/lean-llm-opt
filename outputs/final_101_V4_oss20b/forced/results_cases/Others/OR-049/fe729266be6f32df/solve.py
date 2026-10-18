import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv')
products_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv')
capacity = dict(zip(capacity_df['ShelfID'], capacity_df['Capacity']))
weights = dict(zip(products_df['ProductName'], products_df['Weight']))
values = dict(zip(products_df['ProductName'], products_df['Value']))
shelves = list(capacity.keys())
products = list(weights.keys())
m = gp.Model('RetailShelfAllocation')
x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((values[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weights[j] * x[i, j] for j in products)) <= capacity[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    for i in shelves:
        for j in products:
            if x[i, j].X > 0:
                print(f'Shelf {i}, Product {j}: {int(x[i, j].X)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')