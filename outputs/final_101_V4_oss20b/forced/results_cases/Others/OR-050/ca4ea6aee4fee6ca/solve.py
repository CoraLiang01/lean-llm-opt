import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/capacity.csv'
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
shelves = list(capacity_df['ShelfID'])
products = list(products_df['ProductName'])
capacity = dict(zip(shelves, capacity_df['Capacity']))
weight = dict(zip(products, products_df['Weight']))
value = dict(zip(products, products_df['Value']))
m = gp.Model('RetailDisplayAllocation')
x = m.addVars(shelves, products, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((value[p] * x[s, p] for s in shelves for p in products)), gp.GRB.MAXIMIZE)
for s in shelves:
    m.addConstr(gp.quicksum((weight[p] * x[s, p] for p in products)) <= capacity[s])
first_product = products[0]
m.addConstr(gp.quicksum((x[s, first_product] for s in shelves)) >= 5)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('\n--- Allocation of products to shelves ---')
    for s in shelves:
        for p in products:
            qty = int(x[s, p].X)
            if qty > 0:
                print(f'Shelf {s}, Product "{p}": {qty} units')
else:
    print(f'No optimal solution found. Status: {m.status}')