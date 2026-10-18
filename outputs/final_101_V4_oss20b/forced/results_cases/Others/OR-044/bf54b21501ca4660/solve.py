import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv'
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_df = pd.read_csv(products_path, sep=',')
sections = capacity_df['SectionID'].tolist()
capacities = dict(zip(capacity_df['SectionID'], capacity_df['Capacity']))
products = products_df['ProductName'].tolist()
values = dict(zip(products_df['ProductName'], products_df['Value']))
weights = dict(zip(products_df['ProductName'], products_df['Weight']))
m = gp.Model('SupermarketStocking')
x = m.addVars(sections, products, vtype=gp.GRB.INTEGER)
obj = gp.quicksum((values[j] * x[i, j] for i in sections for j in products))
m.setObjective(obj, gp.GRB.MAXIMIZE)
for i in sections:
    m.addConstr(gp.quicksum((weights[j] * x[i, j] for j in products)) <= capacities[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal}')
    print('\nStocking plan by section:')
    for i in sections:
        total_units = 0
        for j in products:
            if x[i, j].X > 0:
                print(f'  Section {i}: Product {j} -> {int(x[i, j].X)} units')
                total_units += int(x[i, j].X)
        print(f'  Total units in section {i}: {total_units}')
else:
    print(f'No optimal solution found. Status: {m.status}')