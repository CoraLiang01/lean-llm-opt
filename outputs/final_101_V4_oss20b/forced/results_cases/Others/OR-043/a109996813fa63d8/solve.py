import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv', sep=',')
capacity = int(capacity_df['Capacity'].iloc[0])
products_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv', sep=',')
products = products_df['ProductName'].tolist()
values = dict(zip(products, products_df['Value']))
weights = dict(zip(products, products_df['Weight']))
m = gp.Model('PharmaInventoryOptimization')
var_map = {}
for prod in products:
    var_name = 'x_' + re.sub('\\W+', '_', prod)
    var_map[prod] = m.addVar(vtype=gp.GRB.INTEGER, lb=0)
m.setObjective(gp.quicksum((values[p] * var_map[p] for p in products)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * var_map[p] for p in products)) <= capacity)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_benefit = m.objVal
    total_weight_used = sum((weights[p] * var_map[p].X for p in products))
    print(f'Optimal total benefit: {total_benefit:.2f}')
    print(f'Total weight used: {total_weight_used:.2f} / {capacity}')
    print('\nOrder plan (units per day):')
    for prod in products:
        qty = var_map[prod].X
        if qty > 0:
            print(f'  {prod}: {int(qty)} units')
else:
    print(f'Optimization did not find an optimal solution. Status: {m.status}')