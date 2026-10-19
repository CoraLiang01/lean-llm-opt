import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv'
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_products = pd.read_csv(products_path, sep=',')
I = df_capacity['DisplayID'].tolist()
J = df_products['ProductName'].tolist()
capacity = dict(zip(df_capacity['DisplayID'], df_capacity['Capacity']))
value = dict(zip(df_products['ProductName'], df_products['Value']))
weight = dict(zip(df_products['ProductName'], df_products['Weight']))
m = gp.Model('BoatDisplayOptimization')
x = m.addVars(I, J, vtype=gp.GRB.INTEGER)
for i in I:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in J)) <= capacity[i])
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in I for j in J)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('--- Allocation per display area ---')
    for i in I:
        total_value_area = 0
        total_weight_area = 0
        for j in J:
            qty = x[i, j].X
            if qty > 0:
                print(f'  Display {i}: {qty} x {j} (Value: {value[j]}, Weight: {weight[j]})')
                total_value_area += qty * value[j]
                total_weight_area += qty * weight[j]
        print(f'  Total value in area {i}: {total_value_area}')
        print(f'  Total weight in area {i}: {total_weight_area} / {capacity[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')