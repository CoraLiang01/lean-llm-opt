import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv'
products_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
df_products = pd.read_csv(products_path, sep=',')
capacity = dict(zip(df_capacity['DisplayID'], df_capacity['Capacity']))
weight = dict(zip(df_products['ProductName'], df_products['Weight']))
value = dict(zip(df_products['ProductName'], df_products['Value']))
I = list(capacity.keys())
J = list(weight.keys())
m = gp.Model('BoatDisplayAllocation')
x = m.addVars(I, J, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in I for j in J)), gp.GRB.MAXIMIZE)
for i in I:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in J)) <= capacity[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    for i in I:
        for j in J:
            if x[i, j].X > 0:
                print(f'Display {i}, Boat {j}: {int(x[i, j].X)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')