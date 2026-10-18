import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv'
products_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv'
cap_df = pd.read_csv(capacity_path, sep=',')
prod_df = pd.read_csv(products_path, sep=',')
Cap = dict(zip(cap_df['DisplayID'], cap_df['Capacity']))
W = dict(zip(prod_df['ProductName'], prod_df['Weight']))
V = dict(zip(prod_df['ProductName'], prod_df['Value']))
I = list(Cap.keys())
J = list(W.keys())
m = gp.Model('BoatDisplayOptimization')
x = m.addVars(I, J, vtype=gp.GRB.INTEGER)
m.setObjective(gp.quicksum((V[j] * x[i, j] for i in I for j in J)), gp.GRB.MAXIMIZE)
for i in I:
    m.addConstr(gp.quicksum((W[j] * x[i, j] for j in J)) <= Cap[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Allocation of boats to display areas ---')
    for i in I:
        for j in J:
            if x[i, j].X > 0:
                print(f'Display {i}, Boat "{j}": {int(x[i, j].X)} units')
else:
    print(f'No optimal solution found. Status: {m.status}')