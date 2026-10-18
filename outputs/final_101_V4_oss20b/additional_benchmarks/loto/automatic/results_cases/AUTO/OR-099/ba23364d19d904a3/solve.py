import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
path_warehouses = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv'
path_stores = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv'
path_transport = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv'
df_warehouses = pd.read_csv(path_warehouses, sep=',')
df_stores = pd.read_csv(path_stores, sep=',')
df_transport = pd.read_csv(path_transport, sep=',')
I = df_warehouses['Warehouse (i)'].tolist()
J = df_stores['Store (j)'].tolist()
f = dict(zip(df_warehouses['Warehouse (i)'], df_warehouses['Opening Cost (fi)']))
cap = dict(zip(df_warehouses['Warehouse (i)'], df_warehouses['Capacity (units)']))
d = dict(zip(df_stores['Store (j)'], df_stores['Demand (units, dj)']))
df_transport.set_index('Unnamed: 0', inplace=True)
c = {}
for i in I:
    c[i] = {}
    for j in J:
        row_label = f'W{j}'
        col_label = f'W{i}'
        c[i][j] = df_transport.loc[row_label, col_label]
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(I, vtype=gp.GRB.BINARY)
x = m.addVars(I, J, vtype=gp.GRB.CONTINUOUS)
for j in J:
    m.addConstr(gp.quicksum((x[i, j] for i in I)) == d[j])
for i in I:
    m.addConstr(gp.quicksum((x[i, j] for j in J)) <= cap[i] * y[i])
obj = gp.quicksum((f[i] * y[i] for i in I)) + gp.quicksum((c[i][j] * x[i, j] for i in I for j in J))
m.setObjective(obj, sense=gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('Open warehouses:')
    for i in I:
        if y[i].X > 0.5:
            print(f'  Warehouse {i}: opening cost {f[i]}, capacity {cap[i]}')
    print('Shipments:')
    for i in I:
        for j in J:
            if x[i, j].X > 1e-06:
                print(f'  {x[i, j].X:.2f} units from warehouse {i} to store {j}')
else:
    print(f'No optimal solution found. Status: {m.status}')