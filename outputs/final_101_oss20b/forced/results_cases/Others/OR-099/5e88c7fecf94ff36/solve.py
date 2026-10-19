import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
path_warehouses = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/PotentialWarehouses_Costs.csv'
path_stores = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/Stores_Demands.csv'
path_transport = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP14/TransportationCost.csv'
df_warehouses = pd.read_csv(path_warehouses)
df_stores = pd.read_csv(path_stores)
df_transport = pd.read_csv(path_transport)
W = df_warehouses['Warehouse (i)'].tolist()
S = df_stores['Store (j)'].tolist()
f = dict(zip(df_warehouses['Warehouse (i)'], df_warehouses['Opening Cost (fi)']))
cap = dict(zip(df_warehouses['Warehouse (i)'], df_warehouses['Capacity (units)']))
d = dict(zip(df_stores['Store (j)'], df_stores['Demand (units, dj)']))
c = {}
for i in W:
    row_label = f'W{i}'
    row = df_transport.loc[df_transport['Unnamed: 0'] == row_label]
    if row.empty:
        raise ValueError(f'Missing transportation row for warehouse {i}')
    for j in S:
        col_name = f'W{j}'
        if col_name not in df_transport.columns:
            raise ValueError(f'Missing transportation column for store {j}')
        cost_value = row[col_name].values[0]
        c[i, j] = int(cost_value)
m = gp.Model('CapacitatedFacilityLocation')
y = m.addVars(W, vtype=gp.GRB.BINARY)
x = m.addVars([(i, j) for i in W for j in S], vtype=gp.GRB.CONTINUOUS)
obj = gp.quicksum((f[i] * y[i] for i in W)) + gp.quicksum((c[i, j] * x[i, j] for i in W for j in S))
m.setObjective(obj, gp.GRB.MINIMIZE)
for j in S:
    m.addConstr(gp.quicksum((x[i, j] for i in W)) == d[j])
for i in W:
    m.addConstr(gp.quicksum((x[i, j] for j in S)) <= cap[i] * y[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\nWarehouse opening decisions:')
    for i in W:
        if y[i].X > 0.5:
            print(f'  Warehouse {i}: OPEN (cost {f[i]})')
    print('\nShipment plan:')
    for i in W:
        for j in S:
            if x[i, j].X > 1e-06:
                print(f'  Ship {x[i, j].X:.2f} units from Warehouse {i} to Store {j} (cost {c[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')