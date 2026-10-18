import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv'
fixed_cost_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv'
transport_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
fixed_df = pd.read_csv(fixed_cost_path, sep=',')
transport_df = pd.read_csv(transport_path, sep=',')
C = list(demand_df['customer'])
S = list(fixed_df['Unnamed: 0'])
demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost = dict(zip(fixed_df['Unnamed: 0'], fixed_df['fixed_costs']))
tc = {}
for _, row in transport_df.iterrows():
    supplier = row['Unnamed: 0']
    tc[supplier] = {}
    for store in C:
        tc[supplier][store] = row[store]
M = sum(demand.values())
m = gp.Model('UFLP')
y = m.addVars(S, vtype=gp.GRB.BINARY)
x = m.addVars(S, C, vtype=gp.GRB.CONTINUOUS)
for j in C:
    m.addConstr(gp.quicksum((x[i, j] for i in S)) == demand[j])
for i in S:
    for j in C:
        m.addConstr(x[i, j] <= M * y[i])
obj = gp.quicksum((fixed_cost[i] * y[i] for i in S)) + gp.quicksum((tc[i][j] * x[i, j] for i in S for j in C))
m.setObjective(obj, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Total minimum cost: {m.objVal:.2f}')
    print('Open suppliers:')
    for i in S:
        if y[i].X > 0.5:
            print(f'  {i}')
    print('Shipment plan (units):')
    for i in S:
        for j in C:
            if x[i, j].X > 1e-06:
                print(f'  {i} -> {j}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')