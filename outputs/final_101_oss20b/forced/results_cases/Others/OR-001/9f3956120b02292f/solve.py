import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
customer_demand_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP1/customer_demand.csv'
supply_capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP1/supply_capacity.csv'
transportation_costs_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP1/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_cost = pd.read_csv(transportation_costs_path, sep=',')
demand = dict(zip(df_demand['customer'].astype(str), df_demand['demand'].astype(float)))
supply_capacity = dict(zip(df_supply['Unnamed: 0'].astype(str), df_supply['supply_capacity'].astype(float)))
cost = {}
for _, row in df_cost.iterrows():
    s_id = str(row['Unnamed: 0'])
    cost[s_id] = {}
    for c_id in df_cost.columns[1:]:
        cost[s_id][c_id] = float(row[c_id])
S = list(supply_capacity.keys())
C = list(demand.keys())
m = gp.Model('TransportationCostMinimization')
x = {}
var_index = 0
for s in S:
    for c in C:
        var = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0)
        x[s, c] = var
        var_index += 1
obj = gp.quicksum((cost[s][c] * x[s, c] for s in S for c in C))
m.setObjective(obj, gp.GRB.MINIMIZE)
for c in C:
    m.addConstr(gp.quicksum((x[s, c] for s in S)) == demand[c])
for s in S:
    m.addConstr(gp.quicksum((x[s, c] for c in C)) <= supply_capacity[s])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\nShipment plan (non‑zero quantities):')
    for s in S:
        for c in C:
            val = x[s, c].X
            if val > 1e-06:
                print(f'  {s} -> {c}: {val:.2f}')
else:
    print(f'Optimization ended with status {m.status}')