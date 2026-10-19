import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP6/customer_demand.csv'
supply_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP6/supply_capacity.csv'
cost_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP6/transportation_costs.csv'
df_demand = pd.read_csv(demand_path, sep=',')
demand = df_demand.set_index('customer')['demand'].to_dict()
df_supply = pd.read_csv(supply_path, sep=',')
supply_capacity = df_supply.set_index('Unnamed: 0')['supply_capacity'].to_dict()
df_cost = pd.read_csv(cost_path, sep=',')
df_cost.set_index('Unnamed: 0', inplace=True)
cost = df_cost.to_dict(orient='index')
W = list(supply_capacity.keys())
C = list(demand.keys())
m = gp.Model('Transportation')
x = {}
for wi, w in enumerate(W):
    for ci, c in enumerate(C):
        var_name = f'x_{wi}_{ci}'
        x[w, c] = m.addVar(vtype=gp.GRB.CONTINUOUS)
m.setObjective(gp.quicksum((cost[w][c] * x[w, c] for w in W for c in C)), gp.GRB.MINIMIZE)
for c in C:
    m.addConstr(gp.quicksum((x[w, c] for w in W)) == demand[c])
for w in W:
    m.addConstr(gp.quicksum((x[w, c] for c in C)) <= supply_capacity[w])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\nShipment plan:')
    for w in W:
        for c in C:
            val = x[w, c].X
            if val > 1e-06:
                print(f'  {val:.2f} units from {w} to {c}')
else:
    print(f'Optimization ended with status {m.status}')