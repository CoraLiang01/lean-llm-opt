import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
cost_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_cost_matrix.csv', sep=',')
sources_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_sources.csv', sep=',')
destinations_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/Mixture_testing/Mixture12/expanded_destinations.csv', sep=',')
S = sources_df['source_id'].tolist()
D = destinations_df['destination_id'].tolist()
supply_units = dict(zip(sources_df['source_id'], sources_df['supply_units']))
demand_units = dict(zip(destinations_df['destination_id'], destinations_df['demand_units']))
cost_df.set_index('source_id', inplace=True)
cost_matrix = cost_df.to_dict(orient='index')
m = gp.Model('Transportation')
x = {}
t = {}
for s in S:
    for d in D:
        x[s, d] = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0)
        t[s, d] = m.addVar(vtype=gp.GRB.INTEGER, lb=0)
obj = gp.quicksum((cost_matrix[s][d] * x[s, d] for s in S for d in D))
m.setObjective(obj, gp.GRB.MINIMIZE)
for s in S:
    m.addConstr(gp.quicksum((x[s, d] for d in D)) <= supply_units[s])
for d in D:
    m.addConstr(gp.quicksum((x[s, d] for s in S)) == demand_units[d])
truck_capacity = 10
for s in S:
    for d in D:
        m.addConstr(x[s, d] <= truck_capacity * t[s, d])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    for s in S:
        for d in D:
            if x[s, d].X > 1e-06:
                print(f'Route {s} -> {d}: cargo={x[s, d].X:.2f}, trucks={t[s, d].X}')
else:
    print(f'No optimal solution found. Status: {m.status}')