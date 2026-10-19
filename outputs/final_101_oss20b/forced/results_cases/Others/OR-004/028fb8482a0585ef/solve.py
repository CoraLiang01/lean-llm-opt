import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
path_demand = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv'
path_supply = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv'
path_transport = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv'
df_demand = pd.read_csv(path_demand, sep=',')
df_supply = pd.read_csv(path_supply, sep=',')
df_transport = pd.read_csv(path_transport, sep=',')
customers = df_demand['customer'].tolist()
demands = dict(zip(df_demand['customer'], df_demand['demand']))
supply_centers = df_supply['Unnamed: 0'].tolist()
supply_caps = dict(zip(df_supply['Unnamed: 0'], df_supply['supply_capacity']))
df_transport.set_index('Unnamed: 0', inplace=True)
transport_costs = df_transport
m = gp.Model('TransportationProblem')
x = m.addVars(supply_centers, customers, vtype=gp.GRB.CONTINUOUS, lb=0)
m.setObjective(gp.quicksum((transport_costs.loc[s, c] * x[s, c] for s in supply_centers for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in supply_centers)) == demands[c])
for s in supply_centers:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_caps[s])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}\n')
    print('Shipment plan (units):')
    for s in supply_centers:
        for c in customers:
            val = x[s, c].X
            if val > 1e-06:
                print(f'  {s} -> {c}: {val:.2f}')
else:
    print(f'Optimization ended with status {m.status}')