import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP2/customer_demand.csv'
supply_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP2/supply_capacity.csv'
cost_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP2/transportation_costs.csv'
df_demand = pd.read_csv(demand_path, sep=',')
df_supply = pd.read_csv(supply_path, sep=',')
df_cost = pd.read_csv(cost_path, sep=',')
stores = df_supply['Unnamed: 0'].tolist()
supply = dict(zip(df_supply['Unnamed: 0'], df_supply['supply_capacity']))
customers = df_demand['customer'].tolist()
demand = dict(zip(df_demand['customer'], df_demand['demand']))
cost = {}
for _, row in df_cost.iterrows():
    store_id = row['Unnamed: 0']
    cost[store_id] = {}
    for cust in customers:
        cost[store_id][cust] = float(row[cust])
m = gp.Model('Transportation')
x = m.addVars(stores, customers)
m.setObjective(gp.quicksum((cost[s][c] * x[s, c] for s in stores for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in stores)) == demand[c])
for s in stores:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply[s])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\nShipment plan (non-zero quantities):')
    for s in stores:
        for c in customers:
            val = x[s, c].X
            if val > 1e-06:
                print(f'  {s} -> {c}: {val:.2f}')
else:
    print(f'Optimization ended with status {m.status}')