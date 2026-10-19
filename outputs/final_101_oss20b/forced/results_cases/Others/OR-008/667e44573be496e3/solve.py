import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
customer_demand_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP8/customer_demand.csv'
supply_capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP8/supply_capacity.csv'
transportation_costs_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP8/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_costs = pd.read_csv(transportation_costs_path, sep=',')
customers = df_demand['Customers'].tolist()
suppliers = df_supply['Suppliers'].tolist()
dem = dict(zip(df_demand['Customers'], df_demand['demand']))
cap = dict(zip(df_supply['Suppliers'], df_supply['supply_capacity']))
df_costs = df_costs.set_index('Unnamed: 0')
cost = {}
for s in suppliers:
    cost[s] = {}
    for c in customers:
        cost[s][c] = df_costs.loc[s, c]
m = gp.Model('FreshMart_Transportation')
x = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0)
for s in suppliers:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= cap[s])
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == dem[c])
obj = gp.quicksum((cost[s][c] * x[s, c] for s in suppliers for c in customers))
m.setObjective(obj, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\nShipment plan:')
    for s in suppliers:
        for c in customers:
            val = x[s, c].X
            if val > 1e-06:
                print(f'  {s} -> {c}: {val:.2f} units')
else:
    print(f'Optimization ended with status {m.status}')