import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
customer_demand_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv'
supply_capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv'
transportation_costs_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_costs = pd.read_csv(transportation_costs_path, sep=',')
demand = dict(zip(df_demand['Customers'], df_demand['demand']))
supply_capacity = dict(zip(df_supply['Supplier'], df_supply['supply_capacity']))
transport_cost = {}
for _, row in df_costs.iterrows():
    supplier = row['Unnamed: 0']
    transport_cost[supplier] = {}
    for customer in demand.keys():
        transport_cost[supplier][customer] = float(row[customer])
suppliers = list(supply_capacity.keys())
customers = list(demand.keys())
m = gp.Model('Transportation_LP')
x = m.addVars(suppliers, customers)
obj = gp.quicksum((transport_cost[s][c] * x[s, c] for s in suppliers for c in customers))
m.setObjective(obj, gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[s, c] for s in suppliers)) == demand[c] for c in customers))
m.addConstrs((gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s] for s in suppliers))
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.4f}')
    print('\nShipment plan (units):')
    for s in suppliers:
        for c in customers:
            qty = x[s, c].X
            if qty > 1e-06:
                print(f'  {s} -> {c}: {qty:.4f}')
else:
    print(f'Optimization ended with status {m.status}')