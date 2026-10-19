import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
customer_demand_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv'
supply_capacity_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv'
transportation_costs_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_cost = pd.read_csv(transportation_costs_path, sep=',')
customers = df_demand['customer'].tolist()
suppliers = df_supply['Unnamed: 0'].tolist()
demand = dict(zip(df_demand['customer'], df_demand['demand']))
supply_capacity = dict(zip(df_supply['Unnamed: 0'], df_supply['supply_capacity']))
df_cost.set_index('Unnamed: 0', inplace=True)
m = gp.Model('Transportation')
x = m.addVars(suppliers, customers)
obj = gp.quicksum((df_cost.loc[s, c] * x[s, c] for s in suppliers for c in customers))
m.setObjective(obj, gp.GRB.MINIMIZE)
for s in suppliers:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_capacity[s])
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand[c])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.4f}')
    print('\nShipment plan (non-zero quantities):')
    for s in suppliers:
        for c in customers:
            qty = x[s, c].X
            if qty > 1e-06:
                print(f'  {s} -> {c}: {qty:.4f}')
else:
    print(f'No optimal solution found. Status: {m.status}')