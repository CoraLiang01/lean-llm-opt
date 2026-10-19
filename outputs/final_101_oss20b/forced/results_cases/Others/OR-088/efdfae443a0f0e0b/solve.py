import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
cost_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv'
demand_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv'
df_cost = pd.read_csv(cost_path, sep=',')
df_demand = pd.read_csv(demand_path, sep=',')
plants = df_cost['plant'].tolist()
customers = df_demand['customer'].tolist()
fixed_cost = dict(zip(df_cost['plant'], df_cost['fixed_cost']))
capacity = dict(zip(df_cost['plant'], df_cost['capacity']))
demand = dict(zip(df_demand['customer'], df_demand['demand']))
transport_cost = {}
for _, row in df_cost.iterrows():
    plant = row['plant']
    transport_cost[plant] = {}
    for cust in customers:
        transport_cost[plant][cust] = row[cust]
m = gp.Model('CapacitatedFacilityLocation')
x = m.addVars(plants, customers, vtype=gp.GRB.CONTINUOUS)
y = m.addVars(plants, vtype=gp.GRB.BINARY)
obj_fixed = gp.quicksum((fixed_cost[p] * y[p] for p in plants))
obj_ship = gp.quicksum((transport_cost[p][c] * x[p, c] for p in plants for c in customers))
m.setObjective(obj_fixed + obj_ship, gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[p, c] for p in plants)) == demand[c] for c in customers))
m.addConstrs((gp.quicksum((x[p, c] for c in customers)) <= capacity[p] * y[p] for p in plants))
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\nOpened plants:')
    for p in plants:
        if y[p].X > 0.5:
            print(f'  {p} (fixed cost {fixed_cost[p]})')
    print('\nShipping plan:')
    for p in plants:
        for c in customers:
            qty = x[p, c].X
            if qty > 1e-06:
                print(f'  {qty:.2f} units from {p} to {c} (cost {transport_cost[p][c]})')
else:
    print(f'No optimal solution found. Status: {m.status}')