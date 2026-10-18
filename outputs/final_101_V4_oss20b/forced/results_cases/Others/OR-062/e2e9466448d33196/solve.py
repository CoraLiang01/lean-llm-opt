import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv'
fixed_cost_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv'
transport_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
transport_df = pd.read_csv(transport_path, sep=',')
suppliers = list(fixed_cost_df['Unnamed: 0'])
stores = list(demand_df['Customer'])
fixed_costs = dict(zip(suppliers, fixed_cost_df['fixed_costs']))
demand = dict(zip(stores, demand_df['demand']))
transport_costs = {}
for _, row in transport_df.iterrows():
    supplier = row['Unnamed: 0']
    transport_costs[supplier] = {}
    for store in stores:
        transport_costs[supplier][store] = row[store]
M = sum(demand.values())
m = gp.Model('UFLP')
x = m.addVars(suppliers, stores)
y = m.addVars(suppliers, vtype=gp.GRB.BINARY)
fixed_term = gp.quicksum((fixed_costs[i] * y[i] for i in suppliers))
transport_term = gp.quicksum((transport_costs[i][j] * x[i, j] for i in suppliers for j in stores))
m.setObjective(fixed_term + transport_term, gp.GRB.MINIMIZE)
for j in stores:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j])
for i in suppliers:
    for j in stores:
        m.addConstr(x[i, j] <= M * y[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('Opened suppliers:')
    for i in suppliers:
        if y[i].X > 0.5:
            print(f'  {i} (fixed cost: {fixed_costs[i]:.2f})')
    print('\nShipment plan:')
    for i in suppliers:
        for j in stores:
            qty = x[i, j].X
            if qty > 1e-06:
                print(f'  {qty:.2f} units from {i} to {j}')
else:
    print(f'No optimal solution found. Status: {m.status}')