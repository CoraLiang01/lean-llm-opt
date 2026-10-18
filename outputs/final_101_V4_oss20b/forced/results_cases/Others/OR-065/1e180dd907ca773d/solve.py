import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv', sep=',')
fixed_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv', sep=',')
transport_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv', sep=',')
facilities = list(fixed_df['Unnamed: 0'])
customers = list(demand_df['customer'])
fixed_costs = dict(zip(fixed_df['Unnamed: 0'], fixed_df['fixed_costs']))
demand = dict(zip(demand_df['customer'], demand_df['demand']))
transport_costs = {}
for _, row in transport_df.iterrows():
    fac = row['Unnamed: 0']
    transport_costs[fac] = {cust: row[cust] for cust in customers}
M = sum(demand.values())
m = gp.Model('UFLP')
y = m.addVars(facilities, vtype=gp.GRB.BINARY)
x = m.addVars([(i, j) for i in facilities for j in customers], vtype=gp.GRB.CONTINUOUS)
obj = gp.quicksum((fixed_costs[i] * y[i] for i in facilities)) + gp.quicksum((transport_costs[i][j] * x[i, j] for i in facilities for j in customers))
m.setObjective(obj, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in facilities)) >= demand[j])
for i in facilities:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\nFacility activation decisions:')
    for i in facilities:
        if y[i].X > 0.5:
            print(f'  {i} opened (y={y[i].X:.0f})')
    print('\nShipment plan (units):')
    for i in facilities:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'  {i} -> {j}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')