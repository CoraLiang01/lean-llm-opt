import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv'
fixed_cost_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv'
transport_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv'
df_demand = pd.read_csv(demand_path)
df_fixed = pd.read_csv(fixed_cost_path)
df_trans = pd.read_csv(transport_path)
customers = df_demand['customer'].tolist()
demand = dict(zip(df_demand['customer'], df_demand['demand']))
suppliers = df_fixed['Unnamed: 0'].tolist()
fixed_cost = dict(zip(df_fixed['Unnamed: 0'], df_fixed['fixed_costs']))
df_trans_indexed = df_trans.set_index('Unnamed: 0')
trans_cost = {}
for s in suppliers:
    for c in customers:
        trans_cost[s, c] = df_trans_indexed.loc[s, c]
M = sum(demand.values())
m = gp.Model('FixedChargeTransportation')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY)
x = m.addVars([(s, c) for s in suppliers for c in customers], vtype=gp.GRB.CONTINUOUS, lb=0)
for s in suppliers:
    y[s].VarName = f'y_{s}'
for s, c in x:
    x[s, c].VarName = f'x_{s}_{c}'
obj = gp.quicksum((fixed_cost[s] * y[s] for s in suppliers)) + gp.quicksum((trans_cost[s, c] * x[s, c] for s in suppliers for c in customers))
m.setObjective(obj, gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand[c])
for s in suppliers:
    for c in customers:
        m.addConstr(x[s, c] <= M * y[s])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('Suppliers opened:')
    for s in suppliers:
        if y[s].X > 0.5:
            print(f'  {s} (fixed cost: {fixed_cost[s]:.2f})')
    print('\nShipment plan:')
    for c in customers:
        print(f'  Branch {c}:')
        for s in suppliers:
            qty = x[s, c].X
            if qty > 1e-06:
                print(f'    From {s}: {qty:.2f} units (transport cost/unit: {trans_cost[s, c]:.2f})')
else:
    print(f'Optimization ended with status {m.status}')