import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv')
fixed_cost_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv')
transportation_df = pd.read_csv('/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv')
suppliers = list(fixed_cost_df['Unnamed: 0'])
customers = list(demand_df['customer'])
fixed_cost = dict(zip(fixed_cost_df['Unnamed: 0'], fixed_cost_df['fixed_costs']))
demand = dict(zip(demand_df['customer'], demand_df['demand']))
transportation_cost = {}
for _, row in transportation_df.iterrows():
    supplier = row['Unnamed: 0']
    transportation_cost[supplier] = {}
    for customer in customers:
        transportation_cost[supplier][customer] = row[customer]
total_demand = sum(demand.values())
M = total_demand
m = gp.Model('FixedChargeTransportation')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY)
x = m.addVars([(i, j) for i in suppliers for j in customers], vtype=gp.GRB.CONTINUOUS)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) >= demand[j])
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i])
obj = gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gp.quicksum((transportation_cost[i][j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(obj, gp.GRB.MINIMIZE)
m.optimize()
print(f'Optimal objective value: {m.objVal:.2f}')
print('\nSupplier status and shipments:')
for i in suppliers:
    status = 'OPEN' if y[i].X > 0.5 else 'CLOSED'
    print(f'Supplier {i}: {status}')
    if status == 'OPEN':
        for j in customers:
            qty = x[i, j].X
            if qty > 1e-06:
                print(f'  Ship {qty:.0f} vehicles to {j}')