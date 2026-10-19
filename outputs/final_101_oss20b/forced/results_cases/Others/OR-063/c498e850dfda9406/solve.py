import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv'
fixed_cost_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv'
transport_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
transport_df = pd.read_csv(transport_path, sep=',')
I = list(fixed_cost_df['Unnamed: 0'])
J = list(demand_df['customer'])
fixed_cost = dict(zip(fixed_cost_df['Unnamed: 0'], fixed_cost_df['fixed_costs']))
demand = dict(zip(demand_df['customer'], demand_df['demand']))
transport_df_indexed = transport_df.set_index('Unnamed: 0')
trans_cost = transport_df_indexed.to_dict(orient='index')
m = gp.Model('UFLP')
y = m.addVars(I, vtype=gp.GRB.BINARY)
x = m.addVars(I, J, vtype=gp.GRB.CONTINUOUS)
obj = gp.quicksum((fixed_cost[i] * y[i] for i in I)) + gp.quicksum((trans_cost[i][j] * x[i, j] for i in I for j in J))
m.setObjective(obj, gp.GRB.MINIMIZE)
for j in J:
    m.addConstr(gp.quicksum((x[i, j] for i in I)) >= demand[j])
for i in I:
    for j in J:
        m.addConstr(x[i, j] <= demand[j] * y[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal objective value: {m.objVal:.2f}')
    print('Opened warehouses:')
    for i in I:
        if y[i].X > 0.5:
            print(f'  {i}')
    print('Shipments:')
    for i in I:
        for j in J:
            if x[i, j].X > 1e-06:
                print(f'  {x[i, j].X:.2f} units from {i} to {j}')
else:
    print(f'No optimal solution found. Status: {m.status}')