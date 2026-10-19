import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
cost_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv'
warehouse_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv'
demand_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv'
df_cost = pd.read_csv(cost_path, sep=',')
df_warehouse = pd.read_csv(warehouse_path, sep=',')
df_demand = pd.read_csv(demand_path, sep=',')
W = list(df_warehouse['Warehouse ID'])
C = list(df_demand['Customer ID'])
fixed_cost = dict(zip(df_warehouse['Warehouse ID'], df_warehouse['Fixed_Cost']))
capacity = dict(zip(df_warehouse['Warehouse ID'], df_warehouse['Capacity']))
demand = dict(zip(df_demand['Customer ID'], df_demand['Demand']))
df_cost_indexed = df_cost.set_index('Warehouse ID')
cost = {w: df_cost_indexed.loc[w, C].to_dict() for w in W}
m = gp.Model('Capacitated_Facility_Location')
y = m.addVars(W, vtype=gp.GRB.BINARY)
x = m.addVars([(w, c) for w in W for c in C], vtype=gp.GRB.CONTINUOUS)
obj = gp.quicksum((fixed_cost[w] * y[w] for w in W)) + gp.quicksum((cost[w][c] * x[w, c] for w in W for c in C))
m.setObjective(obj, gp.GRB.MINIMIZE)
for c in C:
    m.addConstr(gp.quicksum((x[w, c] for w in W)) == demand[c])
for w in W:
    m.addConstr(gp.quicksum((x[w, c] for c in C)) <= capacity[w] * y[w])
m.optimize()
print(f'Optimal objective value: {m.objVal:.2f}\n')
print('Opened warehouses:')
for w in W:
    if y[w].X > 0.5:
        print(f'  {w} (Fixed Cost: {fixed_cost[w]}, Capacity: {capacity[w]})')
print('\nCustomer assignments (non-zero amounts):')
for w in W:
    for c in C:
        if x[w, c].X > 1e-06:
            print(f'  {c} served by {w}: {x[w, c].X:.2f} units (Cost per unit: {cost[w][c]})')