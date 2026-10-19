import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv'
fixed_cost_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv'
transport_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv'
df_demand = pd.read_csv(demand_path, sep=',')
df_fixed = pd.read_csv(fixed_cost_path, sep=',')
df_transport = pd.read_csv(transport_path, sep=',')
customers = df_demand['customer'].astype(str).tolist()
warehouses = df_fixed['Unnamed: 0'].astype(str).tolist()
demand = dict(zip(df_demand['customer'].astype(str), df_demand['demand'].astype(float)))
fixed_cost = dict(zip(df_fixed['Unnamed: 0'].astype(str), df_fixed['fixed_costs'].astype(float)))
transport_df = df_transport.set_index('Unnamed: 0')
transport_cost = {}
for w in warehouses:
    transport_cost[w] = {}
    for c in customers:
        transport_cost[w][c] = float(transport_df.loc[w, c])
M = sum(demand.values())
m = gp.Model('UFLP')
y = m.addVars(warehouses, vtype=gp.GRB.BINARY)
x = m.addVars([(i, j) for i in warehouses for j in customers], vtype=gp.GRB.CONTINUOUS)
obj = gp.quicksum((fixed_cost[i] * y[i] for i in warehouses)) + gp.quicksum((transport_cost[i][j] * x[i, j] for i in warehouses for j in customers))
m.setObjective(obj, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in warehouses)) >= demand[j])
for i in warehouses:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\nWarehouse activation decisions:')
    for i in warehouses:
        print(f"  {i}: {('OPEN' if y[i].X > 0.5 else 'CLOSED')} (cost {fixed_cost[i]:.2f})")
    print('\nShipment plan (units):')
    for i in warehouses:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'  {i} -> {j}: {x[i, j].X:.2f} units (transport cost {transport_cost[i][j]:.2f} per unit)')
else:
    print(f'No optimal solution found. Status: {m.status}')