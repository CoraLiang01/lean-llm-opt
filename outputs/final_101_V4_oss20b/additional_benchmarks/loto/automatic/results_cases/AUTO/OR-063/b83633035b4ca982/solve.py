import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv'
fixed_cost_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv'
transport_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv'
df_demand = pd.read_csv(demand_path, sep=',')
df_fixed = pd.read_csv(fixed_cost_path, sep=',')
df_trans = pd.read_csv(transport_path, sep=',')
customers = df_demand['customer'].tolist()
demand = dict(zip(df_demand['customer'], df_demand['demand']))
warehouses = df_fixed['Unnamed: 0'].tolist()
fixed_costs = dict(zip(df_fixed['Unnamed: 0'], df_fixed['fixed_costs']))
trans_costs = {}
for w in warehouses:
    if w not in df_trans['Unnamed: 0'].values:
        raise KeyError(f'Warehouse {w} not found in transportation_costs.csv')
    row = df_trans.loc[df_trans['Unnamed: 0'] == w].iloc[0]
    trans_costs[w] = {c: row[c] for c in customers}
m = gp.Model('UFLP')
y = m.addVars(warehouses, vtype=gp.GRB.BINARY)
x = m.addVars([(i, j) for i in warehouses for j in customers], vtype=gp.GRB.CONTINUOUS)
obj = gp.quicksum((fixed_costs[i] * y[i] for i in warehouses)) + gp.quicksum((trans_costs[i][j] * x[i, j] for i in warehouses for j in customers))
m.setObjective(obj, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in warehouses)) == demand[j])
for i in warehouses:
    for j in customers:
        m.addConstr(x[i, j] <= demand[j] * y[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('Activated warehouses:')
    for i in warehouses:
        if y[i].X > 0.5:
            print(f'  {i} (fixed cost: {fixed_costs[i]:.2f})')
    print('\nShipment plan:')
    for i in warehouses:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'  {x[i, j].X:.2f} units from {i} to {j}')
else:
    print(f'No optimal solution found. Status: {m.status}')