import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv'
fixed_cost_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv'
transport_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv'
df_demand = pd.read_csv(demand_path)
df_fixed = pd.read_csv(fixed_cost_path)
df_trans = pd.read_csv(transport_path)
stores = df_demand['customer'].tolist()
demand = dict(zip(df_demand['customer'], df_demand['demand']))
suppliers = df_fixed['Unnamed: 0'].tolist()
fixed_cost = dict(zip(df_fixed['Unnamed: 0'], df_fixed['fixed_costs']))
trans_cost = {}
for _, row in df_trans.iterrows():
    supplier = row['Unnamed: 0']
    trans_cost[supplier] = {}
    for store in stores:
        trans_cost[supplier][store] = row[store]
M = sum(demand.values())
m = gp.Model('UFLP')
x = m.addVars(suppliers, stores, vtype=gp.GRB.CONTINUOUS)
y = m.addVars(suppliers, vtype=gp.GRB.BINARY)
obj_fixed = gp.quicksum((fixed_cost[i] * y[i] for i in suppliers))
obj_trans = gp.quicksum((trans_cost[i][j] * x[i, j] for i in suppliers for j in stores))
m.setObjective(obj_fixed + obj_trans, gp.GRB.MINIMIZE)
for j in stores:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j])
for i in suppliers:
    for j in stores:
        m.addConstr(x[i, j] <= M * y[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal objective value: {m.objVal:.2f}\n')
    print('Open suppliers:')
    for i in suppliers:
        if y[i].X > 0.5:
            print(f'  {i} (fixed cost: {fixed_cost[i]:.2f})')
    print('\nShipment plan:')
    for i in suppliers:
        for j in stores:
            qty = x[i, j].X
            if qty > 1e-06:
                print(f'  {qty:.2f} units from {i} to {j} (cost per unit: {trans_cost[i][j]:.2f})')
else:
    print(f'Optimization ended with status {m.status}')