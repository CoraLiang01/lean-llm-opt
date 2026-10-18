import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
fixed_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
cost_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
df_fixed = pd.read_csv(fixed_path, sep=',')
df_costs = pd.read_csv(cost_path, sep=',')
df_fixed.columns = df_fixed.columns.str.strip()
df_costs.columns = df_costs.columns.str.strip()
df_fixed['Service Center'] = df_fixed['Service Center'].astype(str).str.strip()
df_costs['Customer'] = df_costs['Customer'].astype(str).str.strip()
SC = df_fixed['Service Center'].tolist()
C = df_costs['Customer'].tolist()
fixed_cost = dict(zip(SC, df_fixed['Fixed Opening Cost']))
df_costs_indexed = df_costs.set_index('Customer')
service_cost = {(i, j): df_costs_indexed.loc[i, j] for i in C for j in SC}
m = gp.Model('UFLP')
y = m.addVars(SC, vtype=gp.GRB.BINARY)
x = m.addVars(C, SC, vtype=gp.GRB.BINARY)
obj = gp.quicksum((fixed_cost[j] * y[j] for j in SC)) + gp.quicksum((service_cost[i, j] * x[i, j] for i in C for j in SC))
m.setObjective(obj, gp.GRB.MINIMIZE)
for i in C:
    m.addConstr(x.sum(i, '*') == 1)
for i in C:
    for j in SC:
        m.addConstr(x[i, j] <= y[j])
max_cap = 4
for j in SC:
    m.addConstr(x.sum('*', j) <= max_cap)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal objective value: {m.objVal:.4f}')
    print('Opened service centers:')
    for j in SC:
        if y[j].X > 0.5:
            print(f'  {j}')
    print('Customer assignments:')
    for i in C:
        for j in SC:
            if x[i, j].X > 0.5:
                print(f'  {i} -> {j}')
else:
    print(f'No optimal solution found. Status: {m.status}')