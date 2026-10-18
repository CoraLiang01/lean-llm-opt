import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv'
fixed_cost_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv'
transport_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
transport_df = pd.read_csv(transport_path, sep=',')
suppliers = list(fixed_cost_df['Unnamed: 0'])
customers = list(demand_df['customer'])
fixed_costs = dict(zip(suppliers, fixed_cost_df['fixed_costs']))
demand = dict(zip(customers, demand_df['demand']))
trans_costs = {}
for _, row in transport_df.iterrows():
    supplier = row['Unnamed: 0']
    trans_costs[supplier] = {}
    for cust in customers:
        trans_costs[supplier][cust] = row[cust]
M = sum(demand.values())
m = gp.Model('UFLP')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY)
x = m.addVars([(i, j) for i in suppliers for j in customers], vtype=gp.GRB.CONTINUOUS)
obj = gp.quicksum((fixed_costs[i] * y[i] for i in suppliers)) + gp.quicksum((trans_costs[i][j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(obj, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j])
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('Open suppliers:')
    for i in suppliers:
        if y[i].X > 0.5:
            print(f'  {i} (fixed cost: {fixed_costs[i]:.2f})')
    print('\nShipment plan:')
    for i in suppliers:
        for j in customers:
            qty = x[i, j].X
            if qty > 1e-06:
                print(f'  {i} -> {j}: {qty:.2f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')