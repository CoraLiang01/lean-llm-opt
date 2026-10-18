import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
cost_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv'
demand_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv'
cost_df = pd.read_csv(cost_path, sep=',')
demand_df = pd.read_csv(demand_path, sep=',')
plants = cost_df['plant'].tolist()
customers = demand_df['customer'].tolist()
fixed_cost = dict(zip(plants, cost_df['fixed_cost']))
capacity = dict(zip(plants, cost_df['capacity']))
transport_df = cost_df.set_index('plant')[customers]
transport_cost = transport_df.to_dict('index')
demand = dict(zip(customers, demand_df['demand']))
m = gp.Model('UFLP')
x = m.addVars(plants, customers, vtype=gp.GRB.CONTINUOUS)
y = m.addVars(plants, vtype=gp.GRB.BINARY)
fixed_term = gp.quicksum((fixed_cost[i] * y[i] for i in plants))
transport_term = gp.quicksum((transport_cost[i][j] * x[i, j] for i in plants for j in customers))
m.setObjective(fixed_term + transport_term, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in plants)) == demand[j])
for i in plants:
    m.addConstr(gp.quicksum((x[i, j] for j in customers)) <= capacity[i] * y[i])
m.optimize()
print(f'Optimal total cost: {m.objVal:.2f}')
print('\nOpened plants:')
for i in plants:
    if y[i].X > 0.5:
        print(f'  {i} (fixed cost: {fixed_cost[i]})')
print('\nShipments:')
for i in plants:
    for j in customers:
        qty = x[i, j].X
        if qty > 1e-06:
            print(f'  {i} -> {j}: {qty:.2f} units')