import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv'
fixed_cost_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv'
trans_cost_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
trans_cost_df = pd.read_csv(trans_cost_path, sep=',')
suppliers = list(fixed_cost_df['Unnamed: 0'])
fixed_costs = dict(zip(fixed_cost_df['Unnamed: 0'], fixed_cost_df['fixed_costs']))
store_columns = list(trans_cost_df.columns)
store_columns.remove('Unnamed: 0')
stores = store_columns
if len(demand_df) == len(stores):
    demand_dict = dict(zip(stores, demand_df['demand']))
else:
    demand_dict = {stores[i]: demand_df['demand'].iloc[i] for i in range(min(len(demand_df), len(stores)))}
trans_costs = {}
for _, row in trans_cost_df.iterrows():
    supplier = row['Unnamed: 0']
    for store in stores:
        trans_costs[supplier, store] = row[store]
M = sum(demand_dict.values())
m = gp.Model('UFLP')
x = m.addVars(suppliers, stores)
y = m.addVars(suppliers, vtype=gp.GRB.BINARY)
obj = gp.quicksum((fixed_costs[i] * y[i] for i in suppliers)) + gp.quicksum((trans_costs[i, j] * x[i, j] for i in suppliers for j in stores))
m.setObjective(obj, gp.GRB.MINIMIZE)
for j in stores:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand_dict[j])
for i in suppliers:
    for j in stores:
        m.addConstr(x[i, j] <= M * y[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('Open suppliers:')
    for i in suppliers:
        if y[i].X > 0.5:
            print(f'  {i}')
    print('Shipments:')
    for i in suppliers:
        for j in stores:
            if x[i, j].X > 1e-06:
                print(f'  {i} -> {j}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')