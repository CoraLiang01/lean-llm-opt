import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
customer_demand_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP4/customer_demand.csv'
supply_capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP4/supply_capacity.csv'
transportation_costs_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP4/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path)
df_supply = pd.read_csv(supply_capacity_path)
df_cost = pd.read_csv(transportation_costs_path)
customers = df_demand['customer'].tolist()
demands = dict(zip(df_demand['customer'], df_demand['demand']))
supply_centers = df_supply['Unnamed: 0'].tolist()
supply_caps = dict(zip(df_supply['Unnamed: 0'], df_supply['supply_capacity']))
cost_matrix = df_cost.set_index('Unnamed: 0')
costs = {s: cost_matrix.loc[s].to_dict() for s in supply_centers}
s_index = {s: i for i, s in enumerate(supply_centers)}
c_index = {c: j for j, c in enumerate(customers)}
m = gp.Model('TransportationCostMinimization')
x = {}
for s in supply_centers:
    for c in customers:
        var = m.addVar()
        x[s, c] = var
m.setObjective(gp.quicksum((costs[s][c] * x[s, c] for s in supply_centers for c in customers)), gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in supply_centers)) == demands[c])
for s in supply_centers:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_caps[s])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\nShipment plan (non‑zero quantities):')
    for s in supply_centers:
        for c in customers:
            val = x[s, c].X
            if val > 1e-06:
                print(f'  {s} -> {c}: {val:.2f} units')
else:
    print(f'Optimization ended with status {m.status}')