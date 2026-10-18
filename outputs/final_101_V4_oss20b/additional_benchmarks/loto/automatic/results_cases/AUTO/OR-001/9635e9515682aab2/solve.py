import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP1/customer_demand.csv'
capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP1/supply_capacity.csv'
cost_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP1/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
capacity_df = pd.read_csv(capacity_path, sep=',')
cost_df = pd.read_csv(cost_path, sep=',')
demand = dict(zip(demand_df['customer'], demand_df['demand']))
capacity = dict(zip(capacity_df['Unnamed: 0'], capacity_df['supply_capacity']))
cost_df.set_index('Unnamed: 0', inplace=True)
customers = list(demand.keys())
cost = {s: dict(zip(customers, cost_df.loc[s, customers].values)) for s in cost_df.index}
S = list(cost_df.index)
C = customers
m = gp.Model('TransportationCostMinimization')
x = m.addVars(S, C)
m.addConstrs((gp.quicksum((x[s, c] for s in S)) == demand[c] for c in C))
m.addConstrs((gp.quicksum((x[s, c] for c in C)) <= capacity[s] for s in S))
m.setObjective(gp.quicksum((cost[s][c] * x[s, c] for s in S for c in C)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\nShipment plan (non-zero quantities):')
    for s in S:
        for c in C:
            if x[s, c].X > 1e-06:
                print(f'  {s} -> {c}: {x[s, c].X:.2f} units')
else:
    print(f'No optimal solution found. Status: {m.status}')