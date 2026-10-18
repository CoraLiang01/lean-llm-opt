import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
customer_df = pd.read_csv('/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv', sep=',')
supply_df = pd.read_csv('/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv', sep=',')
cost_df = pd.read_csv('/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv', sep=',')
warehouses = supply_df['region'].tolist()
stores = customer_df['customer'].tolist()
supply = dict(zip(supply_df['region'], supply_df['supply_capacity']))
demand = dict(zip(customer_df['customer'], customer_df['demand']))
cost_df.set_index('Unnamed: 0', inplace=True)
cost = {w: dict(zip(cost_df.columns, cost_df.loc[w].values)) for w in warehouses}
m = gp.Model('GreenMart_Transportation')
x = m.addVars([(w, c) for w in warehouses for c in stores], vtype=gp.GRB.CONTINUOUS)
m.setObjective(gp.quicksum((cost[w][c] * x[w, c] for w in warehouses for c in stores)), gp.GRB.MINIMIZE)
for w in warehouses:
    m.addConstr(gp.quicksum((x[w, c] for c in stores)) <= supply[w])
for c in stores:
    m.addConstr(gp.quicksum((x[w, c] for w in warehouses)) == demand[c])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.4f}')
    print('\nShipping plan (units):')
    for w in warehouses:
        for c in stores:
            val = x[w, c].X
            if val > 1e-06:
                print(f'  {w} -> {c}: {val:.4f}')
else:
    print(f'No optimal solution found. Status: {m.status}')