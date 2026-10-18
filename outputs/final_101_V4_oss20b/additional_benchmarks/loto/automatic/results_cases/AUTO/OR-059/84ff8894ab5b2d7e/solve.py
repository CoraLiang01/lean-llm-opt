import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/demand.csv'
fixed_cost_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/fixed_cost.csv'
transport_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP2/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
transport_df = pd.read_csv(transport_path, sep=',')
supplier_ids = fixed_cost_df['Unnamed: 0'].tolist()
customer_ids = demand_df['customer'].tolist()
fixed_cost = dict(zip(supplier_ids, fixed_cost_df['fixed_costs']))
demand = dict(zip(demand_df['customer'], demand_df['demand']))
transport_cost = transport_df.set_index('Unnamed: 0').to_dict('index')
m = gp.Model('UFLP')
y = m.addVars(supplier_ids, vtype=gp.GRB.BINARY)
x = m.addVars([(i, j) for i in supplier_ids for j in customer_ids], vtype=gp.GRB.CONTINUOUS, lb=0)
obj = gp.quicksum((fixed_cost[i] * y[i] for i in supplier_ids)) + gp.quicksum((transport_cost[i][j] * x[i, j] for i in supplier_ids for j in customer_ids))
m.setObjective(obj, gp.GRB.MINIMIZE)
for j in customer_ids:
    m.addConstr(gp.quicksum((x[i, j] for i in supplier_ids)) == demand[j])
for i in supplier_ids:
    for j in customer_ids:
        m.addConstr(x[i, j] <= demand[j] * y[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('Open suppliers:')
    for i in supplier_ids:
        if y[i].X > 0.5:
            print(f'  {i}')
    print('Shipment plan (non-zero quantities):')
    for i in supplier_ids:
        for j in customer_ids:
            if x[i, j].X > 1e-06:
                print(f'  {i} -> {j}: {x[i, j].X:.0f}')
else:
    print(f'No optimal solution found. Status: {m.status}')