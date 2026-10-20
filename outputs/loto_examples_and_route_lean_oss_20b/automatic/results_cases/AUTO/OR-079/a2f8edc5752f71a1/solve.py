import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
facility_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/facility_costs.csv'
shipping_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/shipping_costs.csv'
demand_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP11/demand_requirements.csv'
df_fac = pd.read_csv(facility_path, sep=',')
df_ship = pd.read_csv(shipping_path, sep=',')
df_dem = pd.read_csv(demand_path, sep=',')
facilities = df_fac['Facility'].tolist()
destinations = df_dem['Destination'].tolist()
fixed_cost = dict(zip(df_fac['Facility'], df_fac['FixedCost']))
capacity = dict(zip(df_fac['Facility'], df_fac['Capacity']))
demand = dict(zip(df_dem['Destination'], df_dem['Demand']))
shipping_cost = {}
for _, row in df_ship.iterrows():
    origin = row['Origin']
    for dest in destinations:
        shipping_cost[origin, dest] = row[dest]
m = gp.Model('CFLP')
y = m.addVars(facilities, vtype=gp.GRB.BINARY)
x = m.addVars(facilities, destinations)
obj = gp.quicksum((fixed_cost[i] * y[i] for i in facilities)) + gp.quicksum((shipping_cost[i, j] * x[i, j] for i in facilities for j in destinations))
m.setObjective(obj, gp.GRB.MINIMIZE)
for j in destinations:
    m.addConstr(gp.quicksum((x[i, j] for i in facilities)) == demand[j])
for i in facilities:
    m.addConstr(gp.quicksum((x[i, j] for j in destinations)) <= capacity[i] * y[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('Facility activation:')
    for i in facilities:
        if y[i].X > 0.5:
            print(f'  {i}: activated (fixed cost {fixed_cost[i]})')
    print('\nShipping plan:')
    for i in facilities:
        for j in destinations:
            if x[i, j].X > 1e-06:
                print(f'  Ship {x[i, j].X:.2f} units from {i} to {j} (cost {shipping_cost[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')