import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
cost_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/cost.csv'
warehouse_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/warehouse.csv'
demand_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP4/demand.csv'
cost_df = pd.read_csv(cost_path, sep=',')
warehouse_df = pd.read_csv(warehouse_path, sep=',')
demand_df = pd.read_csv(demand_path, sep=',')
warehouses = warehouse_df['Warehouse ID'].tolist()
customers = demand_df['Customer ID'].tolist()
fixed_costs = warehouse_df.set_index('Warehouse ID')['Fixed_Cost'].to_dict()
capacities = warehouse_df.set_index('Warehouse ID')['Capacity'].to_dict()
demands = demand_df.set_index('Customer ID')['Demand'].to_dict()
t = {w: {c: cost_df.loc[cost_df['Warehouse ID'] == w, c].iloc[0] for c in customers} for w in warehouses}
m = gp.Model('CFLP')
y_vars = {w: m.addVar(vtype=gp.GRB.BINARY) for w in warehouses}
x_vars = {(w, c): m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0) for w in warehouses for c in customers}
for c in customers:
    m.addConstr(gp.quicksum((x_vars[w, c] for w in warehouses)) == demands[c])
for w in warehouses:
    m.addConstr(gp.quicksum((x_vars[w, c] for c in customers)) <= capacities[w] * y_vars[w])
m.setObjective(gp.quicksum((fixed_costs[w] * y_vars[w] for w in warehouses)) + gp.quicksum((t[w][c] * x_vars[w, c] for w in warehouses for c in customers)), sense=gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal objective value: {m.objVal:.2f}')
    print('\nOpen warehouses:')
    for w in warehouses:
        if y_vars[w].X > 0.5:
            print(f'  {w} (fixed cost: {fixed_costs[w]}, capacity: {capacities[w]})')
    print('\nShipment plan (non-zero shipments):')
    for (w, c), var in x_vars.items():
        if var.X > 1e-06:
            print(f'  {w} -> {c}: {var.X:.2f} units (cost per unit: {t[w][c]})')
else:
    print(f'No optimal solution found. Status: {m.status}')