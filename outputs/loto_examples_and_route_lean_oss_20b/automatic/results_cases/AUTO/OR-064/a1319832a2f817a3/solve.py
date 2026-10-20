import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
fixed_cost_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
transportation_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
transportation_df = pd.read_csv(transportation_path, sep=',')
customers = demand_df['customer'].tolist()
suppliers = fixed_cost_df['Unnamed: 0'].tolist()
demand_dict = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_dict = dict(zip(fixed_cost_df['Unnamed: 0'], fixed_cost_df['fixed_costs']))
transportation_df.set_index('Unnamed: 0', inplace=True)
total_demand = demand_df['demand'].sum()
m = gp.Model('UFLP')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY)
x = m.addVars(suppliers, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0)
obj = gp.quicksum((fixed_cost_dict[s] * y[s] for s in suppliers)) + gp.quicksum((transportation_df.loc[s, c] * x[s, c] for s in suppliers for c in customers))
m.setObjective(obj, gp.GRB.MINIMIZE)
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand_dict[c])
for s in suppliers:
    for c in customers:
        m.addConstr(x[s, c] <= total_demand * y[s])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    opened_suppliers = [s for s in suppliers if y[s].X > 0.5]
    print(f'Number of suppliers opened: {len(opened_suppliers)}')
    print('\nOpened suppliers and shipment amounts:')
    for s in opened_suppliers:
        shipped = sum((x[s, c].X for c in customers))
        print(f'  Supplier {s}: shipped {shipped:.2f} units')
    print('\nShipment distribution to customers:')
    for c in customers:
        total_shipped = sum((x[s, c].X for s in suppliers))
        print(f'  Customer {c}: received {total_shipped:.2f} units')
else:
    print(f'Optimization did not find an optimal solution. Status: {m.status}')