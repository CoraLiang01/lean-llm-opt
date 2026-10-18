import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
customer_demand_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv'
supply_capacity_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv'
transportation_costs_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv'
df_demand = pd.read_csv(customer_demand_path, sep=',')
df_supply = pd.read_csv(supply_capacity_path, sep=',')
df_cost = pd.read_csv(transportation_costs_path, sep=',')
customers = df_demand['Customers'].tolist()
suppliers = df_supply['Supplier'].tolist()
df_cost = df_cost.set_index('Unnamed: 0')
suppliers = [s for s in suppliers if s in df_cost.index]
cost_columns = [c for c in df_cost.columns if c in customers]
df_cost = df_cost[cost_columns]
supply_cap = dict(zip(df_supply['Supplier'], df_supply['supply_capacity']))
demand_req = dict(zip(df_demand['Customers'], df_demand['demand']))
m = gp.Model('Transportation')
m.setParam('OutputFlag', 1)
x = m.addVars(suppliers, customers)
obj = gp.quicksum((df_cost.loc[s, c] * x[s, c] for s in suppliers for c in customers))
m.setObjective(obj, gp.GRB.MINIMIZE)
for s in suppliers:
    m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply_cap[s])
for c in customers:
    m.addConstr(gp.quicksum((x[s, c] for s in suppliers)) == demand_req[c])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.4f}')
    print('\nShipment plan:')
    for s in suppliers:
        for c in customers:
            val = x[s, c].X
            if val > 1e-06:
                print(f'  {s} -> {c}: {val:.4f}')
else:
    print(f'No optimal solution found. Status: {m.status}')