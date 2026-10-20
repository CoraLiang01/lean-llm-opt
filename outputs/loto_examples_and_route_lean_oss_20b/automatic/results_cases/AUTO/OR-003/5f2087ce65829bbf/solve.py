import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
customer_df = pd.read_csv('/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP3/customer_demand.csv', sep=',')
supply_df = pd.read_csv('/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP3/supply_capacity.csv', sep=',')
transport_df = pd.read_csv('/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/TP_testing/TP3/transportation_costs.csv', sep=',')
suppliers = supply_df['Unnamed: 0'].tolist()
cap = dict(zip(suppliers, supply_df['supply_capacity']))
customers = customer_df['customer'].tolist()
dem = dict(zip(customers, customer_df['demand']))
transport_df_indexed = transport_df.set_index('Unnamed: 0')
cost = {s: dict(zip(customers, transport_df_indexed.loc[s, customers].values)) for s in suppliers}
m = gp.Model('Transportation')
x = m.addVars(suppliers, customers)
m.addConstrs((gp.quicksum((x[s, c] for c in customers)) <= cap[s] for s in suppliers))
m.addConstrs((gp.quicksum((x[s, c] for s in suppliers)) == dem[c] for c in customers))
obj = gp.quicksum((cost[s][c] * x[s, c] for s in suppliers for c in customers))
m.setObjective(obj, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.2f}')
    print('\nTransportation plan (non‑zero shipments):')
    for s in suppliers:
        for c in customers:
            val = x[s, c].X
            if val > 1e-06:
                print(f'{s} -> {c}: {val:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')