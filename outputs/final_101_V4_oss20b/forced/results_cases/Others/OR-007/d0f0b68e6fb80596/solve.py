import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
customer_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv'
supply_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv'
cost_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv'
customer_df = pd.read_csv(customer_path)
supply_df = pd.read_csv(supply_path)
cost_df = pd.read_csv(cost_path)
customers = customer_df['customer'].tolist()
warehouses = supply_df['region'].tolist()
demand = dict(zip(customer_df['customer'], customer_df['demand']))
supply = dict(zip(supply_df['region'], supply_df['supply_capacity']))
cost_df.set_index('Unnamed: 0', inplace=True)
cost_df = cost_df[customers]
cost = {(s, d): cost_df.loc[s, d] for s in warehouses for d in customers}
m = gp.Model('GreenMart_Transportation')
x = m.addVars([(s, d) for s in warehouses for d in customers], vtype=gp.GRB.CONTINUOUS)
m.setObjective(gp.quicksum((cost[s, d] * x[s, d] for s in warehouses for d in customers)), gp.GRB.MINIMIZE)
for d in customers:
    m.addConstr(gp.quicksum((x[s, d] for s in warehouses)) == demand[d])
for s in warehouses:
    m.addConstr(gp.quicksum((x[s, d] for d in customers)) <= supply[s])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total transportation cost: {m.objVal:.4f}')
    print('\nShipment plan (units):')
    for s in warehouses:
        for d in customers:
            qty = x[s, d].X
            if qty > 1e-06:
                print(f'  {s} -> {d}: {qty:.4f}')
else:
    print(f'Optimization did not find an optimal solution. Status: {m.status}')