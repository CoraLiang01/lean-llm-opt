import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
fixed_cost_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
transport_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
transport_df = pd.read_csv(transport_path, sep=',')
demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost = dict(zip(fixed_cost_df['Unnamed: 0'], fixed_cost_df['fixed_costs']))
transport_cost = {}
supplier_ids = transport_df['Unnamed: 0'].tolist()
customer_ids = [col for col in transport_df.columns if col != 'Unnamed: 0']
for _, row in transport_df.iterrows():
    supplier = row['Unnamed: 0']
    for cust in customer_ids:
        transport_cost[supplier, cust] = row[cust]
suppliers = list(fixed_cost.keys())
customers = list(demand.keys())
M = sum(demand.values())
m = gp.Model('UFLP')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY)
x = m.addVars([(i, j) for i in suppliers for j in customers], vtype=gp.GRB.CONTINUOUS, lb=0.0)
obj_fixed = gp.quicksum((fixed_cost[i] * y[i] for i in suppliers))
obj_trans = gp.quicksum((transport_cost[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(obj_fixed + obj_trans, sense=gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j])
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('Open suppliers:')
    for i in suppliers:
        if y[i].X > 0.5:
            print(f'  {i} (fixed cost: {fixed_cost[i]:.2f})')
    print('\nShipment plan:')
    for j in customers:
        shipped = [(i, x[i, j].X) for i in suppliers if x[i, j].X > 1e-06]
        if shipped:
            print(f'  Customer {j} (demand {demand[j]}):')
            for i, qty in shipped:
                print(f'    from {i}: {qty:.2f} units (transport cost {transport_cost[i, j]:.2f} per unit)')
else:
    print(f'No optimal solution found. Status: {m.status}')