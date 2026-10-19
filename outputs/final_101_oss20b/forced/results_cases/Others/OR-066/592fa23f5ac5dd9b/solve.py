import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
demand_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv'
fixed_cost_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv'
transport_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
transport_df = pd.read_csv(transport_path, sep=',')
supplier_ids = fixed_cost_df['Unnamed: 0'].tolist()
customer_ids = demand_df['customer'].tolist()
fixed_costs = dict(zip(supplier_ids, fixed_cost_df['fixed_costs']))
demands = dict(zip(demand_df['customer'], demand_df['demand']))
transport_df.set_index('Unnamed: 0', inplace=True)
transport_df = transport_df[customer_ids]
M = sum(demands.values())
m = gp.Model('FixedChargeTransportation')
y = m.addVars(supplier_ids, vtype=gp.GRB.BINARY)
x = m.addVars(supplier_ids, customer_ids, vtype=gp.GRB.CONTINUOUS)
fixed_term = gp.quicksum((fixed_costs[i] * y[i] for i in supplier_ids))
transport_term = gp.quicksum((transport_df.loc[i, j] * x[i, j] for i in supplier_ids for j in customer_ids))
m.setObjective(fixed_term + transport_term, gp.GRB.MINIMIZE)
for j in customer_ids:
    m.addConstr(gp.quicksum((x[i, j] for i in supplier_ids)) == demands[j])
for i in supplier_ids:
    for j in customer_ids:
        m.addConstr(x[i, j] <= M * y[i])
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal objective value: {m.objVal:.4f}')
    print('\nSupplier activation and shipment plan:')
    for i in supplier_ids:
        print(f'  Supplier {i}: {('Activated' if y[i].X > 0.5 else 'Not activated')} (y={y[i].X:.0f})')
        for j in customer_ids:
            qty = x[i, j].X
            if qty > 1e-06:
                print(f'    Ship to {j}: {qty:.4f} units (cost per unit: {transport_df.loc[i, j]:.4f})')
else:
    print(f'No optimal solution found. Status: {m.status}')