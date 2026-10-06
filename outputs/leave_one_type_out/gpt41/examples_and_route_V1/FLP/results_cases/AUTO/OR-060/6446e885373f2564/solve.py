import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
customers = demand_df['customer'].astype(str).tolist()
demand = dict(zip(demand_df['customer'].astype(str), demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
suppliers = fixed_cost_df['Unnamed: 0'].astype(str).tolist()
fixed_costs = dict(zip(fixed_cost_df['Unnamed: 0'].astype(str), fixed_cost_df['fixed_costs']))
trans_cost_df = pd.read_csv(transport_cost_path, sep=',')
trans_cost_df['Unnamed: 0'] = trans_cost_df['Unnamed: 0'].astype(str)
trans_cost_cols = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
if set(customers) != set(trans_cost_cols):
    raise ValueError(f'Mismatch between customers in demand.csv and columns in transportation_costs.csv: {set(customers) ^ set(trans_cost_cols)}')
transport_costs = {}
for idx, row in trans_cost_df.iterrows():
    supplier = str(row['Unnamed: 0'])
    for customer in customers:
        transport_costs[supplier, customer] = float(row[customer])
if set(suppliers) != set(trans_cost_df['Unnamed: 0']):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
if set(customers) != set(demand_df['customer']):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv')
m = gp.Model('UFLP')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
fixed_cost_term = gp.quicksum((fixed_costs[i] * y[i] for i in suppliers))
transport_cost_term = gp.quicksum((transport_costs[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(fixed_cost_term + transport_cost_term, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
M = sum(demand.values())
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}\n')
    print('--- Supplier Opening Decisions ---')
    for i in suppliers:
        print(f"Supplier {i}: {('OPEN' if y[i].X > 0.5 else 'closed')} (y={int(round(y[i].X))})")
    print('\n--- Supply Plan (x[i,j] > 0) ---')
    for i in suppliers:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'Supplier {i} -> Customer {j}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')