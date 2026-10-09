import gurobipy as gp
import pandas as pd
import numpy as np
import math
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', sep=',')
branches = demand_df['customer'].astype(str).str.strip().tolist()
demand = demand_df.set_index(demand_df['customer'].astype(str).str.strip())['demand'].to_dict()
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv', sep=',')
suppliers = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
fixed_costs = fixed_cost_df.set_index(fixed_cost_df['Unnamed: 0'].astype(str).str.strip())['fixed_costs'].to_dict()
trans_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv', sep=',')
trans_costs_df['Unnamed: 0'] = trans_costs_df['Unnamed: 0'].astype(str).str.strip()
trans_costs_df = trans_costs_df.set_index('Unnamed: 0')
if not set(suppliers).issubset(set(trans_costs_df.index)):
    raise ValueError('Some suppliers in fixed_cost.csv are missing from transportation_costs.csv')
if not set(branches).issubset(set(trans_costs_df.columns)):
    raise ValueError('Some branches in demand.csv are missing from transportation_costs.csv')
transportation_costs = {}
for i in suppliers:
    for j in branches:
        val = trans_costs_df.at[i, j]
        if pd.isnull(val):
            raise ValueError(f'Missing transportation cost for supplier {i}, branch {j}')
        transportation_costs[i, j] = float(val)
M = sum((demand[j] for j in branches))
m = gp.Model('UFLP_Superstore')
x = m.addVars(suppliers, branches, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
total_fixed = gp.quicksum((fixed_costs[i] * y[i] for i in suppliers))
total_transport = gp.quicksum((transportation_costs[i, j] * x[i, j] for i in suppliers for j in branches))
m.setObjective(total_fixed + total_transport, gp.GRB.MINIMIZE)
for j in branches:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
for i in suppliers:
    for j in branches:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Supplier Activation ---')
    for i in suppliers:
        status = 'OPEN' if y[i].X > 0.5 else 'CLOSED'
        print(f'  Supplier {i}: {status} (y={int(round(y[i].X))})  Fixed cost: {fixed_costs[i]:.2f}')
    print('\n--- Supply Plan (x[i,j]) ---')
    for j in branches:
        print(f'Branch {j} demand: {demand[j]}')
        for i in suppliers:
            qty = x[i, j].X
            if qty > 1e-06:
                print(f'  Supplied by {i}: {qty:.2f} units (Transp. cost/unit: {transportation_costs[i, j]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')