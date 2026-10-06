import gurobipy as gp
import pandas as pd
import numpy as np
import math
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', sep=',')
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
customers = list(demand_df['customer'])
demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv', sep=',')
fixed_df['Unnamed: 0'] = fixed_df['Unnamed: 0'].astype(str).str.strip()
suppliers = list(fixed_df['Unnamed: 0'])
fixed_costs = dict(zip(fixed_df['Unnamed: 0'], fixed_df['fixed_costs']))
trans_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv', sep=',')
trans_df['Unnamed: 0'] = trans_df['Unnamed: 0'].astype(str).str.strip()
trans_suppliers = list(trans_df['Unnamed: 0'])
trans_customers = [col for col in trans_df.columns if col != 'Unnamed: 0']
if set(suppliers) != set(trans_suppliers):
    raise ValueError(f'Mismatch in supplier IDs between fixed_cost.csv and transportation_costs.csv: {set(suppliers)} vs {set(trans_suppliers)}')
if set(customers) != set(trans_customers):
    raise ValueError(f'Mismatch in customer IDs between demand.csv and transportation_costs.csv: {set(customers)} vs {set(trans_customers)}')
transportation_costs = {}
for _, row in trans_df.iterrows():
    i = row['Unnamed: 0']
    for j in customers:
        transportation_costs[i, j] = float(row[j])
F = suppliers
C = customers
m = gp.Model('UFLP_Superstore')
x = m.addVars(F, C, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(F, vtype=gp.GRB.BINARY, name='')
fixed_term = gp.quicksum((fixed_costs[i] * y[i] for i in F))
trans_term = gp.quicksum((transportation_costs[i, j] * x[i, j] for i in F for j in C))
m.setObjective(fixed_term + trans_term, gp.GRB.MINIMIZE)
for j in C:
    m.addConstr(gp.quicksum((x[i, j] for i in F)) == demand[j], name=f'demand_{j}')
M = sum(demand.values())
for i in F:
    for j in C:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Supplier Activation ---')
    for i in F:
        status = 'OPEN' if y[i].X > 0.5 else 'CLOSED'
        print(f'  Supplier {i}: {status} (y={int(round(y[i].X))})  Fixed cost: {fixed_costs[i]:.2f}')
    print('\n--- Supply Plan (x[i,j]) ---')
    for j in C:
        print(f'Customer {j} (demand={demand[j]}):')
        for i in F:
            qty = x[i, j].X
            if qty > 1e-06:
                print(f'  Supplied by {i}: {qty:.2f} units  (transportation cost per unit: {transportation_costs[i, j]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')