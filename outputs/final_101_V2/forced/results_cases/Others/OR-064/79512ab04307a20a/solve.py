import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
customers = list(demand_df['customer'])
demand_dict = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
fixed_cost_df['supplier'] = fixed_cost_df['Unnamed: 0'].astype(str).str.strip()
suppliers = list(fixed_cost_df['supplier'])
fixed_cost_dict = dict(zip(fixed_cost_df['supplier'], fixed_cost_df['fixed_costs']))
trans_cost_df = pd.read_csv(transport_cost_path, sep=',')
trans_cost_df['supplier'] = trans_cost_df['Unnamed: 0'].astype(str).str.strip()
trans_cost_cols = [c for c in trans_cost_df.columns if c in customers]
if set(trans_cost_cols) != set(customers):
    raise ValueError('Mismatch between customers in demand.csv and columns in transportation_costs.csv')
transport_cost = {}
for idx, row in trans_cost_df.iterrows():
    supplier = str(row['supplier']).strip()
    for customer in customers:
        cost = row[customer]
        if pd.isnull(cost):
            raise ValueError(f'Missing transportation cost for supplier {supplier}, customer {customer}')
        transport_cost[supplier, customer] = float(cost)
trans_suppliers = set(trans_cost_df['supplier'])
if set(suppliers) != trans_suppliers:
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
I = suppliers
J = customers
M = sum((demand_dict[j] for j in J))
m = gp.Model('UFLP_Supplier_Selection')
y = m.addVars(I, vtype=gp.GRB.BINARY, name='')
x = m.addVars(I, J, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
fixed_cost_term = gp.quicksum((fixed_cost_dict[i] * y[i] for i in I))
transport_cost_term = gp.quicksum((transport_cost[i, j] * x[i, j] for i in I for j in J))
m.setObjective(fixed_cost_term + transport_cost_term, gp.GRB.MINIMIZE)
for j in J:
    m.addConstr(gp.quicksum((x[i, j] for i in I)) == demand_dict[j], name=f'demand_{j}')
for i in I:
    for j in J:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Supplier Opening Decisions ---')
    for i in I:
        print(f"  Supplier {i}: {('OPEN' if y[i].X > 0.5 else 'closed')} (y={int(round(y[i].X))})")
    print('\n--- Supply Plan (nonzero flows) ---')
    for i in I:
        for j in J:
            if x[i, j].X > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')