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
trans_cost_customers = [col for col in trans_cost_df.columns if col.startswith('C')]
if set(customers) != set(trans_cost_customers):
    raise ValueError(f'Mismatch between customers in demand.csv and transportation_costs.csv: {set(customers) ^ set(trans_cost_customers)}')
if set(suppliers) != set(trans_cost_df['supplier']):
    raise ValueError(f"Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv: {set(suppliers) ^ set(trans_cost_df['supplier'])}")
transport_cost = {}
for _, row in trans_cost_df.iterrows():
    i = str(row['supplier']).strip()
    for j in customers:
        transport_cost[i, j] = float(row[j])
if len(suppliers) != len(set(suppliers)):
    raise ValueError('Duplicate supplier IDs found.')
if len(customers) != len(set(customers)):
    raise ValueError('Duplicate customer IDs found.')
if not all((j in demand_dict for j in customers)):
    raise ValueError('Some customers missing demand data.')
if not all((i in fixed_cost_dict for i in suppliers)):
    raise ValueError('Some suppliers missing fixed cost data.')
for i in suppliers:
    for j in customers:
        if (i, j) not in transport_cost:
            raise ValueError(f'Missing transportation cost for supplier {i}, customer {j}.')
m = gp.Model('UFLP')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[i] * y[i] for i in suppliers))
transport_cost_expr = gp.quicksum((transport_cost[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand_dict[j], name=f'demand_{j}')
M = sum((demand_dict[j] for j in customers))
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i], name=f'activate_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Opened Suppliers ---')
    for i in suppliers:
        if y[i].X > 0.5:
            print(f'  {i}: OPEN (fixed cost = {fixed_cost_dict[i]:.2f})')
    print('\n--- Supply Plan (x[i,j] > 0) ---')
    for i in suppliers:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {x[i, j].X:.2f} units (cost per unit: {transport_cost[i, j]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')