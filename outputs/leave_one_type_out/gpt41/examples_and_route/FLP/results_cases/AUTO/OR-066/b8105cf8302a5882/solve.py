import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
customers = list(demand_df['customer'])
demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
fixed_cost_df['supplier'] = fixed_cost_df['Unnamed: 0'].astype(str).str.strip()
suppliers = list(fixed_cost_df['supplier'])
fixed_costs = dict(zip(fixed_cost_df['supplier'], fixed_cost_df['fixed_costs']))
trans_cost_df = pd.read_csv(transport_cost_path, sep=',')
trans_cost_df['supplier'] = trans_cost_df['Unnamed: 0'].astype(str).str.strip()
if set(suppliers) != set(trans_cost_df['supplier']):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
if set(customers) - set(trans_cost_df.columns):
    raise ValueError('Some customers in demand.csv are missing in transportation_costs.csv columns')
transport_costs = {}
for _, row in trans_cost_df.iterrows():
    i = str(row['supplier'])
    for j in customers:
        transport_costs[i, j] = float(row[j])
I = suppliers
J = customers
m = gp.Model('UFLP')
y = m.addVars(I, vtype=gp.GRB.BINARY, name='')
x = m.addVars(I, J, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
fixed_cost_term = gp.quicksum((fixed_costs[i] * y[i] for i in I))
transport_cost_term = gp.quicksum((transport_costs[i, j] * x[i, j] for i in I for j in J))
m.setObjective(fixed_cost_term + transport_cost_term, gp.GRB.MINIMIZE)
for j in J:
    m.addConstr(gp.quicksum((x[i, j] for i in I)) == demand[j], name=f'demand_{j}')
M = sum(demand.values())
for i in I:
    for j in J:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('--- Supplier Activation ---')
    for i in I:
        print(f"  Supplier {i}: {('Activated' if y[i].X > 0.5 else 'Not Activated')} (y={int(round(y[i].X))})")
    print('\n--- Shipment Plan (x[i,j]) ---')
    for i in I:
        for j in J:
            if x[i, j].X > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')