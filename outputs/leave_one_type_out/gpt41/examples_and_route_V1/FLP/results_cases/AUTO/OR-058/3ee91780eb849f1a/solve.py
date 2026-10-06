import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
customers = demand_df['customer'].tolist()
demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
fixed_cost_df['supplier'] = fixed_cost_df['Unnamed: 0'].astype(str).str.strip()
suppliers = fixed_cost_df['supplier'].tolist()
fixed_costs = dict(zip(fixed_cost_df['supplier'], fixed_cost_df['fixed_costs']))
trans_cost_df = pd.read_csv(transport_cost_path, sep=',')
trans_cost_df['supplier'] = trans_cost_df['Unnamed: 0'].astype(str).str.strip()
if set(suppliers) != set(trans_cost_df['supplier']):
    raise ValueError('Mismatch between suppliers in fixed_cost.csv and transportation_costs.csv')
if set(customers) != set([c for c in trans_cost_df.columns if c.startswith('C')]):
    raise ValueError('Mismatch between customers in demand.csv and transportation_costs.csv')
transport_costs = {}
for _, row in trans_cost_df.iterrows():
    i = row['supplier']
    for j in customers:
        transport_costs[i, j] = float(row[j])
I = suppliers
J = customers
M = sum((demand[j] for j in J))
m = gp.Model('UFLP_Adidas_Supplier_Selection')
x = m.addVars(I, J, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(I, vtype=gp.GRB.BINARY, name='')
total_fixed = gp.quicksum((fixed_costs[i] * y[i] for i in I))
total_transport = gp.quicksum((transport_costs[i, j] * x[i, j] for i in I for j in J))
m.setObjective(total_fixed + total_transport, gp.GRB.MINIMIZE)
for j in J:
    m.addConstr(gp.quicksum((x[i, j] for i in I)) == demand[j], name=f'demand_{j}')
for i in I:
    for j in J:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('\n--- Supplier Activation ---')
    for i in I:
        print(f"Supplier {i}: {('OPEN' if y[i].X > 0.5 else 'CLOSED')} (y={int(round(y[i].X))})")
    print('\n--- Shipment Plan (x[i,j]) ---')
    for i in I:
        for j in J:
            if x[i, j].X > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')