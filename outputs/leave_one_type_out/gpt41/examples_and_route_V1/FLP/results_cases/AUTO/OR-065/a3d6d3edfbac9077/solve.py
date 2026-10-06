import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
customers = demand_df['customer'].tolist()
demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
fixed_cost_df['facility'] = fixed_cost_df['Unnamed: 0'].astype(str).str.strip()
facilities = fixed_cost_df['facility'].tolist()
fixed_costs = dict(zip(fixed_cost_df['facility'], fixed_cost_df['fixed_costs']))
trans_cost_df = pd.read_csv(transport_cost_path, sep=',')
trans_cost_df['facility'] = trans_cost_df['Unnamed: 0'].astype(str).str.strip()
trans_facilities = trans_cost_df['facility'].tolist()
if set(facilities) != set(trans_facilities):
    raise ValueError(f'Mismatch in facilities between fixed_cost.csv and transportation_costs.csv: {set(facilities)} vs {set(trans_facilities)}')
trans_customers = [col for col in trans_cost_df.columns if col not in ['Unnamed: 0', 'facility']]
if set(customers) != set(trans_customers):
    raise ValueError(f'Mismatch in customers between demand.csv and transportation_costs.csv: {set(customers)} vs {set(trans_customers)}')
transport_cost = {}
for _, row in trans_cost_df.iterrows():
    i = row['facility']
    transport_cost[i] = {}
    for j in customers:
        transport_cost[i][j] = float(row[j])
m = gp.Model('UFLP_Bandcamp')
x = m.addVars(facilities, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(facilities, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_costs[i] * y[i] for i in facilities)) + gp.quicksum((transport_cost[i][j] * x[i, j] for i in facilities for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in facilities)) == demand[j], name=f'demand_{j}')
for i in facilities:
    for j in customers:
        m.addConstr(x[i, j] <= demand[j] * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Facility Activation ---')
    for i in facilities:
        print(f"  Facility {i}: {('OPEN' if y[i].X > 0.5 else 'CLOSED')} (y={int(round(y[i].X))})")
    print('\n--- Shipment Plan (x[i,j]) ---')
    for i in facilities:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'  From {i} to {j}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')