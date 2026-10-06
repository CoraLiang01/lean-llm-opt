import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
customers = list(demand_df['customer'])
demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
fixed_cost_df['warehouse'] = fixed_cost_df['Unnamed: 0'].astype(str).str.strip()
warehouses = list(fixed_cost_df['warehouse'])
fixed_costs = dict(zip(fixed_cost_df['warehouse'], fixed_cost_df['fixed_costs']))
trans_cost_df = pd.read_csv(transport_cost_path, sep=',')
trans_cost_df['warehouse'] = trans_cost_df['Unnamed: 0'].astype(str).str.strip()
if set(warehouses) != set(trans_cost_df['warehouse']):
    raise ValueError('Mismatch between warehouses in fixed_cost.csv and transportation_costs.csv')
if not set(customers).issubset(set(trans_cost_df.columns)):
    raise ValueError('Some customers in demand.csv are missing from transportation_costs.csv columns')
transport_cost = {}
for _, row in trans_cost_df.iterrows():
    i = row['warehouse']
    for j in customers:
        transport_cost[i, j] = float(row[j])
m = gp.Model('UFLP_Bandcamp')
x = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
y = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_costs[i] * y[i] for i in warehouses)) + gp.quicksum((transport_cost[i, j] * x[i, j] for i in warehouses for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in warehouses)) == demand[j], name=f'demand_{j}')
for i in warehouses:
    for j in customers:
        m.addConstr(x[i, j] <= demand[j] * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Warehouse Activation ---')
    for i in warehouses:
        print(f"Warehouse {i}: {('OPEN' if y[i].X > 0.5 else 'CLOSED')} (y={int(round(y[i].X))})")
    print('\n--- Shipment Plan (x[i,j]) ---')
    for i in warehouses:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'  {x[i, j].X:.2f} units from Warehouse {i} to Customer {j}')
else:
    print(f'No optimal solution found. Status: {m.status}')