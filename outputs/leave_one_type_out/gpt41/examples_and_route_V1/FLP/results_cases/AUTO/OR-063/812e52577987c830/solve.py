import gurobipy as gp
import pandas as pd
import numpy as np
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv', sep=',')
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
customers = demand_df['customer'].tolist()
demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv', sep=',')
fixed_df['Unnamed: 0'] = fixed_df['Unnamed: 0'].astype(str).str.strip()
warehouses = fixed_df['Unnamed: 0'].tolist()
fixed_costs = dict(zip(fixed_df['Unnamed: 0'], fixed_df['fixed_costs']))
trans_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv', sep=',')
trans_df['Unnamed: 0'] = trans_df['Unnamed: 0'].astype(str).str.strip()
if set(warehouses) != set(trans_df['Unnamed: 0']):
    raise ValueError('Mismatch in warehouse identifiers between fixed_cost.csv and transportation_costs.csv')
if set(customers) != set([c for c in trans_df.columns if c != 'Unnamed: 0']):
    raise ValueError('Mismatch in customer identifiers between demand.csv and transportation_costs.csv')
transportation_costs = {}
for _, row in trans_df.iterrows():
    w = str(row['Unnamed: 0']).strip()
    for c in customers:
        transportation_costs[w, c] = float(row[c])
m = gp.Model('UFLP_Bandcamp')
x = m.addVars(warehouses, customers, lb=0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
fixed_cost_term = gp.quicksum((fixed_costs[i] * y[i] for i in warehouses))
transport_cost_term = gp.quicksum((transportation_costs[i, j] * x[i, j] for i in warehouses for j in customers))
m.setObjective(fixed_cost_term + transport_cost_term, gp.GRB.MINIMIZE)
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
    print('\n--- Shipment Plan (x[i,j] > 0) ---')
    for i in warehouses:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'  {x[i, j].X:.2f} units from Warehouse {i} to Customer {j} (cost per unit: {transportation_costs[i, j]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')