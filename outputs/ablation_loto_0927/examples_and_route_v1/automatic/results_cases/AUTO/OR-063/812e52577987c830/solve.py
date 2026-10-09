import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP7/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
transport_df = pd.read_csv(transport_cost_path, sep=',')
customers = [str(c).strip() for c in demand_df['customer']]
warehouses = [str(w).strip() for w in fixed_cost_df['Unnamed: 0']]
transport_warehouses = [str(w).strip() for w in transport_df['Unnamed: 0']]
transport_customers = [str(c).strip() for c in transport_df.columns if c != 'Unnamed: 0']
if set(warehouses) != set(transport_warehouses):
    raise ValueError(f'Mismatch in warehouse identifiers between fixed_cost.csv and transportation_costs.csv: {set(warehouses)} vs {set(transport_warehouses)}')
if set(customers) != set(transport_customers):
    raise ValueError(f'Mismatch in customer identifiers between demand.csv and transportation_costs.csv: {set(customers)} vs {set(transport_customers)}')
demand = {}
for (_, row) in demand_df.iterrows():
    cust = str(row['customer']).strip()
    demand[cust] = int(row['demand'])
fixed_cost = {}
for (_, row) in fixed_cost_df.iterrows():
    wh = str(row['Unnamed: 0']).strip()
    fixed_cost[wh] = float(row['fixed_costs'])
transport_cost = {}
for (_, row) in transport_df.iterrows():
    wh = str(row['Unnamed: 0']).strip()
    for cust in customers:
        transport_cost[wh, cust] = float(row[cust])
m = gp.Model('UFLP_Bandcamp')
x = m.addVars(warehouses, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in warehouses)) + gp.quicksum((transport_cost[i, j] * x[i, j] for i in warehouses for j in customers)), gp.GRB.MINIMIZE)
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
    print('\n--- Customer Supply Plan ---')
    for j in customers:
        print(f'Customer {j}:')
        for i in warehouses:
            if x[i, j].X > 1e-06:
                print(f'  Supplied from {i}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')