import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/demand.csv'
fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/fixed_cost.csv'
transport_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP9/transportation_costs.csv'
demand_df = pd.read_csv(demand_path, sep=',')
customers = demand_df['customer'].astype(str).str.strip().tolist()
demand_dict = dict(zip(customers, demand_df['demand']))
fixed_cost_df = pd.read_csv(fixed_cost_path, sep=',')
warehouses = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
fixed_cost_dict = dict(zip(warehouses, fixed_cost_df['fixed_costs']))
trans_cost_df = pd.read_csv(transport_cost_path, sep=',')
trans_cost_df['Unnamed: 0'] = trans_cost_df['Unnamed: 0'].astype(str).str.strip()
trans_warehouses = trans_cost_df['Unnamed: 0'].tolist()
trans_customers = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
if set(warehouses) != set(trans_warehouses):
    raise ValueError(f'Mismatch between warehouses in fixed_cost.csv and transportation_costs.csv: {set(warehouses)} vs {set(trans_warehouses)}')
if set(customers) != set(trans_customers):
    raise ValueError(f'Mismatch between customers in demand.csv and transportation_costs.csv: {set(customers)} vs {set(trans_customers)}')
transport_cost = {}
for (_, row) in trans_cost_df.iterrows():
    i = row['Unnamed: 0']
    for j in customers:
        transport_cost[i, j] = float(row[j])
m = gp.Model('UFLP_Bandcamp')
y = m.addVars(warehouses, vtype=gp.GRB.BINARY, name='')
x = m.addVars(warehouses, customers, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
fixed_cost_expr = gp.quicksum((fixed_cost_dict[i] * y[i] for i in warehouses))
transport_cost_expr = gp.quicksum((transport_cost[i, j] * x[i, j] for i in warehouses for j in customers))
m.setObjective(fixed_cost_expr + transport_cost_expr, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in warehouses)) == demand_dict[j], name=f'demand_{j}')
for i in warehouses:
    for j in customers:
        m.addConstr(x[i, j] <= demand_dict[j] * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Warehouse Activation ---')
    for i in warehouses:
        print(f"Warehouse {i}: {('Activated' if y[i].X > 0.5 else 'Not Activated')} (y={int(round(y[i].X))})")
    print('\n--- Supply Plan ---')
    for j in customers:
        print(f'Customer {j}:')
        for i in warehouses:
            if x[i, j].X > 1e-06:
                print(f'  Supplied from {i}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')