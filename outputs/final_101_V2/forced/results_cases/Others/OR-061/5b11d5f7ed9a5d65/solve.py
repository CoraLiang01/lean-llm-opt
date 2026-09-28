import gurobipy as gp
import pandas as pd
import numpy as np
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', sep=',')
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv', sep=',')
transport_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv', sep=',')
suppliers = [str(s).strip() for s in fixed_cost_df['Unnamed: 0']]
branches = [str(c).strip() for c in demand_df['customer']]
transport_suppliers = [str(s).strip() for s in transport_df['Unnamed: 0']]
transport_branches = [str(c).strip() for c in transport_df.columns if c != 'Unnamed: 0']
if set(suppliers) != set(transport_suppliers):
    raise ValueError(f'Supplier mismatch between fixed_cost.csv and transportation_costs.csv: {set(suppliers)} vs {set(transport_suppliers)}')
if set(branches) != set(transport_branches):
    raise ValueError(f'Branch/customer mismatch between demand.csv and transportation_costs.csv: {set(branches)} vs {set(transport_branches)}')
fixed_cost = {str(row['Unnamed: 0']).strip(): float(row['fixed_costs']) for _, row in fixed_cost_df.iterrows()}
demand = {str(row['customer']).strip(): int(row['demand']) for _, row in demand_df.iterrows()}
transport_cost = {}
for _, row in transport_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    for branch in branches:
        transport_cost[supplier, branch] = float(row[branch])
m = gp.Model('UFLP_Superstore')
x = m.addVars(suppliers, branches, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
total_fixed = gp.quicksum((fixed_cost[i] * y[i] for i in suppliers))
total_transport = gp.quicksum((transport_cost[i, j] * x[i, j] for i in suppliers for j in branches))
m.setObjective(total_fixed + total_transport, gp.GRB.MINIMIZE)
for j in branches:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
M = sum(demand.values())
for i in suppliers:
    for j in branches:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Supplier Activation ---')
    for i in suppliers:
        status = 'OPEN' if y[i].X > 0.5 else 'CLOSED'
        print(f'  Supplier {i}: {status} (y={int(round(y[i].X))})')
    print('\n--- Supply Plan (x[i,j]) ---')
    for j in branches:
        print(f'Branch {j} demand: {demand[j]}')
        for i in suppliers:
            qty = x[i, j].X
            if qty > 1e-06:
                print(f'  Supplied by {i}: {qty:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')