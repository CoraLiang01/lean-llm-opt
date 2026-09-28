import gurobipy as gp
import pandas as pd
import numpy as np
import re
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/demand.csv', sep=',')
demand_df['customer'] = demand_df['customer'].astype(str).str.strip()
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/fixed_cost.csv', sep=',')
fixed_cost_df['supplier'] = fixed_cost_df['Unnamed: 0'].astype(str).str.strip()
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP3/transportation_costs.csv', sep=',')
trans_cost_df['supplier'] = trans_cost_df['Unnamed: 0'].astype(str).str.strip()
suppliers = list(fixed_cost_df['supplier'])
trans_suppliers = set(trans_cost_df['supplier'])
if set(suppliers) != trans_suppliers:
    raise ValueError(f'Supplier mismatch between fixed_cost.csv and transportation_costs.csv: {set(suppliers) ^ trans_suppliers}')
customers = list(demand_df['customer'])
trans_customers = [col for col in trans_cost_df.columns if re.fullmatch('C\\d+', col)]
if set(customers) != set(trans_customers):
    raise ValueError(f'Customer mismatch between demand.csv and transportation_costs.csv: {set(customers) ^ set(trans_customers)}')
demand = dict(zip(demand_df['customer'], demand_df['demand']))
fixed_cost = dict(zip(fixed_cost_df['supplier'], fixed_cost_df['fixed_costs']))
trans_cost = {}
for _, row in trans_cost_df.iterrows():
    i = row['supplier']
    for j in customers:
        val = row[j]
        if pd.isnull(val):
            raise ValueError(f'Missing transportation cost for supplier {i}, customer {j}')
        trans_cost[i, j] = float(val)
m = gp.Model('UFLP')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)) + gp.quicksum((trans_cost[i, j] * x[i, j] for i in suppliers for j in customers)), gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= demand[j] * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}\n')
    print('--- Supplier Opening Decisions ---')
    for i in suppliers:
        print(f"Supplier {i}: {('OPEN' if y[i].X > 0.5 else 'closed')} (y={int(round(y[i].X))})")
    print('\n--- Supply Plan (x[i,j] > 0) ---')
    for i in suppliers:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'Supplier {i} -> Customer {j}: {x[i, j].X:.2f} units (cost per unit: {trans_cost[i, j]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')