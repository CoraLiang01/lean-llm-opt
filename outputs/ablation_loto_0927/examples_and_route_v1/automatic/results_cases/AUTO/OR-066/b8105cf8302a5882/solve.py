import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/demand.csv', sep=',')
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/fixed_cost.csv', sep=',')
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP10/transportation_costs.csv', sep=',')
suppliers_fc = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
suppliers_tc = trans_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
suppliers = sorted(set(suppliers_fc) | set(suppliers_tc))
customers_demand = demand_df['customer'].astype(str).str.strip().tolist()
customers_tc = [c for c in trans_cost_df.columns if c != 'Unnamed: 0']
customers = sorted(set(customers_demand) | set(customers_tc))
fixed_cost = {}
for (_, row) in fixed_cost_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    if supplier not in suppliers:
        continue
    fixed_cost[supplier] = float(row['fixed_costs'])
for s in suppliers:
    if s not in fixed_cost:
        raise ValueError(f'Missing fixed cost for supplier {s}')
demand = {}
for (_, row) in demand_df.iterrows():
    customer = str(row['customer']).strip()
    if customer not in customers:
        continue
    demand[customer] = int(row['demand'])
for c in customers:
    if c not in demand:
        raise ValueError(f'Missing demand for customer {c}')
trans_cost = {}
for (_, row) in trans_cost_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    if supplier not in suppliers:
        continue
    for customer in customers:
        if customer not in trans_cost_df.columns:
            raise ValueError(f'Missing transportation cost column for customer {customer}')
        cost = float(row[customer])
        trans_cost[supplier, customer] = cost
m = gp.Model('UFLP')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
total_fixed = gp.quicksum((fixed_cost[i] * y[i] for i in suppliers))
total_trans = gp.quicksum((trans_cost[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(total_fixed + total_trans, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demand[j], name=f'demand_{j}')
M = sum(demand.values())
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('--- Supplier Activation ---')
    for i in suppliers:
        print(f"  Supplier {i}: {('ACTIVE' if y[i].X > 0.5 else 'inactive')} (y={int(round(y[i].X))})")
    print('\n--- Supply Plan (x[i,j]) ---')
    for i in suppliers:
        for j in customers:
            if x[i, j].X > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')