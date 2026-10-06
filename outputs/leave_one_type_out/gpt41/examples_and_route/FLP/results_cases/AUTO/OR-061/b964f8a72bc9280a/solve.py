import gurobipy as gp
import pandas as pd
import numpy as np
import math
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv', sep=',')
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv', sep=',')
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv', sep=',')
suppliers = [str(s).strip() for s in fixed_cost_df['Unnamed: 0']]
customers = [str(c).strip() for c in demand_df['customer']]
trans_suppliers = [str(s).strip() for s in trans_cost_df['Unnamed: 0']]
trans_customers = [str(c).strip() for c in trans_cost_df.columns if c != 'Unnamed: 0']
if set(suppliers) != set(trans_suppliers):
    raise ValueError(f'Supplier mismatch between fixed_cost.csv and transportation_costs.csv: {suppliers} vs {trans_suppliers}')
if set(customers) != set(trans_customers):
    raise ValueError(f'Customer mismatch between demand.csv and transportation_costs.csv: {customers} vs {trans_customers}')
fixed_costs = {}
for _, row in fixed_cost_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    fixed_costs[supplier] = float(row['fixed_costs'])
demands = {}
for _, row in demand_df.iterrows():
    customer = str(row['customer']).strip()
    demands[customer] = int(row['demand'])
trans_costs = {}
for idx, row in trans_cost_df.iterrows():
    supplier = str(row['Unnamed: 0']).strip()
    for customer in customers:
        cost = float(row[customer])
        trans_costs[supplier, customer] = cost
m = gp.Model('UFLP_Superstore')
x = m.addVars(suppliers, customers, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(suppliers, vtype=gp.GRB.BINARY, name='')
total_fixed = gp.quicksum((fixed_costs[i] * y[i] for i in suppliers))
total_trans = gp.quicksum((trans_costs[i, j] * x[i, j] for i in suppliers for j in customers))
m.setObjective(total_fixed + total_trans, gp.GRB.MINIMIZE)
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in suppliers)) == demands[j], name=f'demand_{j}')
M = sum(demands.values())
for i in suppliers:
    for j in customers:
        m.addConstr(x[i, j] <= M * y[i], name=f'link_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Supplier Activation ---')
    for i in suppliers:
        status = 'OPEN' if y[i].X > 0.5 else 'CLOSED'
        print(f'  Supplier {i}: {status} (y={int(round(y[i].X))})  Fixed cost: {fixed_costs[i]:.2f}')
    print('\n--- Supply Plan (x[i,j]) ---')
    for i in suppliers:
        for j in customers:
            qty = x[i, j].X
            if qty > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {qty:.2f} units  (Transp. cost/unit: {trans_costs[i, j]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')