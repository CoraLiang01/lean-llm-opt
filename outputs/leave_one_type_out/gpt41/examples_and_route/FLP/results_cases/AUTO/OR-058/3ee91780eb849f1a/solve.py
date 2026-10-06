import gurobipy as gp
import pandas as pd
import numpy as np
import math
demand_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv', sep=',')
fixed_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv', sep=',')
trans_cost_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv', sep=',')
customers = demand_df['customer'].astype(str).str.strip().tolist()
demand = dict(zip(demand_df['customer'].astype(str).str.strip(), demand_df['demand']))
suppliers = fixed_cost_df['Unnamed: 0'].astype(str).str.strip().tolist()
fixed_costs = dict(zip(fixed_cost_df['Unnamed: 0'].astype(str).str.strip(), fixed_cost_df['fixed_costs']))
trans_cost_df = trans_cost_df.copy()
trans_cost_df['Unnamed: 0'] = trans_cost_df['Unnamed: 0'].astype(str).str.strip()
trans_cost_df.set_index('Unnamed: 0', inplace=True)
trans_cost_df.columns = [str(c).strip() for c in trans_cost_df.columns]
missing_suppliers = set(suppliers) - set(trans_cost_df.index)
missing_customers = set(customers) - set(trans_cost_df.columns)
if missing_suppliers:
    raise ValueError(f'Missing suppliers in transportation_costs.csv: {missing_suppliers}')
if missing_customers:
    raise ValueError(f'Missing customers in transportation_costs.csv: {missing_customers}')
transportation_costs = {}
for i in suppliers:
    for j in customers:
        val = trans_cost_df.at[i, j]
        if pd.isnull(val):
            raise ValueError(f'Missing transportation cost for supplier {i}, customer {j}')
        transportation_costs[i, j] = float(val)
I = suppliers
J = customers
M = sum((demand[j] for j in J))
m = gp.Model('UFLP_Adidas_Supplier_Selection')
x = m.addVars(I, J, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
y = m.addVars(I, vtype=gp.GRB.BINARY, name='')
total_fixed = gp.quicksum((fixed_costs[i] * y[i] for i in I))
total_transport = gp.quicksum((transportation_costs[i, j] * x[i, j] for i in I for j in J))
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
        print(f"  Supplier {i}: {('OPEN' if y[i].X > 0.5 else 'CLOSED')} (y={int(round(y[i].X))})")
    print('\n--- Shipments (x[i,j]) ---')
    for i in I:
        for j in J:
            if x[i, j].X > 1e-06:
                print(f'  Supplier {i} -> Customer {j}: {x[i, j].X:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')