import gurobipy as gp
import pandas as pd
import numpy as np
service_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
df_service = pd.read_csv(service_costs_path, sep=',')
df_fixed = pd.read_csv(fixed_costs_path, sep=',')
customers = df_service['Customer'].astype(str).tolist()
centres = df_fixed['Service Center'].astype(str).tolist()
expected_sc_cols = centres
for sc in expected_sc_cols:
    if sc not in df_service.columns:
        raise KeyError(f"Service centre column '{sc}' missing in service costs file.")
service_cost = {}
for _, row in df_service.iterrows():
    cust = str(row['Customer'])
    for sc in centres:
        service_cost[cust, sc] = float(row[sc])
fixed_opening_cost = {}
for _, row in df_fixed.iterrows():
    sc = str(row['Service Center'])
    fixed_opening_cost[sc] = float(row['Fixed Opening Cost'])
if set(customers) != set(df_service['Customer'].astype(str)):
    raise ValueError('Mismatch in customer identifiers.')
if set(centres) != set(df_fixed['Service Center'].astype(str)):
    raise ValueError('Mismatch in service centre identifiers.')
m = gp.Model('UFLP_Capacitated')
y = m.addVars(centres, vtype=gp.GRB.BINARY, name='')
x = m.addVars(customers, centres, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_opening_cost[j] * y[j] for j in centres)) + gp.quicksum((service_cost[i, j] * x[i, j] for i in customers for j in centres)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in centres)) == 1 for i in customers), name='')
m.addConstrs((x[i, j] <= y[j] for i in customers for j in centres), name='')
m.addConstrs((gp.quicksum((x[i, j] for i in customers)) <= 4 for j in centres), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Opened Centres ---')
    for j in centres:
        if y[j].X > 0.5:
            assigned_customers = [i for i in customers if x[i, j].X > 0.5]
            print(f"  {j}: OPEN (Fixed cost: {fixed_opening_cost[j]:.2f}) - Assigned customers: {', '.join(assigned_customers)}")
    print('\n--- Customer Assignments ---')
    for i in customers:
        for j in centres:
            if x[i, j].X > 0.5:
                print(f'  Customer {i} assigned to {j} (Service cost: {service_cost[i, j]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')