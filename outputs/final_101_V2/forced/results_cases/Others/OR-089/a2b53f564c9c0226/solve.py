import gurobipy as gp
import pandas as pd
import numpy as np
service_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv'
fixed_costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv'
df_service = pd.read_csv(service_costs_path, sep=',')
df_fixed = pd.read_csv(fixed_costs_path, sep=',')
centres = df_fixed['Service Center'].astype(str).tolist()
customers = df_service['Customer'].astype(str).tolist()
expected_sc_cols = centres
for sc in expected_sc_cols:
    if sc not in df_service.columns:
        raise KeyError(f"Service centre column '{sc}' missing in service costs CSV.")
fixed_opening_cost = dict(zip(df_fixed['Service Center'].astype(str), df_fixed['Fixed Opening Cost']))
service_cost = {}
for _, row in df_service.iterrows():
    cust = str(row['Customer'])
    service_cost[cust] = {}
    for sc in centres:
        val = row[sc]
        if pd.isnull(val):
            raise ValueError(f'Missing service cost for customer {cust}, centre {sc}')
        service_cost[cust][sc] = float(val)
m = gp.Model('UFLP_Capacitated')
y = m.addVars(centres, vtype=gp.GRB.BINARY, name='')
x = m.addVars(customers, centres, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_opening_cost[j] * y[j] for j in centres)) + gp.quicksum((service_cost[i][j] * x[i, j] for i in customers for j in centres)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in centres)) == 1 for i in customers), name='')
m.addConstrs((x[i, j] <= y[j] for i in customers for j in centres), name='')
m.addConstrs((gp.quicksum((x[i, j] for i in customers)) <= 4 for j in centres), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('\n--- Opened Centres ---')
    for j in centres:
        if y[j].X > 0.5:
            print(f'  {j}: OPEN (Fixed cost: {fixed_opening_cost[j]:.2f})')
    print('\n--- Customer Assignments ---')
    for i in customers:
        for j in centres:
            if x[i, j].X > 0.5:
                print(f'  Customer {i} assigned to {j} (Service cost: {service_cost[i][j]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')