import gurobipy as gp
import pandas as pd
import numpy as np
fixed_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/service_centers_fixed_costs.csv', sep=',')
fixed_costs_df['Service Center'] = fixed_costs_df['Service Center'].astype(str).str.strip()
service_costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP13/expanded_customer_service_costs.csv', sep=',')
service_costs_df['Customer'] = service_costs_df['Customer'].astype(str).str.strip()
centres = list(fixed_costs_df['Service Center'])
if len(centres) != 10:
    raise ValueError(f'Expected 10 service centres, got {len(centres)}: {centres}')
customers = list(service_costs_df['Customer'])
if len(customers) != 15:
    raise ValueError(f'Expected 15 customers, got {len(customers)}: {customers}')
fixed_cost = {}
for (idx, row) in fixed_costs_df.iterrows():
    centre = str(row['Service Center']).strip()
    if centre not in centres:
        raise ValueError(f'Service centre {centre} in fixed_costs_df not in expected centre list.')
    fixed_cost[centre] = float(row['Fixed Opening Cost'])
service_cost = {}
for (i, row) in service_costs_df.iterrows():
    cust = str(row['Customer']).strip()
    if cust not in customers:
        raise ValueError(f'Customer {cust} in service_costs_df not in expected customer list.')
    for centre in centres:
        if centre not in row:
            raise ValueError(f'Centre {centre} not found as column in service_costs_df.')
        service_cost[cust, centre] = float(row[centre])
m = gp.Model('UFLP_Capacitated')
y = m.addVars(centres, vtype=gp.GRB.BINARY, name='')
x = m.addVars(customers, centres, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[j] * y[j] for j in centres)) + gp.quicksum((service_cost[i, j] * x[i, j] for i in customers for j in centres)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in centres)) == 1 for i in customers), name='')
m.addConstrs((x[i, j] <= y[j] for i in customers for j in centres), name='')
m.addConstrs((gp.quicksum((x[i, j] for i in customers)) <= 4 for j in centres), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}\n')
    print('Opened Centres:')
    for j in centres:
        if y[j].X > 0.5:
            assigned_customers = [i for i in customers if x[i, j].X > 0.5]
            print(f"  {j}: Opened (Fixed cost: {fixed_cost[j]:.2f}), Assigned customers: {', '.join(assigned_customers)}")
    print('\nCustomer Assignments:')
    for i in customers:
        for j in centres:
            if x[i, j].X > 0.5:
                print(f'  Customer {i} assigned to {j} (Service cost: {service_cost[i, j]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')